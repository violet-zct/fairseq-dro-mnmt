# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn.functional as F
import torch
import math
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import logging

logger = logging.getLogger(__name__)


def convert_to_list(st, t):
    return list(map(t, st.strip().split(',')))


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.)
        smooth_loss.masked_fill_(pad_mask, 0.)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('hier_dro_label_smoothed_cross_entropy')
class HierarchicalDROLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing, outer_group_level,
                 dro_outer_alpha, dro_inner_beta,
                 baselines,
                 update_dro_freq, start_ft_steps):
        super().__init__(task)
        self.distributed_world_size = self.task.args.distributed_world_size
        self.eps = label_smoothing
        self.group_level = outer_group_level
        self.alpha = dro_outer_alpha
        self.beta = dro_inner_beta
        self.baselines = baselines
        self.update_freq = update_dro_freq

        self.device = torch.cuda.current_device()
        self.temp_idx = 0
        self.print_steps = 100

        self.update_steps = 0
        self.start_ft_steps = start_ft_steps
        self.EMA_alpha = 0.05

        if self.group_level == "source_lang":
            self.n_groups = len(task.data_manager.src_langs)
        elif self.group_level == "target_lang":
            self.n_groups = len(task.data_manager.tgt_langs)
        else:
            raise ValueError
        self.inner_groups = len(task.target_dictionary)

        self.initialize()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--outer-group-level', type=str, choices=['source_lang', 'target_lang'])
        parser.add_argument('--dro-outer-alpha', default=1., type=float, help='alpha value for the DRO loss.')
        parser.add_argument('--dro-inner-beta', default=1., type=float)
        parser.add_argument('--baselines', default=None, type=str, help='baseline loss values.')
        parser.add_argument('--update-dro-freq', default=1, type=int)
        parser.add_argument('--start-ft-steps', default=0, type=int)
        # fmt: on

    def initialize(self):
        logger.info("Outer group num = {}, Inner group num = {}".format(self.n_groups, self.inner_groups))
        if self.baselines is None:
            self.loss_baselines = torch.Tensor([0. for _ in range(self.n_groups)]).to(self.device)
        else:
            self.loss_baselines = torch.Tensor(convert_to_list(self.baselines, float)).to(self.device)
        self.register_buffer('outer_h_fun', torch.ones(self.n_groups))
        self.register_buffer('outer_sum_losses', torch.zeros(self.n_groups))  # historical loss sum over category
        self.register_buffer('outer_count_cat', torch.ones(self.n_groups))

        self.register_buffer('inner_h_fun', torch.ones(self.n_groups * self.inner_groups))
        self.register_buffer('inner_sum_losses', torch.zeros(self.n_groups * self.inner_groups))  # historical loss sum over category
        self.register_buffer('inner_count_cat', torch.ones(self.n_groups * self.inner_groups))

    def reset_history(self):
        self.outer_h_fun.fill_(1.)
        self.outer_sum_losses.fill_(0.)

    def update_mw(self):
        # version that uses EMA. (sum_losses is EMA running loss, count_cat is EMA running sum)
        past_losses = self.outer_sum_losses
        baselined_losses = past_losses - self.loss_baselines

        past_frac = self.outer_count_cat / self.outer_count_cat.sum()  # p_train_t
        #
        sorted_losses, sort_id = torch.sort(baselined_losses, descending=True)
        sorted_frac = past_frac[sort_id]
        cutoff_count = torch.sum(torch.cumsum(sorted_frac, 0) < self.alpha)
        if cutoff_count == len(sorted_frac):
            cutoff_count = len(sorted_frac) - 1
        self.outer_h_fun.fill_(0.1)
        self.outer_h_fun[sort_id[:cutoff_count]] = 1.0 / self.alpha
        leftover_mass = 1.0 - sorted_frac[:cutoff_count].sum().div(self.alpha)
        tiebreak_fraction = leftover_mass / sorted_frac[cutoff_count]  # check!
        self.outer_h_fun[sort_id[cutoff_count]] = tiebreak_fraction

        self.temp_idx += 1
        if self.temp_idx % self.print_steps == 0:
            logger.info("EMA past losses: {}".format(past_losses[0:self.n_groups]))
            # logger.info("Baseline losses: {}".format(baselined_losses[0:self.n_train_groups]))
            logger.info("EMA group fractions: {}".format(past_frac[0:self.n_groups]))
            logger.info("Group loss weights: {}".format(self.outer_h_fun[0:self.n_groups]))

    def update_mw_token(self):
        # version that uses EMA. (sum_losses is EMA running loss, count_cat is EMA running sum)
        baselined_losses = self.inner_sum_losses.view(self.n_groups, self.inner_groups)
        count_cat = self.inner_count_cat.view(self.n_groups, self.inner_groups)

        past_frac = count_cat / count_cat.sum(1, keepdim=True)  # p_train_t
        #
        sorted_losses, sort_id = torch.sort(baselined_losses, dim=-1, descending=True)

        sorted_frac = past_frac[torch.arange(self.n_groups), sort_id.transpose(0, 1)].transpose(0, 1)
        cutoff_count = torch.sum(torch.cumsum(sorted_frac, 1) < self.beta, dim=1)
        cutoff_count[cutoff_count == sorted_frac.size(1)] = sorted_frac.size(1) - 1

        inner_h_fun = self.inner_h_fun.new_full((self.n_groups, self.inner_groups), 0.1)
        leftover_masses = inner_h_fun.new_zeros(self.n_groups)
        for idx, cutoff in enumerate(cutoff_count):
            inner_h_fun[idx, sort_id[idx, :cutoff_count[idx]]] = 1.0 / self.beta
            leftover_masses[idx] = 1.0 - sorted_frac[idx, :cutoff_count[idx]].sum().div(self.beta)

        tiebreak_fraction = leftover_masses / sorted_frac.gather(1, cutoff_count.unsqueeze(1))  # check!
        inner_h_fun.scatter_(1, cutoff_count.unsqueeze(1), tiebreak_fraction)
        self.inner_h_fun = inner_h_fun.view(-1)

    def individual_losses(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)

        if self.eps > 0.0:
            losses = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
        else:
            losses = F.nll_loss(lprobs, target, ignore_index=self.padding_idx, reduction='none')
        return losses

    def retrieve_group_labels(self, sample):
        if self.group_level == "source_lang":
            index = sample["src_lang_id"]
        elif self.group_level == "target_lang":
            index = sample["tgt_lang_id"]
        else:
            index = None
        return index, sample['target']

    def compute_loss(self, model, sample):
        net_output = model(**sample['net_input'])
        mask = (sample['target'] != self.padding_idx).float()
        token_losses = self.individual_losses(model, net_output, sample)
        if isinstance(token_losses, tuple):
            nll_loss = token_losses[1].reshape_as(sample['target']).sum(1)
            token_losses = token_losses[0].reshape_as(sample['target']) * mask
        else:
            nll_loss = (token_losses.reshape_as(sample['target']) * mask).sum(1)
            token_losses = token_losses.reshape_as(sample['target']) * mask

        if not self.training:
            return nll_loss, token_losses.sum(), 0, 0, 0

        outer_index, inner_index = self.retrieve_group_labels(sample)
        offset_index = (inner_index + outer_index.unsqueeze(1) * self.inner_groups).view(-1)

        weights = self.inner_h_fun[offset_index].reshape_as(sample['target'])
        weighted_token_losses = token_losses * weights
        outer_ind_loss = weighted_token_losses.sum(1)

        zero_vec = torch.zeros(self.n_groups, device='cuda')  # G
        outer_group_losses = zero_vec.scatter_add(0, outer_index, outer_ind_loss)
        outer_group_counts = zero_vec.scatter_add(0, outer_index, mask.sum(1))

        inner_zero_vec = torch.zeros(self.inner_groups*self.n_groups, device='cuda')
        inner_group_losses = inner_zero_vec.scatter_add(0, offset_index, token_losses.view(-1))
        one_vec = torch.ones(offset_index.numel(), device='cuda')  # B
        inner_group_counts = inner_zero_vec.scatter_add(0, offset_index, one_vec)

        return nll_loss.sum(), outer_group_losses, outer_group_counts, inner_group_losses, inner_group_counts

    def simple_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        if hasattr(self, "start_ft_steps") and self.update_steps < self.start_ft_steps:
            if self.training:
                self.update_steps += 1
            net_output = model(**sample['net_input'])
            loss, nll_loss = self.simple_loss(model, net_output, sample, reduce=reduce)
            sample_size = sample['ntokens']
            logging_output = {
                'loss': loss.data,
                'nll_loss': nll_loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': sample['target'].size(0),
                'sample_size': sample_size,
            }
            return loss, sample_size, logging_output

        if self.update_steps % self.update_freq == 1:
            self.update_mw_token()
            self.reset_history()

        nll_loss, outer_group_losses, outer_group_counts, inner_group_losses, inner_group_counts = \
            self.compute_loss(model, sample)
        nsentences = sample['target'].size(0)

        if not self.training:
            loss = outer_group_losses.sum()
            sample_size = sample['ntokens']

            fg_labels, _ = self.retrieve_group_labels(sample)
            fg_one_vec = torch.ones(sample['nsentences'], device='cuda')  # B
            fg_zero_vec = torch.zeros(self.n_groups, device='cuda')
            fg_group_nll = fg_zero_vec.scatter_add(0, fg_labels, nll_loss)
            fg_group_count = fg_zero_vec.scatter_add(0, fg_labels, fg_one_vec)

            nll_loss = nll_loss.sum()
        else:
            self.update_steps += 1
            outer_denom = outer_group_losses.ne(0).sum()

            reduce_outer_group_losses = outer_group_losses.detach().clone()
            reduce_inner_group_losses = inner_group_losses.detach().clone()

            if torch.cuda.device_count() > 1:
                torch.distributed.all_reduce(outer_group_counts)
                torch.distributed.all_reduce(reduce_outer_group_losses)
                torch.distributed.all_reduce(inner_group_counts)
                torch.distributed.all_reduce(reduce_inner_group_losses)

            outer_group_denom = outer_group_counts + 1e-8
            reduce_outer_group_losses = reduce_outer_group_losses / outer_group_denom
            outer_group_losses = outer_group_losses * self.distributed_world_size / outer_group_denom / outer_denom

            valid_outer_index, valid_inner_index = reduce_outer_group_losses.ne(0), reduce_inner_group_losses.ne(0)
            self.outer_sum_losses[valid_outer_index] = self.outer_sum_losses[valid_outer_index].mul(1 - self.EMA_alpha).add(reduce_outer_group_losses[valid_outer_index], alpha=self.EMA_alpha)
            self.outer_count_cat[valid_outer_index] = self.outer_count_cat[valid_outer_index].mul(1 - self.EMA_alpha).add(outer_group_counts[valid_outer_index], alpha=self.EMA_alpha)
            self.inner_sum_losses[valid_inner_index] = self.inner_sum_losses[valid_inner_index].mul(1 - self.EMA_alpha).add(reduce_inner_group_losses[valid_inner_index], alpha=self.EMA_alpha)
            self.inner_count_cat[valid_inner_index] = self.inner_count_cat[valid_inner_index].mul(1 - self.EMA_alpha).add(inner_group_counts[valid_inner_index], alpha=self.EMA_alpha)

            self.update_mw()
            loss = (outer_group_losses * self.outer_h_fun).sum()
            sample_size = 1

        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        if self.training:
            for ii in range(self.n_groups):
                logging_output['w{}'.format(ii)] = self.outer_h_fun[ii]
                logging_output['l{}'.format(ii)] = self.outer_sum_losses[ii]
                logging_output["n_groups"] = self.n_groups
                logging_output['gpu_count'] = 1
        else:
            for ii in range(self.n_groups):
                logging_output["fg_gnll{}".format(ii)] = fg_group_nll[ii].data
                logging_output["fg_gcount{}".format(ii)] = fg_group_count[ii].data

        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = utils.item(sum(log.get('loss', 0) for log in logging_outputs))
        nll_loss_sum = utils.item(sum(log.get('nll_loss', 0) for log in logging_outputs))
        ntokens = utils.item(sum(log.get('ntokens', 0) for log in logging_outputs))
        sample_size = utils.item(sum(log.get('sample_size', 0) for log in logging_outputs))
        nsentences = utils.item(sum(log.get('nsentences', 0) for log in logging_outputs))
        gpu_counts = utils.item(sum(log.get('gpu_count', 0) for log in logging_outputs))

        if sample_size > 1:
            metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and 'nll_loss' in logging_outputs[0]:
            metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

        if len(logging_outputs) > 0 and 'w1' in logging_outputs[0]:
            ngroups = sum(log.get('n_groups', 0) for log in logging_outputs) / gpu_counts
            ngroups = int(ngroups.item()) if torch.is_tensor(ngroups) else int(ngroups)
            for ii in range(ngroups):
                group_loss = sum(log.get('l{}'.format(ii), 0) for log in logging_outputs) / gpu_counts
                metrics.log_scalar('acl{}'.format(ii), group_loss, 1, round=3)

            for ii in range(ngroups):
                group_loss = sum(log.get('l{}'.format(ii), 0) for log in logging_outputs) / gpu_counts
                metrics.log_scalar('l{}'.format(ii), group_loss, 0, round=3)

            for ii in range(ngroups):
                weight = sum(log.get('w{}'.format(ii), 0) for log in logging_outputs) / gpu_counts
                metrics.log_scalar('w{}'.format(ii), weight, 1, round=3)

            for ii in range(ngroups):
                metrics.log_derived_with_key('gppl{}'.format(ii),
                                             lambda value: utils.get_perplexity(value, base=math.e),
                                             "acl{}".format(ii))

        if len(logging_outputs) > 0 and 'fg_gnll0' in logging_outputs[0]:
            for ii in range(5):
                g_nll = sum(log.get('fg_gnll{}'.format(ii), 0) for log in logging_outputs)
                g_tokens = sum(log.get('fg_gcount{}'.format(ii), 0) for log in logging_outputs)
                division_g_ntokens = g_tokens if g_tokens > 0 else 1
                metrics.log_scalar('fg_gnll{}'.format(ii), g_nll / division_g_ntokens, g_tokens, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True