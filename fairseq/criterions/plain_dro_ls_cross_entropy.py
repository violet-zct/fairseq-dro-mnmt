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


@register_criterion('plain_dro_label_smoothed_cross_entropy')
class PlainDROLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing, group_level, dro_alpha, baselines,
                 update_dro_freq, start_ft_steps, ema, resampling):
        super().__init__(task)
        self.distributed_world_size = self.task.args.distributed_world_size
        self.eps = label_smoothing
        self.group_level = group_level
        self.alpha = dro_alpha
        self.baselines = baselines
        self.update_freq = update_dro_freq

        self.resampling = resampling
        self.device = torch.cuda.current_device()
        self.temp_idx = 0
        self.print_steps = 100

        self.update_steps = 0
        self.start_ft_steps = start_ft_steps
        self.EMA_alpha = ema

        self.logging = True
        if group_level == "source_lang":
            self.n_groups = len(task.data_manager.src_langs)
        elif group_level == "target_lang":
            self.n_groups = len(task.data_manager.tgt_langs)
        elif group_level == "token":
            self.n_groups = len(task.target_dictionary)
            self.logging = False
        else:
            raise ValueError

        self.initialize()

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--group-level', type=str, choices=['source_lang', 'target_lang', 'token'])
        parser.add_argument('--dro-alpha', default=1., type=float, help='alpha value for the DRO loss.')
        parser.add_argument('--baselines', default=None, type=str, help='baseline loss values.')
        parser.add_argument('--update-dro-freq', default=1, type=int)
        parser.add_argument('--start-ft-steps', default=0, type=int)
        parser.add_argument('--ema', default=0.1, type=float)
        parser.add_argument('--resampling', default=0, type=int)
        # fmt: on

    def initialize(self):
        logger.info("Group num = {}".format(self.n_groups))
        if self.baselines is None:
            self.loss_baselines = torch.Tensor([0. for _ in range(self.n_groups)]).to(self.device)
        else:
            fields = self.baselines.split(",")
            tdict = {fd.split(":")[0]: float(fd.split(":")[-1]) for fd in fields}
            baselines = [-1 for _ in range(self.n_groups)]
            for lang, value in tdict.items():
                lang_dict = self.task.data_manager.tgt_lang_dict if self.group_level == "target_lang" else self.task.data_manager.src_lang_dict
                baselines[lang_dict.index(lang) - 1] = value
            self.loss_baselines = torch.Tensor(baselines).to(self.device)
        self.register_buffer('h_fun', torch.ones(self.n_groups))
        self.register_buffer('sum_losses', torch.zeros(self.n_groups))  # historical loss sum over category
        self.register_buffer('count_cat', torch.ones(self.n_groups))

    def update_mw(self, epoch=-1):
        if epoch == 1 or self.p_train is None:
            return None

        # version that uses EMA. (sum_losses is EMA running loss, count_cat is EMA running sum)
        past_losses = self.sum_losses
        baselined_losses = past_losses - self.loss_baselines

        past_frac = self.p_train
        #
        sorted_losses, sort_id = torch.sort(baselined_losses, descending=True)
        sorted_frac = past_frac[sort_id]
        cutoff_count = torch.sum(torch.cumsum(sorted_frac, 0) < self.alpha)
        if cutoff_count == len(sorted_frac):
            cutoff_count = len(sorted_frac) - 1
        self.h_fun.fill_(0.1)
        self.h_fun[sort_id[:cutoff_count]] = 1.0 / self.alpha
        leftover_mass = 1.0 - sorted_frac[:cutoff_count].sum().div(self.alpha)
        tiebreak_fraction = leftover_mass / sorted_frac[cutoff_count]  # check!
        self.h_fun[sort_id[cutoff_count]] = tiebreak_fraction

        self.temp_idx += 1
        if self.logging and self.temp_idx % self.print_steps == 0:
            logger.info("EMA past losses: {}".format(" ".join(["{:.6f}".format(xx.item()) for xx in baselined_losses[0:self.n_groups]])))
            logger.info("EMA group fractions: {}".format(" ".join(["{:.6f}".format(xx.item()) for xx in past_frac[0:self.n_groups]])))
            logger.info("Group loss weights: {}".format(" ".join(["{:.6f}".format(xx.item()) for xx in self.h_fun[0:self.n_groups]])))

        if self.resampling:
            q = self.h_fun * self.p_train
            logger.info("Q = {}".format(
                " ".join(["{:.6f}".format(xx.item()) for xx in q[0:self.n_groups]])))
            return q

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
            index = sample['target'].view(-1)
        return index

    def compute_loss(self, model, sample):
        net_output = model(**sample['net_input'])
        mask = (sample['target'] != self.padding_idx).float()
        token_losses = self.individual_losses(model, net_output, sample)
        if isinstance(token_losses, tuple):
            nll_loss = token_losses[1].reshape_as(sample['target']).sum(1)
            token_losses = token_losses[0]
        else:
            nll_loss = (token_losses.reshape_as(sample['target']) * mask).sum(1)

        if self.group_level == "token":
            ind_loss = (token_losses.reshape_as(sample['target']) * mask).view(-1)
        else:
            ind_loss = (token_losses.reshape_as(sample['target']) * mask).sum(1)

        if not self.training:
            return nll_loss, ind_loss, 0

        index = self.retrieve_group_labels(sample)
        zero_vec = torch.zeros(self.n_groups, device='cuda')  # G
        group_losses = zero_vec.scatter_add(0, index, ind_loss)

        if self.group_level != "token":
            group_counts = zero_vec.scatter_add(0, index, mask.sum(1))
        else:
            one_vec = torch.ones(ind_loss.size(0), device='cuda')  # B
            group_counts = zero_vec.scatter_add(0, index, one_vec)

        return nll_loss.sum(), group_losses, group_counts

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
        if self.p_train is None:
            self.p_train = torch.Tensor(self.task.data_manager.data_ratios).to(self.device)
            logger.info("Fixed P train = {}".format(self.p_train))

        # pure warmup
        if self.update_steps < self.start_ft_steps:
            nsentences = sample['target'].size(0)
            net_output = model(**sample['net_input'])
            if self.training:
                self.update_steps += 1
                loss, nll_loss = self.simple_loss(model, net_output, sample, reduce=reduce)
                sample_size = sample['ntokens']
            else:
                loss, nll_loss = self.simple_loss(model, net_output, sample, reduce=False)
                loss = loss.sum()
                nll_loss = nll_loss.reshape_as(sample['target']).sum(1)
                mask = (sample['target'] != self.padding_idx).float()
                sample_size = sample['ntokens']
                fg_labels = self.retrieve_group_labels(sample)
                fg_zero_vec = torch.zeros(self.n_groups, device='cuda')
                fg_group_nll = fg_zero_vec.scatter_add(0, fg_labels, nll_loss)
                fg_group_count = fg_zero_vec.scatter_add(0, fg_labels, mask.sum(1))
                nll_loss = nll_loss.sum()
            logging_output = {
                'loss': loss.data,
                'nll_loss': nll_loss.data,
                'ntokens': sample['ntokens'],
                'nsentences': nsentences,
                'sample_size': sample_size,
                'n_groups': self.n_groups,
                'gpu_count': 1,
            }
            if not self.training:
                for ii in range(self.n_groups):
                    logging_output["fg_gnll{}".format(ii)] = fg_group_nll[ii].data
                    logging_output["fg_gcount{}".format(ii)] = fg_group_count[ii].data
            return loss, sample_size, logging_output

        nll_loss, group_losses, group_counts = self.compute_loss(model, sample)
        nsentences = sample['target'].size(0)

        if not self.training:
            loss = group_losses.sum()
            sample_size = sample['ntokens']

            if self.logging:
                mask = (sample['target'] != self.padding_idx).float()
                fg_labels = self.retrieve_group_labels(sample)
                fg_zero_vec = torch.zeros(self.n_groups, device='cuda')
                fg_group_nll = fg_zero_vec.scatter_add(0, fg_labels, nll_loss)
                fg_group_count = fg_zero_vec.scatter_add(0, fg_labels, mask.sum(1))

            nll_loss = nll_loss.sum()
        else:
            self.update_steps += 1
            denom = group_losses.ne(0).sum()

            reduce_group_losses = group_losses.detach().clone()
            if torch.cuda.device_count() > 1:
                torch.distributed.all_reduce(group_counts)
                torch.distributed.all_reduce(reduce_group_losses)

            group_denom = group_counts + 1e-8
            reduce_group_losses = reduce_group_losses / group_denom
            # group_losses = group_losses * self.distributed_world_size / group_denom / denom

            valid_index = reduce_group_losses.ne(0)
            valid_losses = self.sum_losses[valid_index]
            valid_counts = self.count_cat[valid_index]
            self.sum_losses[valid_index] = valid_losses.mul(1 - self.EMA_alpha).add(reduce_group_losses[valid_index], alpha=self.EMA_alpha)
            self.count_cat[valid_index] = valid_counts.mul(1 - 0.01).add(group_counts[valid_index], alpha=0.01)

            if not self.resampling:
                self.update_mw()
                loss = (group_losses * self.h_fun).sum()
            else:
                loss = group_losses.sum()

            sample_size = sample['ntokens']

        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': nsentences,
            'sample_size': sample_size,
            'n_groups': self.n_groups,
            'gpu_count': 1,
        }

        if self.logging and not self.training:
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
        ngroups = sum(log.get('n_groups', 0) for log in logging_outputs) / gpu_counts
        ngroups = int(ngroups.item()) if torch.is_tensor(ngroups) else int(ngroups)

        if sample_size > 1:
            metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and 'nll_loss' in logging_outputs[0]:
            metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
            metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

        if len(logging_outputs) > 0 and 'w1' in logging_outputs[0]:
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
            for ii in range(ngroups):
                g_nll = sum(log.get('fg_gnll{}'.format(ii), 0) for log in logging_outputs)
                g_tokens = sum(log.get('fg_gcount{}'.format(ii), 0) for log in logging_outputs)
                division_g_ntokens = g_tokens if g_tokens > 0 else 1
                metrics.log_scalar('fg_gnll{}'.format(ii), g_nll / division_g_ntokens, g_tokens, round=1)
                metrics.log_derived_with_key('fg_ppl{}'.format(ii), lambda value: utils.get_perplexity(value, base=math.e),
                                             "fg_gnll{}".format(ii))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True