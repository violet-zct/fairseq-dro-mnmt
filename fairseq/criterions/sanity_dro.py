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


@register_criterion('sanity_plain_dro_label_smoothed_cross_entropy')
class SanityPlainDROLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing, group_level, dro_alpha, baselines,
                 update_dro_freq, start_ft_steps, ema,):
        super().__init__(task)
        self.distributed_world_size = self.task.args.distributed_world_size
        self.eps = label_smoothing
        self.group_level = group_level
        self.alpha = dro_alpha
        self.baselines = baselines
        self.update_freq = update_dro_freq

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
        # fmt: on

    def initialize(self):
        logger.info("Group num = {}".format(self.n_groups))
        weights = torch.Tensor(self.task.data_manager.lang_weights)
        self.register_buffer('h_fun', weights)

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
        # pure warmup
        if self.update_steps < self.start_ft_steps:
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

        nll_loss, group_losses, group_counts = self.compute_loss(model, sample)
        nsentences = sample['target'].size(0)

        if not self.training:
            loss = group_losses.sum()
            sample_size = sample['ntokens']

            if self.logging:
                fg_labels = self.retrieve_group_labels(sample)
                fg_one_vec = torch.ones(sample['nsentences'], device='cuda')  # B
                fg_zero_vec = torch.zeros(self.n_groups, device='cuda')
                fg_group_nll = fg_zero_vec.scatter_add(0, fg_labels, nll_loss)
                fg_group_count = fg_zero_vec.scatter_add(0, fg_labels, fg_one_vec)

            nll_loss = nll_loss.sum()
        else:
            self.update_steps += 1
            loss = (group_losses * self.h_fun).sum()
            sample_size = sample['ntokens']

        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        if self.logging:
            if not self.training:
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