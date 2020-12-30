# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch
import os


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


@register_criterion('baseline_label_smoothed_cross_entropy')
class BaselineLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, group_level, log_dir):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing

        self.group_level = group_level
        self.inner_groups = len(task.target_dictionary)

        if self.group_level == "source_lang":
            # xx - en
            self.n_groups = len(task.data_manager.src_langs)
            self.lang_dict = task.data_manager.src_lang_dict
            self.f_inner_log = open(os.path.join(log_dir, "xxen_inner_baselines"), "w")
            self.f_outer_log = open(os.path.join(log_dir, "xxen_outer_baselines"), "w")
        elif self.group_level == "target_lang":
            # en - xx
            self.n_groups = len(task.data_manager.tgt_langs)
            self.lang_dict = task.data_manager.tgt_lang_dict
            self.f_inner_log = open(os.path.join(log_dir, "enxx_inner_baselines"), "w")
            self.f_outer_log = open(os.path.join(log_dir, "enxx_outer_baselines"), "w")
        else:
            raise ValueError

        self.register_buffer('outer_losses', torch.zeros(self.n_groups))
        self.register_buffer('inner_losses', torch.zeros(self.n_groups * self.inner_groups))
        self.register_buffer('outer_counts', torch.zeros(self.n_groups))
        self.register_buffer('inner_counts', torch.zeros(self.n_groups * self.inner_groups))

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--group-level', type=str, choices=['source_lang', 'target_lang'])
        parser.add_argument('--log-dir', type=str, default=None)
        # fmt: on

    def retrieve_group_labels(self, sample):
        if self.group_level == "source_lang":
            index = sample["src_lang_id"]
        elif self.group_level == "target_lang":
            index = sample["tgt_lang_id"]
        else:
            index = None
        return index, sample['target']

    def individual_losses(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)

        losses = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
        )
        return losses

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']

        net_output = model(**sample['net_input'])
        mask = (sample['target'] != self.padding_idx).float()
        token_losses = self.individual_losses(model, net_output, sample)
        if isinstance(token_losses, tuple):
            nll_loss = token_losses[1].reshape_as(sample['target']).sum(1)
            token_losses = token_losses[0].reshape_as(sample['target']) * mask
        else:
            nll_loss = (token_losses.reshape_as(sample['target']) * mask).sum(1)
            token_losses = token_losses.reshape_as(sample['target']) * mask

        loss = token_losses.sum()

        outer_index, inner_index = self.retrieve_group_labels(sample)
        offset_index = (inner_index + outer_index.unsqueeze(1) * self.inner_groups).view(-1)

        self.outer_losses.scatter_add_(0, outer_index, token_losses.sum(1))
        self.outer_counts.scatter_add_(0, outer_index, mask.sum(1))

        self.inner_losses.scatter_add_(0, offset_index, token_losses.view(-1))
        one_vec = torch.ones(offset_index.numel(), device='cuda')  # B
        self.inner_counts.scatter_add_(0, offset_index, one_vec)

        nll_loss = nll_loss.sum()
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def summarize(self):
        avg_outer_losses = self.outer_losses / self.outer_counts
        self.inner_counts[self.inner_counts == 0] = 1.
        avg_inner_losses = (self.inner_losses / self.inner_counts).view(self.n_groups, self.inner_groups)
        for idx, loss in enumerate(avg_outer_losses):
            self.f_outer_log.write("{}={}\n".format(self.lang_dict[idx+1], loss.item()))

        for idx in range(self.n_groups):
            ss = " ".join([str(ele.item()) for ele in avg_inner_losses[idx]])
            self.f_inner_log.write("{}={}\n".format(self.lang_dict[idx+1], ss))
        self.f_outer_log.close()
        self.f_inner_log.close()

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
