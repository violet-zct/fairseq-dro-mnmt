# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch


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


@register_criterion('train_dynamics_label_smoothed_cross_entropy')
class TrainDynamicsLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, group_level):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.group_level = group_level
        if self.group_level == "source_lang":
            # xx - en
            self.n_groups = len(task.data_manager.src_langs)
        elif self.group_level == "target_lang":
            # en - xx
            self.n_groups = len(task.data_manager.tgt_langs)
        else:
            raise ValueError

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--group-level', type=str, choices=['source_lang', 'target_lang'])
        parser.add_argument('--compute-train-dynamics', type=int, default=1)
        parser.add_argument('--analyze', type=int, default=1)
        # fmt: on

    def retrieve_group_labels(self, sample):
        if self.group_level == "source_lang":
            index = sample["src_lang_id"]

        elif self.group_level == "target_lang":
            index = sample["tgt_lang_id"]
        else:
            index = sample['target'].view(-1)
        return index

    def forward(self, model, sample, reduce=True, train_dynamic=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        if self.training:
            loss, nll_loss = self.simple_loss(model, net_output, sample, reduce=True)
            sample_size = sample['target'].size(0) if self.sentence_avg else sample['ntokens']
        elif train_dynamic:
            loss, nll_loss = self.more_loss(model, net_output, sample, reduce=False)
            word_mask = (sample['target'] != self.padding_idx).float()
            pad_mask = (sample['target'] == self.padding_idx)
            nll_loss = nll_loss.reshape_as(sample['target'])
            probs = torch.exp(-nll_loss)
            probs[pad_mask] = 0
            avg_probs = torch.sum(probs, dim=-1) / word_mask.sum(1)
            probs.masked_fill_(pad_mask, float('inf'))
            min_probs, _ = probs.min(dim=-1)
            median_indices = word_mask.sum(1, keepdim=True).long() // 2
            sorted_probs, _ = torch.sort(probs, dim=-1)  # ascending
            median_probs = torch.gather(sorted_probs, 1, median_indices).squeeze(-1)
            loss = loss.sum()
            nll_loss = nll_loss.sum()
            sample_size = sample['ntokens']
            return_metrics = {'min_probs': min_probs, 'median_probs': median_probs, "avg_probs": avg_probs}
        else:
            loss, nll_loss = self.simple_loss(model, net_output, sample, reduce=False)
            mask = (sample['target'] != self.padding_idx).float()
            ind_loss = (loss.reshape_as(sample['target']) * mask).sum(1)
            loss = loss.sum()
            nll_loss = nll_loss.reshape_as(sample['target']).sum(1)

            sample_size = sample['ntokens']
            if self.n_groups > 1:
                fg_labels = self.retrieve_group_labels(sample)
                fg_zero_vec = torch.zeros(self.n_groups, device='cuda')
                fg_group_nll = fg_zero_vec.scatter_add(0, fg_labels, nll_loss)
                fg_group_count = fg_zero_vec.scatter_add(0, fg_labels, mask.sum(1))
                fg_group_loss = fg_zero_vec.scatter_add(0, fg_labels, ind_loss)
            nll_loss = nll_loss.sum()

        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': len(sample['id']),
            'sample_size': sample_size,
            'n_groups': self.n_groups,
            'gpu_count': 1,
        }

        if train_dynamic:
            return loss, sample_size, logging_output, sample['concat_ds_id'], return_metrics

        if not self.training and not train_dynamic and self.n_groups > 1:
            for ii in range(self.n_groups):
                logging_output["fg_gnll{}".format(ii)] = fg_group_nll[ii].data
                logging_output["fg_loss{}".format(ii)] = fg_group_loss[ii].data
                logging_output["fg_gcount{}".format(ii)] = fg_group_count[ii].data
        return loss, sample_size, logging_output

    def simple_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def more_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)

        probs = torch.exp(lprobs)
        mask = (sample['target'] != self.padding_idx).float()
        # entropy = -(probs * lprobs).sum(-1) * mask  # B X L
        # avg_entropy = entropy.sum(1) / mask.sum(1)

        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss #, avg_entropy

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get('nll_loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

        gpu_counts = utils.item(sum(log.get('gpu_count', 0) for log in logging_outputs))
        ngroups = sum(log.get('n_groups', 0) for log in logging_outputs) / gpu_counts
        ngroups = int(ngroups.item()) if torch.is_tensor(ngroups) else int(ngroups)

        metrics.log_scalar('loss', loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_scalar('nll_loss', nll_loss_sum / ntokens / math.log(2), ntokens, round=3)
        metrics.log_derived('ppl', lambda meters: utils.get_perplexity(meters['nll_loss'].avg))

        if len(logging_outputs) > 0 and 'fg_gnll0' in logging_outputs[0]:
            for ii in range(ngroups):
                g_nll = sum(log.get('fg_gnll{}'.format(ii), 0) for log in logging_outputs)
                g_tokens = sum(log.get('fg_gcount{}'.format(ii), 0) for log in logging_outputs)
                division_g_ntokens = g_tokens if g_tokens > 0 else 1

                g_loss = sum(log.get('fg_loss{}'.format(ii), 0) for log in logging_outputs)
                metrics.log_scalar('fg_gnll{}'.format(ii), g_nll / division_g_ntokens / math.log(2), g_tokens, round=3)
                metrics.log_scalar('fg_gloss{}'.format(ii), g_loss / division_g_ntokens, g_tokens, round=3)
                metrics.log_derived_with_key('fg_ppl{}'.format(ii), lambda value: utils.get_perplexity(value),
                                             "fg_gnll{}".format(ii))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
