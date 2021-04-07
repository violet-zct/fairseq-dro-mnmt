# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
import torch
from fairseq.criterions.fast_dro.robust_losses import RobustLoss


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


@register_criterion('chi_square_batch_dro')
class ChiSquareBatchDROCriterion(FairseqCriterion):

    def __init__(self, task, sentence_avg, label_smoothing, group_level, rho):
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

        self.robust_loss = RobustLoss(rho, reg=0, geometry='chi-square')
        self.ata_parallel_world_size = task.args.distributed_world_size

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--group-level', type=str, choices=['source_lang', 'target_lang'])
        parser.add_argument('--rho', type=float, default=0.1)
        # fmt: on

    def retrieve_group_labels(self, sample):
        if self.group_level == "source_lang":
            index = sample["src_lang_id"]

        elif self.group_level == "target_lang":
            index = sample["tgt_lang_id"]
        else:
            index = sample['target'].view(-1)
        return index

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])

        if self.training:
            token_losses, nll_loss = self.simple_loss(model, net_output, sample, reduce=False)
            nll_loss = nll_loss.sum()
            mask = (sample['target'] != self.padding_idx).float()

            data_labels = self.retrieve_group_labels(sample)

            ind_losses = token_losses.reshape_as(sample['target']).sum(1) / mask.sum(1)
            batch_size = sample['id'].new_tensor([len(ind_losses)])
            batch_list = [torch.zeros_like(batch_size) for _ in range(self.data_parallel_world_size)]
            torch.distributed.all_gather(batch_list, batch_size)

            # expand
            batch_list = torch.cat(batch_list)
            max_batch = torch.max(batch_list)
            if batch_size < max_batch:
                diff = max_batch - batch_size
                data_labels = torch.cat([data_labels, data_labels.new_zeros(diff)])
                ind_losses = torch.cat([ind_losses, ind_losses.new_zeros(diff)])

            gather_data_labels = [data_labels.new_zeros(max_batch) for bs in batch_list if bs != 0]
            gather_ind_losses = [ind_losses.new_zeros(max_batch) for bs in batch_list if bs != 0]
            torch.distributed.all_gather(gather_data_labels, data_labels)
            torch.distributed.all_gather(gather_ind_losses, ind_losses)

            # compact
            gather_data_labels = torch.cat([gather_data_labels[ii][:bs]
                                            for ii, bs in enumerate(batch_list) if bs != 0])
            gather_ind_losses = torch.cat([gather_ind_losses[ii][:bs]
                                           for ii, bs in enumerate(batch_list) if bs != 0])

            best_response = self.robust_loss(gather_ind_losses)
            loss = torch.dot(best_response, gather_ind_losses)

            with torch.no_grad():
                fg_zero_vec = torch.zeros(self.n_groups, device='cuda')
                fg_group_losses = fg_zero_vec.scatter_add(0, gather_data_labels, gather_ind_losses)
                fg_group_weights = fg_zero_vec.scatter_add(0, gather_data_labels, best_response)
                one_vec = torch.ones(len(best_response), device='cuda')
                fg_group_count = fg_zero_vec.scatter_add(0, gather_data_labels, one_vec)
                # for each group: average token losses over samples = \sum avg_token_loss / #samples of G_i
                avg_group_losses = fg_group_losses / (fg_group_count + 1e-8)
                # for each group: average group weight = \sum instance_q / #sample of G_i
                avg_group_weight = fg_group_weights / (fg_group_count + 1e-8)
            sample_size = 1
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
            'nsentences': len(sample['id']),
            'sample_size': sample_size,
            'n_groups': self.n_groups,
            'gpu_count': 1,
        }
        if not self.training:
            for ii in range(self.n_groups):
                logging_output["fg_gnll{}".format(ii)] = fg_group_nll[ii].data
                logging_output["fg_gcount{}".format(ii)] = fg_group_count[ii].data
        else:
            for ii in range(self.n_groups):
                logging_output["g{}_l".format(ii)] = avg_group_losses[ii].data
                logging_output["g{}_w".format(ii)] = avg_group_weight[ii].data

        return loss, sample_size, logging_output

    def simple_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

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
                metrics.log_scalar('fg_gnll{}'.format(ii), g_nll / division_g_ntokens / math.log(2), g_tokens, round=3)
                metrics.log_derived_with_key('fg_ppl{}'.format(ii), lambda value: utils.get_perplexity(value),
                                             "fg_gnll{}".format(ii))

        if len(logging_outputs) > 0 and 'g0_l' in logging_outputs[0]:
            for ii in range(ngroups):
                g_loss = sum(log.get('g{}_l'.format(ii), 0) for log in logging_outputs) / gpu_counts
                g_w = sum(log.get('g{}_w'.format(ii), 0) for log in logging_outputs) / gpu_counts
                metrics.log_scalar('g{}_l'.format(ii), g_loss, 1, round=3)
                metrics.log_scalar('g{}_w'.format(ii), g_w, 1, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
