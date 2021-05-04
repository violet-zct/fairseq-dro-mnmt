import math
import torch.nn.functional as F
import torch.nn as nn
import torch
import torch.distributed as dist
import numpy as np
from fairseq import metrics, utils
from fairseq import utils
from collections import defaultdict
from fairseq import distributed_utils
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.sentence_prediction import SentencePredictionTask
from . import FairseqCriterion, register_criterion

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
        if pad_mask.any():
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


@register_criterion('group_dro_eg')
class GroupDROEGCriterion(FairseqCriterion):

    def __init__(self, task, label_smoothing, group_level, baselines, eg_step_size):
        super().__init__(task)
        self.args = task.args

        self.device = torch.cuda.current_device()
        self.step_size = eg_step_size
        self.temp_idx = 0

        self.group_level = group_level
        self.baselines = baselines

        self.normalize_loss = self.args.eg_normalize
        self.eps = label_smoothing

        self.logging = True
        if group_level == "source_lang":
            self.n_groups = len(task.data_manager.src_langs)
        elif group_level == "target_lang":
            self.n_groups = len(task.data_manager.tgt_langs)
        else:
            raise ValueError

        self.register_buffer('adv_probs', torch.ones(self.n_groups) / self.n_groups)
        if baselines is None:
            self.loss_baselines = torch.Tensor([0. for _ in range(self.n_groups)]).to(self.device)
        else:
            fields = baselines.split(",")
            tdict = {fd.split(":")[0]: float(fd.split(":")[-1]) for fd in fields}
            baselines = [-1 for _ in range(self.n_groups)]
            for lang, value in tdict.items():
                lang_dict = self.task.data_manager.tgt_lang_dict if self.group_level == "target_lang" else self.task.data_manager.src_lang_dict
                baselines[lang_dict.index(lang) - 1] = value
            self.loss_baselines = torch.Tensor(baselines).to(self.device)

    def add_args(parser):
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--eg-step-size', type=float, default=-1)
        parser.add_argument('--eg-normalize', type=int, default=0)

        parser.add_argument('--group-level', type=str, choices=['source_lang', 'target_lang'])
        parser.add_argument('--baselines', default=None, type=str, help='baseline loss values.')

    def retrieve_group_labels(self, sample):
        if self.group_level == "source_lang":
            index = sample["src_lang_id"]

        elif self.group_level == "target_lang":
            index = sample["tgt_lang_id"]
        else:
            index = sample['target'].view(-1)
        return index

    def compute_loss(self, model, sample, reduce=True):
        net_output = model(**sample['net_input'])
        target_kw = 'target'
        token_loss, nll_loss = self.individual_losses(model, net_output, sample)

        mask = (sample[target_kw] != self.padding_idx).float()

        if not reduce:
            ind_loss = token_loss.reshape_as(sample[target_kw]) * mask
            nll_loss = nll_loss.reshape_as(sample[target_kw])
            return nll_loss, ind_loss, None, None

        ind_loss = (token_loss.reshape_as(sample[target_kw]) * mask).sum(1)
        nll_loss = (nll_loss.reshape_as(sample[target_kw]) * mask).sum(1)

        with torch.no_grad():
            index = self.retrieve_group_labels(sample)
            zero_vec = torch.zeros(self.n_groups, device='cuda')  # G
            group_losses = zero_vec.scatter_add(0, index, ind_loss)
            group_counts = zero_vec.scatter_add(0, index, mask.sum(1))

        return nll_loss, ind_loss, group_losses, group_counts

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        # train: nll_loss is summed,
        # valid: nll_loss is sentence-wise,
        nsentences = sample['target'].size(0)
        sample_size = sample['ntokens']
        nll_loss, ind_loss, group_losses, group_counts = self.compute_loss(model, sample)

        if not self.training:
            fg_labels = self.retrieve_group_labels(sample)
            fg_zero_vec = torch.zeros(self.n_groups, device='cuda')
            fg_group_nll = fg_zero_vec.scatter_add(0, fg_labels, nll_loss)
            fg_group_count = group_counts.detach().clone()
            fg_loss_vec = group_losses
            loss = ind_loss.sum()
        else:
            reduce_group_losses = group_losses.detach().clone()
            if torch.cuda.device_count() > 1:
                torch.distributed.all_reduce(group_counts)
                torch.distributed.all_reduce(reduce_group_losses)

            group_denom = group_counts + 1e-8
            reduce_group_losses = reduce_group_losses / group_denom

            loss = self.compute_eg_robust_loss(group_losses, reduce_group_losses)

        nll_loss = nll_loss.sum()
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': nsentences,
            'sample_size': sample_size,
        }

        if self.logging and not self.training:
            for ii in range(self.n_groups):
                logging_output["fg_gnll{}".format(ii)] = fg_group_nll[ii].data
                logging_output["fg_gcount{}".format(ii)] = fg_group_count[ii].data
                logging_output["fg_gloss{}".format(ii)] = fg_loss_vec[ii].data

        return loss, sample_size, logging_output

    def compute_eg_robust_loss(self, group_loss, reduce_group_loss):
        with torch.no_grad():
            adjusted_loss = reduce_group_loss - self.loss_baselines
            if self.normalize_loss:
                adjusted_loss = adjusted_loss / (adjusted_loss.sum())
            exp_weights = torch.exp(self.step_size * adjusted_loss)
            self.adv_probs.mul_(exp_weights)
            self.adv_probs.div_(self.adv_probs.sum())

        self.temp_idx += 1
        if self.temp_idx % 1000 == 0:
            logger.info("Group Loss = {}".format(reduce_group_loss))
            logger.info("EG Weights = {}".format(exp_weights / exp_weights.max()))
        robust_loss = group_loss * self.adv_probs
        return robust_loss.sum()

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
                division_g_ntokens = g_tokens + 1e-8
                metrics.log_scalar('fg_gnll{}'.format(ii), g_nll / division_g_ntokens / math.log(2), g_tokens, round=3)
                metrics.log_derived_with_key('fg_ppl{}'.format(ii), lambda value: utils.get_perplexity(value),
                                             "fg_gnll{}".format(ii))

                if 'fg_gloss0' in logging_outputs[0]:
                    g_loss = sum(log.get('fg_gloss{}'.format(ii), 0) for log in logging_outputs)
                    metrics.log_scalar('fg_gloss{}'.format(ii), g_loss / division_g_ntokens, g_tokens, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True