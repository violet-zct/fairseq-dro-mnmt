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
import numpy as np
from scipy import optimize

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


def project_to_cs_ball(v, rho, p_train):
    """Numpy/Scipy projection to chi-square ball of radius rho"""
    n = len(v)

    def cs_div(p):
        return 0.5 * np.mean((p / p_train - 1)**2)

    # first, check if a simplex projection is within the chi-square ball
    target_simplex = lambda eta: np.sum(np.maximum(v - eta, 0)) - 1.0
    eta_min_simplex = v.min() - 1 / n
    eta_max_simplex = v.max()
    eta_simplex = optimize.brentq(
        target_simplex, eta_min_simplex, eta_max_simplex)
    p_candidate = np.maximum(v - eta_simplex, 0)
    if cs_div(p_candidate) <= rho:
        return p_candidate

    # second, compute a chi-square best response
    def target_cs(eta, return_p=False):
        p = np.maximum(v - eta, 0)
        if p.sum() == 0.0:
            p[np.argmax(v)] = 1.0
        else:
            p /= p.sum()
        err = cs_div(p) - rho
        return p if return_p else err
    eta_max_cs = v.max()
    eta_min_cs = v.min()
    if target_cs(eta_max_cs) <= 0:
        return target_cs(eta_max_cs, return_p=True)
    while target_cs(eta_min_cs) > 0.0:  # find left interval edge for bisection
        eta_min_cs = 2 * eta_min_cs - eta_max_cs
    eta_cs = optimize.brentq(
        target_cs, eta_min_cs, eta_max_cs)
    p_candidate = target_cs(eta_cs, return_p=True)
    assert np.abs(cs_div(p_candidate) - rho) < rho * 1e-2
    return p_candidate


@register_criterion('chi_square_primal_dual')
class ChiSquarePrimalDualLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing, group_level, step_size, rho,
                 update_dro_freq, start_ft_steps, clip):
        super().__init__(task)

        self.args = self.task.args
        self.distributed_world_size = self.task.args.distributed_world_size
        self.eps = label_smoothing
        self.group_level = group_level
        self.step_size = step_size
        self.rho = rho

        self.update_freq = update_dro_freq

        self.device = torch.cuda.current_device()
        self.temp_idx = 0
        self.print_steps = 100

        self.update_steps = 0
        self.start_ft_steps = start_ft_steps
        self.clip = clip

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

        self.p_train = None
        # fixme: initialize with uniform?
        self.h_fun = np.ones(self.n_groups) / self.n_groups

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--group-level', type=str, choices=['source_lang', 'target_lang', 'token'])
        parser.add_argument('--step-size', default=1e-4, type=float, help='lr for q')
        parser.add_argument('--rho', default=0.1, type=float)
        parser.add_argument('--update-dro-freq', default=1, type=int)
        parser.add_argument('--start-ft-steps', default=0, type=int)
        parser.add_argument('--clip', default=None, type=float)
        # fmt: on

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

        index = self.retrieve_group_labels(sample)
        zero_vec = torch.zeros(self.n_groups, device='cuda')  # G
        group_losses = zero_vec.scatter_add(0, index, ind_loss)

        if self.group_level != "token":
            group_counts = zero_vec.scatter_add(0, index, mask.sum(1))
        else:
            one_vec = torch.ones(ind_loss.size(0), device='cuda')  # B
            group_counts = zero_vec.scatter_add(0, index, one_vec)

        return nll_loss, group_losses, group_counts

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
            self.p_train = self.task.data_manager.data_ratios
            self.p_train_tensor = torch.Tensor(self.p_train).to(self.device)
            # self.h_fun = self.p_train
            logger.info("Fixed P train = {}".format(self.p_train))

        # pure warmup
        if self.update_steps < self.start_ft_steps:
            nsentences = sample['target'].size(0)
            net_output = model(**sample['net_input'])
            if self.training:
                self.update_steps += 1
                net_output = model(**sample['net_input'])
                loss, nll_loss = self.simple_loss(model, net_output, sample, reduce=reduce)
                sample_size = sample['ntokens']
            else:
                loss, nll_loss = self.simple_loss(model, net_output, sample, reduce=False)
                nll_loss = nll_loss.reshape_as(sample['target']).sum(1)
                loss = loss.reshape_as(sample['target']).sum(1)

                mask = (sample['target'] != self.padding_idx).float()
                sample_size = sample['ntokens']
                fg_labels = self.retrieve_group_labels(sample)
                fg_zero_vec = torch.zeros(self.n_groups, device='cuda')
                fg_group_nll = fg_zero_vec.scatter_add(0, fg_labels, nll_loss)
                fg_group_count = fg_zero_vec.scatter_add(0, fg_labels, mask.sum(1))

                loss = loss.sum()
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
            sample_size = sample['ntokens']
            if self.logging:
                fg_labels = self.retrieve_group_labels(sample)
                fg_zero_vec = torch.zeros(self.n_groups, device='cuda')
                fg_group_nll = fg_zero_vec.scatter_add(0, fg_labels, nll_loss)
                fg_group_count = group_counts.detach().clone()
            # fixme: valid loss = robust loss?
            loss = group_losses.sum()
        else:
            self.update_steps += 1
            sample_size = 1

            reduce_group_losses = group_losses.detach().clone()
            if torch.cuda.device_count() > 1:
                torch.distributed.all_reduce(group_counts)
                torch.distributed.all_reduce(reduce_group_losses)

            group_denom = group_counts + 1e-8
            reduce_group_losses = reduce_group_losses / group_denom
            loss = self.compute_robust_loss(reduce_group_losses, group_losses)

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

        if self.logging and not self.training:
            for ii in range(self.n_groups):
                logging_output["fg_gnll{}".format(ii)] = fg_group_nll[ii].data
                logging_output["fg_gcount{}".format(ii)] = fg_group_count[ii].data

        return loss, sample_size, logging_output

    def compute_robust_loss(self, reduce_group_losses, group_losses):
        # h_fun is q
        # reduce_group_losses[i] = mean of group i's losses in a batch
        np_group_losses = reduce_group_losses.cpu().numpy()
        coefs = self.step_size * self.h_fun / self.p_train
        q_update = coefs * np_group_losses
        if self.clip is not None:
            q_update = np.minimum(q_update, self.clip)
        q = self.h_fun + q_update
        self.h_fun = project_to_cs_ball(q, self.rho, self.p_train)
        q = reduce_group_losses.new_tensor(self.h_fun, requires_grad=False)
        loss = (q * group_losses).sum()

        if self.update_steps % 100 == 0:
            logger.info("Group loss weights: {}".format(" ".join(["{:.6f}".format(xx.item()) for xx in self.h_fun])))
        return loss

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