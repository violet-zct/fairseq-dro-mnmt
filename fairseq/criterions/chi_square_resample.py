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

logger = logging.getLogger(__name__)


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


def bisection(eta_min, eta_max, f, tol=1e-6, max_iter=1000):
    """Expects f an increasing function and return eta in [eta_min, eta_max]
    s.t. |f(eta)| <= tol (or the best solution after max_iter iterations"""
    lower = f(eta_min)
    upper = f(eta_max)

    # until the root is between eta_min and eta_max, double the length of the
    # interval starting at either endpoint.
    while lower > 0 or upper < 0:
        length = eta_max - eta_min
        if lower > 0:
            eta_max = eta_min
            eta_min = eta_min - 2 * length
        if upper < 0:
            eta_min = eta_max
            eta_max = eta_max + 2 * length

        lower = f(eta_min)
        upper = f(eta_max)

    for _ in range(max_iter):
        eta = 0.5 * (eta_min + eta_max)

        v = f(eta)

        if torch.abs(v) <= tol:
            return eta

        if v > 0:
            eta_max = eta
        elif v < 0:
            eta_min = eta

    # if the minimum is not reached in max_iter, returns the current value
    logger.info('Maximum number of iterations exceeded in bisection')
    return 0.5 * (eta_min + eta_max)

@register_criterion('chi_square_resample')
class ChiSquareResampleLabelSmoothedCrossEntropyCriterion(FairseqCriterion):
    def __init__(self, task, label_smoothing, group_level, rho, baselines,
                 warmup_epochs, ema, min_prob, clamp_q_to_min, resample):
        super().__init__(task)

        self.args = self.task.args
        self.distributed_world_size = self.task.args.distributed_world_size
        self.eps = label_smoothing
        self.group_level = group_level
        self.rho = rho
        self.reg = self.args.reg
        self.baselines = baselines
        self.resample = resample
        self.tol = 1e-4
        self.min_prob = min_prob

        self.device = torch.cuda.current_device()
        self.temp_idx = 0
        self.print_steps = 100

        self.update_steps = 0
        self.epochs = 0
        self.warmup_epochs = warmup_epochs
        self.EMA_alpha = ema

        self.logging = True
        if group_level == "source_lang":
            self.n_groups = len(task.data_manager.src_langs)
        elif group_level == "target_lang":
            self.n_groups = len(task.data_manager.tgt_langs)
        else:
            raise ValueError
        self.initialize()

        self.clamp_q_to_min = clamp_q_to_min
        self.p_train = None

        self.generalization_errors = None

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--group-level', type=str, choices=['source_lang', 'target_lang'])
        parser.add_argument('--rho', default=0.1, type=float)
        parser.add_argument('--baselines', default=None, type=str, help='baseline loss values.')  # DELETE
        parser.add_argument('--warmup-epochs', default=1, type=int)
        parser.add_argument('--ema', default=0.1, type=float)
        parser.add_argument('--dro-K', default=-1, type=float)  # DELETE
        parser.add_argument('--min-prob', default=0.2, type=float)
        parser.add_argument('--clamp-q-to-min', default=0, type=int)
        parser.add_argument('--reg', default=0, type=float)

        parser.add_argument('--clear-history', default=1, type=int)
        parser.add_argument('--resample', default=1, type=int, help="resample=0 is ERM")

        parser.add_argument('--compute-train-dynamics', type=int, default=0)
        parser.add_argument('--burnout-epochs', type=int, default=-1)

        # competence-based CL
        parser.add_argument('--competent-cl', type=int, default=0)
        parser.add_argument('--hardness', type=str, default='median_prob',
                            choices=['median_prob', 'min_prob', 'sum_log_prob', 'avg_prob'])
        # fmt: on

    def _print(self, x):
        return " ".join(["{:.6f}".format(xx.item()) for xx in x])

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
        logger.info("baseline loss = {}".format(self.loss_baselines))
        self.register_buffer('valid_losses', torch.zeros(self.n_groups))
        self.register_buffer('sum_losses', torch.zeros(self.n_groups))  # historical loss sum over category
        self.register_buffer('count_cat', torch.ones(self.n_groups))

    def set_p_train(self, data_ratios):
        if self.p_train is not None:
            return
        logger.info("reloaded sum_losses = {}".format(self._print(self.sum_losses)))
        logger.info("reloaded valid losses = {}".format(self._print(self.valid_losses)))
        self.p_train = torch.Tensor(data_ratios).to(self.device)
        self.generalization_errors = self.sum_losses - self.valid_losses

    def set_valid_baselines(self, baselines):
        self.valid_losses.copy_(baselines)
        self.generalization_errors = self.valid_losses - self.sum_losses
        logger.info("gen error (valid-train) = {}".format(self._print(self.generalization_errors)))

    def get_generalization_errors(self):
        return self.generalization_errors

    def update_mw(self, epoch):
        self.epochs = epoch
        if epoch == 1:
            return None
        # version that uses EMA. (sum_losses is EMA running loss, count_cat is EMA running sum)
        past_losses = self.sum_losses - self.loss_baselines
        if past_losses.max() < 0:
            past_losses = self.sum_losses

        rho = self.rho
        p_train = self.p_train

        if hasattr(self, 'min_prob'):
            min_prob = self.min_prob
        else:
            min_prob = 0.2

        def p(eta):
            pp = p_train * torch.relu(past_losses - eta)
            q = pp / pp.sum()
            cq = torch.clamp(q / p_train, min=min_prob)
            return cq * p_train / (cq * p_train).sum()

        def bisection_target(eta):
            pp = p(eta)
            return 0.5 * ((pp / p_train - 1) ** 2 * p_train).sum() - rho

        eta_min = -(1.0 / (np.sqrt(2 * rho + 1) - 1)) * past_losses.max()
        eta_max = past_losses.max()
        eta_star = bisection(
            eta_min, eta_max, bisection_target,
            tol=self.tol, max_iter=1000)

        q = p(eta_star)
        if hasattr(self, 'clamp_q_to_min') and self.clamp_q_to_min:
            q = torch.clamp(q, min=torch.min(self.p_train).item())
            q = q / q.sum()

        self.temp_idx += 1
        if self.logging:
            logger.info("EMA before-baseline losses: {}".format(
                " ".join(["{:.6f}".format(xx.item()) for xx in self.sum_losses[0:self.n_groups]])))
            logger.info("EMA after-baseline losses: {}".format(" ".join(["{:.6f}".format(xx.item()) for xx in past_losses[0:self.n_groups]])))
            logger.info("EMA group fractions: {}".format(" ".join(["{:.6f}".format(xx.item()) for xx in self.p_train[0:self.n_groups]])))
            sum_weights = q[0:self.n_groups].sum().item()
            logger.info("Group loss weights: {}".format(" ".join(["{:.6f}".format(xx.item() / sum_weights) for xx in q[0:self.n_groups]])))

        if self.args.clear_history:
            self.sum_losses.zero_()
        # self.count_cat.fill_(1.)

        if epoch <= self.warmup_epochs:
            return None
        else:
            return q

    def individual_losses(self, model, net_output, sample):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1)

        if self.eps > 0.0:
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=False,
            )
            return loss, nll_loss
        else:
            loss = F.nll_loss(lprobs, target, ignore_index=self.padding_idx, reduction='none')
            return loss, loss

    def individual_mix_losses(self, model, net_output, sample, lambda_):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        loss_a, nll_loss_a = label_smoothed_nll_loss(
            lprobs, sample["target_a"].view(-1, 1), self.eps, ignore_index=self.padding_idx, reduce=False,
        )
        loss_b, nll_loss_b = label_smoothed_nll_loss(
            lprobs, sample["target_b"].view(-1, 1), self.eps, ignore_index=self.padding_idx, reduce=False,
        )

        bsz, slen = sample["target_a"].size()
        assert bsz == len(sample["id"])
        loss_a = loss_a.reshape(bsz, slen)
        nll_loss_a = nll_loss_a.reshape(bsz, slen)
        loss_b = loss_b.reshape(bsz, slen)
        nll_loss_b = nll_loss_b.reshape(bsz, slen)
        assert lambda_.size() == (bsz,)

        loss = loss_a * lambda_.view(-1, 1) + loss_b * (1 - lambda_).view(-1, 1)
        nll_loss = nll_loss_a * lambda_.view(-1, 1) + nll_loss_b * (1 - lambda_).view(-1, 1)
        return loss, nll_loss

    def retrieve_group_labels(self, sample):
        if self.group_level == "source_lang":
            index = sample["src_lang_id"]

        elif self.group_level == "target_lang":
            index = sample["tgt_lang_id"]
        else:
            index = sample['target'].view(-1)
        return index

    def compute_loss(self, model, sample, reduce=True):
        if 'lambda_' in sample and sample['lambda_'] is not None:
            net_output = model.mix(sample['lambda_'],
                                   sample["net_input_a"]["src_tokens"],
                                   sample["net_input_a"]["src_lengths"],
                                   sample["net_input_b"]["src_tokens"],
                                   sample["net_input_b"]["src_lengths"],
                                   sample["net_input_a"]["prev_output_tokens"],
                                   sample["net_input_b"]["prev_output_tokens"])
            token_loss, nll_loss = self.individual_mix_losses(model, net_output, sample, sample['lambda_'])
            target_kw = 'target_a'
        else:
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

    def simple_loss(self, model, net_output, sample, reduce=True):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = model.get_targets(sample, net_output).view(-1, 1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs, target, self.eps, ignore_index=self.padding_idx, reduce=reduce,
        )
        return loss, nll_loss

    def forward(self, model, sample, reduce=True, train_dynamic=False):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        if self.p_train is None:
            self.p_train = torch.Tensor(self.task.data_manager.data_ratios).to(self.device)
            logger.info("Fixed P train = {}".format(self.p_train))

        nsentences = sample['id'].size(0)
        sample_size = sample['ntokens']

        if not self.training:
            if train_dynamic:
                nll_loss, ind_loss, group_losses, group_counts = self.compute_loss(model, sample, reduce=False)
                word_mask = (sample['target'] != self.padding_idx).float()
                pad_mask = (sample['target'] == self.padding_idx)
                probs = torch.exp(-nll_loss)
                if self.args.hardness == 'median_prob':
                    probs.masked_fill_(pad_mask, float('inf'))
                    median_indices = word_mask.sum(1, keepdim=True).long() // 2
                    sorted_probs, _ = torch.sort(probs, dim=-1)
                    hardness_metrics = torch.gather(sorted_probs, 1, median_indices).squeeze(-1)
                elif self.args.hardness == 'avg_prob':
                    probs[pad_mask] = 0
                    hardness_metrics = torch.sum(probs, dim=-1) / word_mask.sum(1)
                elif self.args.hardness == 'min_prob':
                    probs.masked_fill_(pad_mask, float('inf'))
                    hardness_metrics, _ = probs.min(dim=-1)
                elif self.args.hardness == 'sum_log_prob':
                    hardness_metrics = -nll_loss.sum(1)
                else:
                    raise NotImplementedError
            else:
                nll_loss, ind_loss, group_losses, group_counts = self.compute_loss(model, sample)
                fg_labels = self.retrieve_group_labels(sample)
                fg_zero_vec = torch.zeros(self.n_groups, device='cuda')
                fg_group_nll = fg_zero_vec.scatter_add(0, fg_labels, nll_loss)
                fg_group_count = group_counts.detach().clone()
                fg_loss_vec = group_losses
        else:
            self.update_steps += 1
            nll_loss, ind_loss, group_losses, group_counts = self.compute_loss(model, sample)
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
            self.count_cat[valid_index] = valid_counts.add(group_counts[valid_index])

        nll_loss = nll_loss.sum()
        loss = ind_loss.sum()
        logging_output = {
            'loss': loss.data,
            'nll_loss': nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': nsentences,
            'sample_size': sample_size,
            'n_groups': self.n_groups,
            'gpu_count': 1,
        }
        if train_dynamic:
            return loss, sample_size, logging_output, sample['concat_ds_id'], hardness_metrics

        if self.logging and not self.training:
            for ii in range(self.n_groups):
                logging_output["fg_gnll{}".format(ii)] = fg_group_nll[ii].data
                logging_output["fg_gcount{}".format(ii)] = fg_group_count[ii].data
                logging_output["fg_gloss{}".format(ii)] = fg_loss_vec[ii].data

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