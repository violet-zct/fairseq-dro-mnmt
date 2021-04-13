# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import datetime
import time

import torch
from fairseq.data import (
    data_utils,
    FairseqDataset,
    iterators,
    LanguagePairDataset,
    ListDataset,
    encoders,
)

from torch.distributions.beta import Beta
from fairseq.tasks import register_task, LegacyFairseqTask
from fairseq.data.multilingual.sampling_method import SamplingMethod
from fairseq.data.multilingual.multilingual_data_manager import MultilingualDatasetManager

import numpy as np
from fairseq import metrics, utils
from argparse import Namespace
import json
import os
EVAL_BLEU_ORDER = 4


###
def get_time_gap(s, e):
    return (datetime.datetime.fromtimestamp(e) - datetime.datetime.fromtimestamp(s)).__str__()
###


logger = logging.getLogger(__name__)


@register_task('translation_multi_simple_epoch')
class TranslationMultiSimpleEpochTask(LegacyFairseqTask):
    """
    Translate from one (source) language to another (target) language.

    Args:
        langs (List[str]): a list of languages that are being supported
        dicts (Dict[str, fairseq.data.Dictionary]): mapping from supported languages to their dictionaries
        training (bool): whether the task should be configured for training or not

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='inference source language')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='inference target language')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('--keep-inference-langtok', action='store_true',
                            help='keep language tokens in inference output (e.g. for analysis or debugging)')

        # options for mixed data augmentation
        parser.add_argument('--aug-option', default="none", type=str,
                            choices=['in_group', 'global', 'none'])
        parser.add_argument('--mix-beta-type', default='fixed', type=str, choices=['fixed', 'gen_strength'])
        parser.add_argument('--beta-dist-alpha', default=0.2, type=float)

        # options for reporting BLEU during validation
        parser.add_argument('--eval-bleu', action='store_true',
                            help='evaluation with BLEU scores')
        parser.add_argument('--eval-bleu-detok', type=str, default="space",
                            help='detokenize before computing BLEU (e.g., "moses"); '
                                 'required if using --eval-bleu; use "space" to '
                                 'disable detokenization; see fairseq.data.encoders '
                                 'for other options')
        parser.add_argument('--eval-bleu-detok-args', type=str, metavar='JSON',
                            help='args for building the tokenizer, if needed')
        parser.add_argument('--eval-tokenized-bleu', action='store_true', default=False,
                            help='compute tokenized BLEU instead of sacrebleu')
        parser.add_argument('--eval-bleu-remove-bpe', nargs='?', const='@@ ', default=None,
                            help='remove BPE before computing BLEU')
        parser.add_argument('--eval-bleu-args', type=str, metavar='JSON',
                            help='generation args for BLUE scoring, '
                                 'e.g., \'{"beam": 4, "lenpen": 0.6}\'')
        parser.add_argument('--eval-bleu-print-samples', action='store_true',
                            help='print sample generations during validation')
        SamplingMethod.add_arguments(parser)
        MultilingualDatasetManager.add_args(parser)
        # fmt: on

    def __init__(self, args, langs, dicts, training):
        super().__init__(args)
        self.langs = langs
        self.dicts = dicts
        self.training = training
        if training:
            self.lang_pairs = args.lang_pairs
        else:
            self.lang_pairs = ['{}-{}'.format(args.source_lang, args.target_lang)]

        self.pseudo_source_lang, self.pseudo_target_lang = self.lang_pairs[0].split("-")
        # eval_lang_pairs for multilingual translation is usually all of the
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = self.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = self.lang_pairs
        self.sampling_method = SamplingMethod.build_sampler(args, self)
        self.data_manager = MultilingualDatasetManager.setup_data_manager(
            args, self.lang_pairs, langs, dicts, self.sampling_method)

    @classmethod
    def setup_task(cls, args, **kwargs):
        langs, dicts, training = MultilingualDatasetManager.prepare(
            cls.load_dictionary, args, **kwargs
        )
        return cls(args, langs, dicts, training)

    def has_sharded_data(self, split):
        return self.data_manager.has_sharded_data(split)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        if split in self.datasets:
            dataset = self.datasets[split]
            if self.has_sharded_data(split) and dataset.load_next_shard:
                shard_epoch = dataset.shard_epoch
            else:
                # no need to load next shard so skip loading
                # also this avoid always loading from beginning of the data
                return
        else:
            shard_epoch = None
        logger.info(f'loading data for {split} epoch={epoch}/{shard_epoch}')
        logger.info(f"mem usage: {data_utils.get_mem_usage()}")
        if split in self.datasets:
            del self.datasets[split]
            logger.info('old dataset deleted manually')
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")
        self.datasets[split] = self.data_manager.load_sampled_multi_epoch_dataset(
            split,
            self.training,
            epoch=epoch, combine=combine, shard_epoch=shard_epoch, **kwargs
        )

        if split == 'train' and hasattr(self.args, 'compute_train_dynamics') and self.args.compute_train_dynamics:
            opt_path = os.path.join(self.args.save_dir, 'info.opt')
            if not os.path.exists(opt_path):
                with open(opt_path, "w") as fout:
                    for ii in range(len(self.datasets[split])):
                        ds_idx, _ = self.datasets[split][ii]
                        fout.write("{}\n".format(self.datasets[split].keys[ds_idx]))

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            raise NotImplementedError("Constrained decoding with the multilingual_translation task is not supported")

        src_data = ListDataset(src_tokens, src_lengths)
        dataset = LanguagePairDataset(src_data, src_lengths, self.source_dictionary)
        src_langtok_spec, tgt_langtok_spec = self.args.langtoks['main']
        if self.args.lang_tok_replacing_bos_eos:
            dataset = self.data_manager.alter_dataset_langtok(
                    dataset,
                    src_eos=self.source_dictionary.eos(),
                    src_lang=self.args.source_lang,
                    tgt_eos=self.target_dictionary.eos(),
                    tgt_lang=self.args.target_lang,
                    src_langtok_spec=src_langtok_spec,
                    tgt_langtok_spec=tgt_langtok_spec,
                )
        else:
            dataset.src = self.data_manager.src_dataset_tranform_func(
                self.args.source_lang,
                self.args.target_lang,
                dataset=dataset.src,
                spec=src_langtok_spec,
                )
        return dataset

    def build_generator(
        self, models, args,
        seq_gen_cls=None, extra_gen_cls_kwargs=None,
    ):
        if not getattr(args, 'keep_inference_langtok', False):
            _, tgt_langtok_spec = self.args.langtoks['main']
            if tgt_langtok_spec:
                tgt_lang_tok = self.data_manager.get_decoder_langtok(self.args.target_lang, tgt_langtok_spec)
                extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}
                extra_gen_cls_kwargs['symbols_to_strip_from_output'] = {tgt_lang_tok}

        return super().build_generator(
            models, args,
            seq_gen_cls=None,
            extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_model(self, args):
        model = super().build_model(args)
        if getattr(args, 'eval_bleu', False):
            assert getattr(args, 'eval_bleu_detok', None) is not None, (
                '--eval-bleu-detok is required if using --eval-bleu; '
                'try --eval-bleu-detok=moses (or --eval-bleu-detok=space '
                'to disable detokenization, e.g., when using sentencepiece)'
            )
            detok_args = json.loads(getattr(args, 'eval_bleu_detok_args', '{}') or '{}')
            self.tokenizer = encoders.build_tokenizer(Namespace(
                tokenizer=getattr(args, 'eval_bleu_detok', None),
                **detok_args
            ))

            gen_args = json.loads(getattr(args, 'eval_bleu_args', '{}') or '{}')
            self.sequence_generator = self.build_generator([model], Namespace(**gen_args))
        return model

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        model.train()
        model.set_num_updates(update_num)

        def reorder_dict(x, ordered_indices):
            return {key: value[ordered_indices] for key, value in x.items()}

        def get_reg_strength(gen_errors):
            # self.args.beta_dist_alpha = max_strength, 0.4
            return torch.sigmoid(gen_errors) * self.args.beta_dist_alpha

        if self.args.aug_option == "in_group":
            if self.args.group_level == "source_lang":
                group_idx = sample["src_lang_id"]
            else:
                group_idx = sample["tgt_lang_id"]
            uniq_groups = torch.unique(group_idx, sorted=True)

            inds = torch.arange(len(group_idx)).to(group_idx.device)
            shuffled_inds = torch.randperm(len(group_idx)).to(group_idx.device)
            shuffled_group_idx = group_idx[shuffled_inds]
            inds_a = torch.cat([inds[group_idx == gid] for gid in uniq_groups])
            inds_b = torch.cat([shuffled_inds[shuffled_group_idx == gid] for gid in uniq_groups])

        elif self.args.aug_option == "global":
            inds_b = torch.randperm(len(sample['id'])).to(sample['id'].device)
            inds_a = None
        else:
            inds_a = inds_b = None

        if inds_a is not None or inds_b is not None:
            new_sample = {
                'id': sample['id'] if inds_a is None else sample['id'][inds_a],
                'nsentences': sample["nsentences"],
                'ntokens': sample["ntokens"],
                'net_input_a': sample['net_input'] if inds_a is None else reorder_dict(sample['net_input'], inds_a),
                'net_input_b': reorder_dict(sample['net_input'], inds_b),
                'target_a': sample['target'] if inds_a is None else sample['target'][inds_a],
                'target_b': sample['target'][inds_b],
            }
            if 'src_lang_id' in sample:
                new_sample['src_lang_id'] = sample['src_lang_id'][inds_a] if inds_a is not None else sample['src_lang_id']
            if 'tgt_lang_id' in sample:
                new_sample['tgt_lang_id'] = sample['tgt_lang_id'][inds_a] if inds_a is not None else sample['tgt_lang_id']

            bsz = len(sample['id'])
            device = sample['id'].device
            if self.args.mix_beta_type == "fixed":
                dist = Beta(self.args.beta_dist_alpha, self.args.beta_dist_alpha)
                lambda_ = dist.sample(sample_shape=[bsz]).to(device)
            else:
                lang_ids = new_sample['src_lang_id'] if self.args.group_level == "source_lang" else new_sample['tgt_lang_id']
                uniq_groups = torch.unique(lang_ids, sorted=True)
                gen_error = criterion.get_generalization_errors()
                lambda_ = torch.ones(bsz).to(device)
                if gen_error is not None:
                    alphas = get_reg_strength(gen_error)
                    for gid in uniq_groups:
                        dist = Beta(alphas[gid], alphas[gid])
                        valid_ids = (lang_ids == gid)
                        lambda_[valid_ids] = dist.sample(sample_shape=[sum(valid_ids.long())]).to(device)

            lambda_ = torch.max(lambda_, 1 - lambda_)
            if self.args.fp16:
                lambda_ = lambda_.half()
            new_sample['lambda_'] = lambda_
            sample = new_sample
        else:
            sample['lambda_'] = None

        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output['_bleu_ref_len'] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output['_bleu_counts_' + str(i)] = bleu.counts[i]
                logging_output['_bleu_totals_' + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def train_dynamic_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            loss, sample_size, logging_output, sample_ids, average_p, median_p, avg_entropy = \
                criterion(model, sample, train_dynamic=True)
        return loss, sample_size, logging_output, sample_ids, average_p, median_p, avg_entropy

    def inference_step(self, generator, models, sample, prefix_tokens=None, constraints=None):
        with torch.no_grad():
            _, tgt_langtok_spec = self.args.langtoks['main']
            if not self.args.lang_tok_replacing_bos_eos:
                if prefix_tokens is None and tgt_langtok_spec:
                    tgt_lang_tok = self.data_manager.get_decoder_langtok(self.args.target_lang, tgt_langtok_spec)
                    src_tokens = sample['net_input']['src_tokens']
                    bsz = src_tokens.size(0)
                    prefix_tokens = torch.LongTensor(
                        [[tgt_lang_tok]]
                        ).expand(bsz, 1).to(src_tokens)
                return generator.generate(
                        models,
                        sample,
                        prefix_tokens=prefix_tokens,
                        constraints=constraints,
                )
            else:
                return generator.generate(
                        models,
                        sample,
                        prefix_tokens=prefix_tokens,
                        bos_token=self.data_manager.get_decoder_langtok(self.args.target_lang, tgt_langtok_spec)
                        if tgt_langtok_spec else self.target_dictionary.eos(),
                )

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    @property
    def source_dictionary(self):
        return self.dicts[self.pseudo_source_lang]

    @property
    def target_dictionary(self):
        return self.dicts[self.pseudo_target_lang]

    def create_batch_sampler_func(
        self, max_positions, ignore_invalid_inputs,
        max_tokens, max_sentences,
        required_batch_size_multiple=1,
        seed=1,
    ):
        def construct_batch_sampler(
            dataset, epoch
        ):
            splits = [s for s, _ in self.datasets.items() if self.datasets[s] == dataset]
            split = splits[0] if len(splits) > 0 else None
            # NEW implementation
            if epoch is not None:
                # initialize the dataset with the correct starting epoch
                dataset.set_epoch(epoch)

            # get indices ordered by example size
            start_time = time.time()
            logger.info(f"start batch sampler: mem usage: {data_utils.get_mem_usage()}")

            with data_utils.numpy_seed(seed):
                indices = dataset.ordered_indices()
            logger.info(f'[{split}] @batch_sampler order indices time: {get_time_gap(start_time, time.time())}')
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            # filter examples that are too large
            if max_positions is not None:
                my_time = time.time()
                indices = self.filter_indices_by_size(
                    indices, dataset, max_positions, ignore_invalid_inputs
                )
                logger.info(f'[{split}] @batch_sampler filter_by_size time: {get_time_gap(my_time, time.time())}')
                logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            # create mini-batches with given size constraints
            my_time = time.time()
            batch_sampler = dataset.batch_by_size(
                indices,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )

            logger.info(f'[{split}] @batch_sampler batch_by_size time: {get_time_gap(my_time, time.time())}')
            logger.info(f'[{split}] per epoch batch_sampler set-up time: {get_time_gap(start_time, time.time())}')
            logger.info(f"mem usage: {data_utils.get_mem_usage()}")

            return batch_sampler
        return construct_batch_sampler

    # we need to override get_batch_iterator because we want to reset the epoch iterator each time
    def get_batch_iterator(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1,
        seed=1, num_shards=1, shard_id=0, num_workers=0, epoch=1,
        data_buffer_size=0, disable_iterator_cache=False,
        reset_sample_ratios=None, new_iterator=False,
    ):
        """
        Get an iterator that yields batches of data from the given dataset.

        Args:
            dataset (~fairseq.data.FairseqDataset): dataset to batch
            max_tokens (int, optional): max number of tokens in each batch
                (default: None).
            max_sentences (int, optional): max number of sentences in each
                batch (default: None).
            max_positions (optional): max sentence length supported by the
                model (default: None).
            ignore_invalid_inputs (bool, optional): don't raise Exception for
                sentences that are too long (default: False).
            required_batch_size_multiple (int, optional): require batch size to
                be a multiple of N (default: 1).
            seed (int, optional): seed for random number generator for
                reproducibility (default: 1).
            num_shards (int, optional): shard the data iterator into N
                shards (default: 1).
            shard_id (int, optional): which shard of the data iterator to
                return (default: 0).
            num_workers (int, optional): how many subprocesses to use for data
                loading. 0 means the data will be loaded in the main process
                (default: 0).
            epoch (int, optional): the epoch to start the iterator from
                (default: 0).
            data_buffer_size (int, optional): number of batches to
                preload (default: 0).
            disable_iterator_cache (bool, optional): don't cache the
                EpochBatchIterator (ignores `FairseqTask::can_reuse_epoch_itr`)
                (default: False).
        Returns:
            ~fairseq.iterators.EpochBatchIterator: a batched iterator over the
                given dataset split
        """
        # initialize the dataset with the correct starting epoch
        assert isinstance(dataset, FairseqDataset)
        if new_iterator:
            if "special" in self.dataset_to_epoch_iter:
                return self.dataset_to_epoch_iter["special"]
        elif dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]

        if (
            self.args.sampling_method == 'RoundRobin'
        ):
            batch_iter = super().get_batch_iterator(
                dataset,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                data_buffer_size=data_buffer_size,
                disable_iterator_cache=disable_iterator_cache,
            )
            self.dataset_to_epoch_iter[dataset] = batch_iter
            return batch_iter

        if reset_sample_ratios is not None:
            # dataset is the train split
            dataset.adjust_sampling(None, reset_sample_ratios, None)

        construct_batch_sampler = self.create_batch_sampler_func(
            max_positions, ignore_invalid_inputs,
            max_tokens, max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
            seed=seed,
        )

        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=construct_batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )

        if new_iterator:
            self.dataset_to_epoch_iter['special'] = epoch_iter
        elif ('valid' in self.datasets and dataset == self.datasets['valid']) or self.args.sampling_method == 'concat':
            self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:

            def sum_logs(key):
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs('_bleu_counts_' + str(i)))
                totals.append(sum_logs('_bleu_totals_' + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                counts = [c.item() if torch.is_tensor(c) else c for c in counts]
                totals = [c.item() if torch.is_tensor(c) else c for c in totals]
                metrics.log_scalar('_bleu_counts', np.array(counts))
                metrics.log_scalar('_bleu_totals', np.array(totals))
                metrics.log_scalar('_bleu_sys_len', sum_logs('_bleu_sys_len'))
                metrics.log_scalar('_bleu_ref_len', sum_logs('_bleu_ref_len'))

                def compute_bleu(meters):
                    import inspect
                    import sacrebleu
                    fn_sig = inspect.getfullargspec(sacrebleu.compute_bleu)[0]
                    if 'smooth_method' in fn_sig:
                        smooth = {'smooth_method': 'exp'}
                    else:
                        smooth = {'smooth': 'exp'}
                    bleu = sacrebleu.compute_bleu(
                        correct=meters['_bleu_counts'].sum,
                        total=meters['_bleu_totals'].sum,
                        sys_len=meters['_bleu_sys_len'].sum,
                        ref_len=meters['_bleu_ref_len'].sum,
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived('bleu', compute_bleu)

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu

        def decode(toks, escape_unk=False):
            s = self.target_dictionary.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                ),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(decode(
                utils.strip_pad(sample['target'][i], self.target_dictionary.pad()),
                escape_unk=True,  # don't count <unk> as matches to the hypo
            ))
        if self.args.eval_bleu_print_samples:
            logger.info('example hypothesis: ' + hyps[0])
            logger.info('example reference: ' + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize='none')
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])