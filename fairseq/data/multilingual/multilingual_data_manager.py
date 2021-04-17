# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import json
import logging
import os
from collections import OrderedDict

import numpy as np
from fairseq import options, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    Dictionary,
    LanguagePairDataset,
    PrependTokenDataset,
    SampledMultiDataset,
    SampledMultiEpochDataset,
    StripTokenDataset,
    TransformEosLangPairDataset,
    TruncateDataset,
    data_utils,
    indexed_dataset,
)
from fairseq.data.multilingual.sampled_multi_dataset import CollateFormat
from fairseq.file_io import PathManager
from fairseq.options import csv_str_list, eval_str_dict


logger = logging.getLogger(__name__)


def _lang_token(lang: str, style="__{}__"):
    return style.format(lang)


def _lang_token_index(dic: Dictionary, lang: str, style="__{}__"):
    """Return language token index."""
    idx = dic.index(_lang_token(lang, style))
    assert idx != dic.unk_index, "cannot find language token for lang {}".format(lang)
    return idx


def _lang_id(dic: Dictionary, lang: str):
    """Return language ID index."""
    idx = dic.index(lang)
    assert idx != dic.unk_index, "cannot find language ID for lang {}".format(lang)
    # minus unk index
    return idx - 1


def load_sampling_weights(from_file):
    with open(from_file) as f:
        weights = json.load(f)
    return weights


class MultilingualDatasetManager(object):
    def __init__(self, args, lang_pairs, langs, dicts, sampling_method):
        super().__init__()
        self.args = args
        self.seed = args.seed
        self.lang_pairs = lang_pairs
        self.langs = langs
        self.dicts = dicts
        self.src_langs = sorted(list(set([langpair.split("-")[0] for langpair in lang_pairs])))
        self.tgt_langs = sorted(list(set([langpair.split("-")[1] for langpair in lang_pairs])))

        self.src_lang_dict = self.create_lang_dictionary(self.src_langs)
        self.tgt_lang_dict = self.create_lang_dictionary(self.tgt_langs)

        # used for sanity check
        low_langs = ['aze', 'bel', 'glg', 'slk', 'bos', 'mar', 'hin', 'mkd']
        if len(self.src_langs) > 1:
            self.lang_weights = [1.0] * len(self.src_langs)
            for lang in low_langs:
                if lang in self.src_langs:
                    self.lang_weights[_lang_id(self.src_lang_dict, lang)] = 2.0
        else:
            self.lang_weights = [1.0] * len(self.tgt_langs)
            for lang in low_langs:
                if lang in self.tgt_langs:
                    self.lang_weights[_lang_id(self.tgt_lang_dict, lang)] = 2.0
        #
        self.sampling_method = sampling_method
        self.sampling_scheduler = None
        self._has_sharded_data = False
        self._num_shards_dict = {}

        self.uniq_token_counts = [-1] * len(self.tgt_langs)

        if os.path.exists(os.path.join(args.data, "uniq.token.counts")):
            with open(os.path.join(args.data, "uniq.token.counts")) as fin:
                for line in fin:
                    key, value = line.strip().split()
                    if key in self.tgt_langs:
                        self.uniq_token_counts[_lang_id(self.tgt_lang_dict, key)] = float(value)

        if hasattr(args, 'group_level') or hasattr(args, 'outer_group_level'):
            self.target_group = args.group_level if hasattr(args, 'group_level') else args.outer_group_level
            self.lang_dict = self.tgt_lang_dict if self.target_group == "target_lang" else self.src_lang_dict
            n_groups = len(self.tgt_langs) if self.target_group == "target_lang" else len(self.src_langs)

        if args.outer_baseline_file is not None:
            outer_baseline = np.zeros(n_groups)
            with open(args.outer_baseline_file) as fin:
                for line in fin:
                    fields = line.strip().split("=")
                    lang = _lang_id(self.lang_dict, fields[0])
                    outer_baseline[lang] = float(fields[1])
            self.outer_baseline = outer_baseline
        else:
            self.outer_baseline = None

        if args.inner_baseline_file is not None:
            fmat = open(args.inner_baseline_file).readlines()
            vocab_size = len(fmat[0].strip().split("=")[-1].split())
            inner_baseline = np.zeros((n_groups, vocab_size))
            for line in fmat:
                fields = line.strip().split("=")
                lang = _lang_id(self.lang_dict, fields[0])
                bls = np.array(list(map(float, fields[1].split())))
                inner_baseline[lang] = bls
            self.inner_baseline = inner_baseline
        else:
            self.inner_baseline = None

    @classmethod
    def setup_data_manager(cls, args, lang_pairs, langs, dicts, sampling_method):
        return MultilingualDatasetManager(
            args, lang_pairs, langs, dicts, sampling_method
        )

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--langs",
            default=None,
            type=csv_str_list,
            help="a list of languages comma sperated languages which can appear in lang-pairs; "
            "note that the ordering determines language token IDs",
        )
        parser.add_argument(
            "--lang-dict",
            default=None,
            type=str,
            help="an external file which contains a list of "
            "languages which can appear in lang-pairs; "
            "note that the ordering determines language token IDs; "
            "--langs and --lang-dict are two exclusive options",
        )
        parser.add_argument(
            "--lang-tok-style",
            default="multilingual",
            type=str,
            choices=["multilingual", "mbart"],
            help="language token styles",
        )

        parser.add_argument(
            "--load-alignments",
            action="store_true",
            help="load the binarized alignments",
        )
        parser.add_argument(
            "--left-pad-source",
            default="True",
            type=str,
            metavar="BOOL",
            help="pad the source on the left",
        )
        parser.add_argument(
            "--left-pad-target",
            default="False",
            type=str,
            metavar="BOOL",
            help="pad the target on the left",
        )
        parser.add_argument(
            "--max-source-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--upsample-primary",
            default=1,
            type=int,
            help="amount to upsample primary dataset",
        )
        parser.add_argument(
            "--truncate-source",
            action="store_true",
            default=False,
            help="truncate source to max-source-positions",
        )
        parser.add_argument(
            "--encoder-langtok",
            default=None,
            type=str,
            choices=["src", "tgt"],
            metavar="SRCTGT",
            help="prepend to the beginning of source sentence the source or target "
            "language token. (src/tgt)",
        )
        parser.add_argument(
            "--decoder-langtok",
            action="store_true",
            help="prepend to the beginning of target sentence the target language token",
        )
        parser.add_argument(
            "--lang-tok-replacing-bos-eos", action="store_true", default=False
        )
        parser.add_argument(
            "--enable-lang-ids",
            default=True,
            action="store_true",
            help="whether to include language IDs in samples",
        )
        parser.add_argument(
            "--enable-reservsed-directions-shared-datasets",
            default=False,
            action="store_true",
            help="whether to allow datasets be used in reversed directions",
        )

        parser.add_argument(
            "--extra-data",
            help='a dictionary of data name to this path, \
                            e.g. {"mined", path_to_mined_data, "denoised": path_to_denoised_data}',
            type=lambda uf: eval_str_dict(uf, type=str),
            default=None,
        )
        parser.add_argument(
            "--extra-lang-pairs",
            help='a dictionary of data name to the language pairs they serve, \
                            e.g. {"mined": comma-separated-lang-pairs, "denoised":  comma-separated-lang-pairs}',
            type=lambda uf: eval_str_dict(uf, type=str),
            default=None,
        )
        parser.add_argument(
            "--langtoks-specs",
            help='a list of comma separated data types that a set of language tokens to be specialized for, \
                            e.g. "main,dae,mined". There will be a set of language tokens added to the vocab to \
                            distinguish languages in different training data types. If not specified, default language \
                            tokens per languages will be added',
            default="main",
            type=csv_str_list,
        )
        parser.add_argument(
            "--langtoks",
            help='a dictionary of how to add language tokens, \
                            e.g. {"mined": (None, "tgt"), "mono_dae": ("src.dae", "tgt"), "main": \
                            ("src", "tgt")}, or {"mined": ("src.mined", "tgt")}',
            default=None,
            type=lambda uf: eval_str_dict(uf, type=str),
        )
        parser.add_argument(
            "--sampling-weights-from-file",
            help='a file contain a python dictionary of how to sample data sets, \
                                e.g. { "main:en_XX-es_XX": 0.2, "mined:en_XX-pt_XX": 0.5, \
                                    "mono_dae:es_XX-es_XX: 0.3, "main:en_xx-fr_XX": 0.8 }',
            default=None,
            type=str,
        )
        parser.add_argument(
            "--sampling-weights",
            help='a dictionary of how to sample data sets, \
                            e.g. { "main:en_XX-es_XX": 0.2, "mined:en_XX-pt_XX": 0.5, \
                                   "mono_dae:es_XX-es_XX: 0.3, "main:en_xx-fr_XX": 0.8 }',
            default=None,
            type=lambda uf: eval_str_dict(uf, type=str),
        )
        parser.add_argument(
            "--virtual-epoch-size",
            default=10000000,
            type=int,
            help="virtual epoch size to speed up data loading",
        )
        parser.add_argument(
            "--virtual-data-size",
            default=None,
            type=int,
            help="virtual data size of the whole joint dataset to speed"
            "up data loading and have specific dynamic sampling strategy interval",
        )

        parser.add_argument('--max-scale-up', type=float, default=1.5,
                            help="max scale ratio when doing resampling")
        parser.add_argument('--exclude-c', type=float, default=0)

        parser.add_argument(
            '--inner-baseline-file',
            default=None, type=str
        )
        parser.add_argument(
            '--outer-baseline-file',
            default=None, type=str
        )
    @classmethod
    def load_langs(cls, args, **kwargs):
        if args.lang_dict and args.langs:
            raise ValueError("--langs and --lang-dict can not both be specified")
        if args.lang_dict is None and args.langs is None:
            logger.warning(
                "External language dictionary is not provided; "
                "use lang-pairs to infer the set of supported languages. "
                "The language ordering is not stable which might cause "
                "misalignment in pretraining and finetuning."
            )
            # infer from lang_pairs as it is
            langs = list(
                {x for lang_pair in args.lang_pairs for x in lang_pair.split("-")}
            )
            langs = sorted(langs)
            logger.info(f"inferred language list: {langs}")
        elif args.lang_dict:
            with PathManager.open(args.lang_dict, "r", encoding="utf-8") as f:
                langs = [lang.strip() for lang in f.readlines() if lang.strip()]
                logger.info(
                    f"loaded language list from {args.lang_dict} as they are ordered in file"
                )
        elif args.langs:
            langs = args.langs
            logger.info(
                f"parsed the language list as they are ordered in the option: {langs}"
            )
        return langs

    def has_sharded_data(self, split):
        return self._has_sharded_data and split == getattr(
            self.args, "train_subset", None
        )

    def _shared_collater(self):
        return not (self.args.extra_data and "mono_dae" in self.args.extra_data) and (
            not self.args.lang_tok_replacing_bos_eos
        )

    @classmethod
    def prepare(cls, load_dictionary, args, **kargs):
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        if not hasattr(args, "shuffle_instance"):
            args.shuffle_instance = False
        if args.langtoks is None:
            args.langtoks = {}
        if "main" not in args.langtoks:
            src_langtok_spec = args.encoder_langtok if args.encoder_langtok else None
            tgt_langtok_spec = "tgt" if args.decoder_langtok else None
            args.langtoks["main"] = (src_langtok_spec, tgt_langtok_spec)

        def check_langs(langs, pairs):
            messages = []
            for src, tgt in pairs:
                if src not in langs or tgt not in langs:
                    messages.append(
                        f"language pair {src}-{tgt} contains languages "
                        "that are not in the language dictionary"
                    )
            if len(messages) > 0:
                raise ValueError(" ".join(messages) + f"; langs: {langs}")

        if args.lang_pairs is None:
            raise ValueError(
                "--lang-pairs is required. List all the language pairs in the training objective."
            )
        if isinstance(args.lang_pairs, str):
            args.lang_pairs = args.lang_pairs.split(",")
        if args.source_lang is not None or args.target_lang is not None:
            training = False
        else:
            training = True
        sorted_langs = cls.load_langs(args, **kargs)
        check_langs(
            sorted_langs,
            (
                [p.split("-") for p in args.lang_pairs]
                if training
                else [(args.source_lang, args.target_lang)]
            ),
        )

        # load dictionaries
        if training:
            extra_lang_pairs = (
                list(
                    {p for _, v in args.extra_lang_pairs.items() for p in v.split(",")}
                )
                if args.extra_lang_pairs
                else []
            )
            langs_to_load_dicts = sorted(
                {x for p in args.lang_pairs + extra_lang_pairs for x in p.split("-")}
            )
        else:
            langs_to_load_dicts = sorted([args.source_lang, args.target_lang])

        dicts = OrderedDict()
        supported_langtok_specs = args.langtoks_specs
        for lang in langs_to_load_dicts:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dicts[lang] = load_dictionary(
                os.path.join(paths[0], "dict.{}.txt".format(lang))
            )
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[langs_to_load_dicts[0]].pad()
                assert dicts[lang].eos() == dicts[langs_to_load_dicts[0]].eos()
                assert dicts[lang].unk() == dicts[langs_to_load_dicts[0]].unk()

            # keep the langs consistent for all experiments with the same lang dict
            # for finetuning regardless of whether lang_tok is required or not just add the tokens to the dicts
            for spec in supported_langtok_specs:
                for lang_to_add in sorted_langs:
                    dicts[lang].add_symbol(
                        MultilingualDatasetManager.get_lang_tok(lang_to_add, args, spec)
                    )
            if args.lang_tok_style == "mbart" or (
                args.extra_data and "mono_dae" in args.extra_data
            ):
                dicts[lang].add_symbol("<mask>")
            logger.info("[{}] dictionary: {} types".format(lang, len(dicts[lang])))
        return sorted_langs, dicts, training

    TOKEN_STYLES = {"mbart": "[{}]", "multilingual": "__{}__"}

    @classmethod
    def create_lang_dictionary(cls, langs):
        unk = "<unk>"
        # hack to remove symbols other than unk as they are not needed by lang dict
        lang_dict = Dictionary(pad=unk, eos=unk, unk=unk, bos=unk)
        for lang in langs:
            lang_dict.add_symbol(lang)
        return lang_dict

    @classmethod
    def get_lang_tok_style(cls, args):
        return cls.TOKEN_STYLES[args.lang_tok_style]

    @classmethod
    def get_lang_tok(cls, lang, args, spec=""):
        if spec is None:
            return None
        if spec.endswith("dae"):
            lang = f"{lang}_dae"
        elif spec.endswith("mined"):
            lang = f"{lang}_mined"
        return _lang_token(lang, cls.get_lang_tok_style(args))

    @classmethod
    def get_langtok_index(cls, lang_tok, dic):
        idx = dic.index(lang_tok)
        assert (
            idx != dic.unk_index
        ), "cannot find language token {} in the dictionary".format(lang_tok)
        return idx

    def get_encoder_langtok(self, src_lang, tgt_lang, spec=None):
        if spec is None:
            return None
        if spec and spec.startswith("src"):
            if src_lang is None:
                return None
            langtok = self.get_lang_tok(src_lang, self.args, spec)
        else:
            if tgt_lang is None:
                return None
            langtok = self.get_lang_tok(tgt_lang, self.args, spec)
        return self.get_langtok_index(langtok, self.dicts[src_lang])

    def get_decoder_langtok(self, tgt_lang, spec=None):
        if spec is None:
            return None
        langtok = self.get_lang_tok(tgt_lang, self.args, spec)
        return self.get_langtok_index(langtok, self.dicts[tgt_lang])

    @classmethod
    def load_data(cls, path, vdict, impl):
        dataset = data_utils.load_indexed_dataset(path, vdict, impl)
        return dataset

    @classmethod
    def split_exists(cls, split, src, tgt, lang, data_path, dataset_impl):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    @classmethod
    def mono_split_exists(cls, split, lang, data_path, dataset_impl):
        filename = os.path.join(data_path, "{}.{}".format(split, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    @classmethod
    def bitext_split_exists(cls, split, src, tgt, data_path, dataset_impl):
        src_exists = cls.split_exists(
            split, src, tgt, lang=src, data_path=data_path, dataset_impl=dataset_impl
        ) or cls.split_exists(
            split, tgt, src, lang=src, data_path=data_path, dataset_impl=dataset_impl
        )
        # check source exists to determine shard number
        # also note that during inference time target is not required
        # so checking target will fail inference time data loading
        return src_exists

    @classmethod
    def get_split_num_shards(cls, split, src, tgt, data_paths, dataset_impl):
        return sum(
            1
            for path in data_paths
            if cls.bitext_split_exists(split, src, tgt, path, dataset_impl)
        )

    @classmethod
    def get_mono_split_num_shards(cls, split, lang, data_paths, dataset_impl):
        return sum(
            1
            for path in data_paths
            if cls.mono_split_exists(split, lang, path, dataset_impl)
        )

    def load_lang_dataset(
        self,
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        max_source_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
    ):

        src_datasets = []
        tgt_datasets = []

        for k in itertools.count():
            split_k = split + (str(k) if k > 0 else "")

            # infer langcode
            if self.split_exists(split_k, src, tgt, src, data_path, dataset_impl):
                prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
            elif self.split_exists(split_k, tgt, src, src, data_path, dataset_impl):
                prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
            else:
                if k > 0:
                    break
                else:
                    logger.error(
                        f"Dataset not found: {data_path}, {split_k}, {src}, {tgt}"
                    )
                    raise FileNotFoundError(
                        "Dataset not found: {} ({})".format(split, data_path)
                    )

            src_dataset = self.load_data(prefix + src, src_dict, dataset_impl)
            if truncate_source:
                src_dataset = AppendTokenDataset(
                    TruncateDataset(
                        StripTokenDataset(src_dataset, src_dict.eos()),
                        max_source_positions - 1,
                    ),
                    src_dict.eos(),
                )
            src_datasets.append(src_dataset)
            tgt_datasets.append(self.load_data(prefix + tgt, tgt_dict, dataset_impl))

            logger.info(
                "{} {} {}-{} {} examples".format(
                    data_path, split_k, src, tgt, len(src_datasets[-1])
                )
            )

            if not combine:
                break

        assert len(src_datasets) == len(tgt_datasets)

        if len(src_datasets) == 1:
            src_dataset, tgt_dataset = src_datasets[0], tgt_datasets[0]
        else:
            sample_ratios = [1] * len(src_datasets)
            sample_ratios[0] = upsample_primary
            src_dataset = ConcatDataset(src_datasets, sample_ratios)
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)

        if prepend_bos:
            assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
            src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())

        align_dataset = None
        if load_alignments:
            align_path = os.path.join(
                data_path, "{}.align.{}-{}".format(split, src, tgt)
            )
            if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
                align_dataset = data_utils.load_indexed_dataset(
                    align_path, None, dataset_impl
                )

        return src_dataset, tgt_dataset, align_dataset

    def load_langpair_dataset(
        self,
        data_path,
        split,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        dataset_impl,
        upsample_primary,
        left_pad_source,
        left_pad_target,
        max_source_positions,
        max_target_positions,
        prepend_bos=False,
        load_alignments=False,
        truncate_source=False,
        src_dataset_transform_func=lambda dataset: dataset,
        tgt_dataset_transform_func=lambda dataset: dataset,
        src_lang_id=None,
        tgt_lang_id=None,
        langpairs_sharing_datasets=None,
    ):
        norm_direction = "-".join(sorted([src, tgt]))
        if langpairs_sharing_datasets is not None:
            src_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, src), "NotInCache"
            )
            tgt_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, tgt), "NotInCache"
            )
            align_dataset = langpairs_sharing_datasets.get(
                (data_path, split, norm_direction, src, tgt), "NotInCache"
            )

        # a hack: any one is not in cache, we need to reload them
        if (
            langpairs_sharing_datasets is None
            or src_dataset == "NotInCache"
            or tgt_dataset == "NotInCache"
            or align_dataset == "NotInCache"
            or split != getattr(self.args, "train_subset", None)
        ):
            # source and target datasets can be reused in reversed directions to save memory
            # reversed directions of valid and test data will not share source and target datasets
            src_dataset, tgt_dataset, align_dataset = self.load_lang_dataset(
                data_path,
                split,
                src,
                src_dict,
                tgt,
                tgt_dict,
                combine,
                dataset_impl,
                upsample_primary,
                max_source_positions=max_source_positions,
                prepend_bos=prepend_bos,
                load_alignments=load_alignments,
                truncate_source=truncate_source,
            )
            src_dataset = src_dataset_transform_func(src_dataset)
            tgt_dataset = tgt_dataset_transform_func(tgt_dataset)
            if langpairs_sharing_datasets is not None:
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, src)
                ] = src_dataset
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, tgt)
                ] = tgt_dataset
                langpairs_sharing_datasets[
                    (data_path, split, norm_direction, src, tgt)
                ] = align_dataset
                if align_dataset is None:
                    # no align data so flag the reverse direction as well in sharing
                    langpairs_sharing_datasets[
                        (data_path, split, norm_direction, tgt, src)
                    ] = align_dataset
        else:
            logger.info(
                f"Reusing source and target datasets of [{split}] {tgt}-{src} for reversed direction: "
                f"[{split}] {src}-{tgt}: src length={len(src_dataset)}; tgt length={len(tgt_dataset)}"
            )

        return LanguagePairDataset(
            src_dataset,
            src_dataset.sizes,
            src_dict,
            tgt_dataset,
            tgt_dataset.sizes if tgt_dataset is not None else None,
            tgt_dict,
            left_pad_source=left_pad_source,
            left_pad_target=left_pad_target,
            align_dataset=align_dataset,
            src_lang_id=src_lang_id,
            tgt_lang_id=tgt_lang_id,
        )

    def src_dataset_tranform_func(self, src_lang, tgt_lang, dataset, spec=None):
        if self.args.lang_tok_replacing_bos_eos:
            # it is handled by self.alter_dataset_langtok
            # TODO: Unifiy with alter_dataset_langtok
            return dataset
        if spec is None:
            return dataset
        tok = self.get_encoder_langtok(src_lang, tgt_lang, spec)
        if tok:
            return PrependTokenDataset(dataset, tok)
        return dataset

    def tgt_dataset_tranform_func(self, source_lang, target_lang, dataset, spec=None):
        if dataset is None:
            # note that target dataset can be None during inference time
            return None
        if self.args.lang_tok_replacing_bos_eos:
            # TODO: Unifiy with alter_dataset_langtok
            # It is handled by self.alter_dataset_langtok.
            # The complication in self.alter_dataset_langtok
            # makes a unified framework difficult.
            return dataset
        # if not self.args.decoder_langtok:
        if not spec:
            return dataset
        tok = self.get_decoder_langtok(target_lang, spec)
        if tok:
            return PrependTokenDataset(dataset, tok)
        return dataset

    def alter_dataset_langtok(
        self,
        lang_pair_dataset,
        src_eos=None,
        src_lang=None,
        tgt_eos=None,
        tgt_lang=None,
        src_langtok_spec=None,
        tgt_langtok_spec=None,
    ):
        if src_langtok_spec is None and tgt_langtok_spec is None:
            return lang_pair_dataset

        new_src_eos = None
        if (
            src_langtok_spec is not None
            and src_eos is not None
            and (src_lang is not None or tgt_lang is not None)
        ):
            new_src_eos = self.get_encoder_langtok(src_lang, tgt_lang, src_langtok_spec)
        else:
            src_eos = None

        new_tgt_bos = None
        if tgt_langtok_spec and tgt_eos is not None and tgt_lang is not None:
            new_tgt_bos = self.get_decoder_langtok(tgt_lang, tgt_langtok_spec)
        else:
            tgt_eos = None

        return TransformEosLangPairDataset(
            lang_pair_dataset,
            src_eos=src_eos,
            new_src_eos=new_src_eos,
            tgt_bos=tgt_eos,
            new_tgt_bos=new_tgt_bos,
        )

    def load_a_dataset(
        self,
        split,
        data_path,
        src,
        src_dict,
        tgt,
        tgt_dict,
        combine,
        prepend_bos=False,
        langpairs_sharing_datasets=None,
        data_category=None,
        **extra_kwargs,
    ):
        dataset_impl = self.args.dataset_impl
        upsample_primary = self.args.upsample_primary
        left_pad_source = self.args.left_pad_source
        left_pad_target = self.args.left_pad_target
        max_source_positions = self.args.max_source_positions
        max_target_positions = self.args.max_target_positions
        load_alignments = self.args.load_alignments
        truncate_source = self.args.truncate_source
        src_dataset_transform_func = self.src_dataset_tranform_func
        tgt_dataset_transform_func = self.tgt_dataset_tranform_func
        enable_lang_ids = self.args.enable_lang_ids
        src_lang_dictionary = self.src_lang_dict
        tgt_lang_dictionary = self.tgt_lang_dict
        src_langtok_spec, tgt_langtok_spec = extra_kwargs["langtok_spec"]

        src_langtok = self.get_encoder_langtok(src, tgt, src_langtok_spec)
        tgt_langtok = self.get_decoder_langtok(tgt, tgt_langtok_spec)
        logger.info(
            f"{data_category}:{src}-{tgt} src_langtok: {src_langtok}; tgt_langtok: {tgt_langtok}"
        )

        langpair_ds = self.load_langpair_dataset(
            data_path,
            split,
            src,
            src_dict,
            tgt,
            tgt_dict,
            combine,
            dataset_impl,
            upsample_primary,
            left_pad_source,
            left_pad_target,
            max_source_positions,
            max_target_positions,
            prepend_bos,
            load_alignments,
            truncate_source,
            src_dataset_transform_func=lambda dataset: src_dataset_transform_func(
                src, tgt, dataset, src_langtok_spec
            ),
            tgt_dataset_transform_func=lambda dataset: tgt_dataset_transform_func(
                src, tgt, dataset, tgt_langtok_spec
            ),
            src_lang_id=_lang_id(src_lang_dictionary, src)
            if enable_lang_ids and src_lang_dictionary is not None
            else None,
            tgt_lang_id=_lang_id(tgt_lang_dictionary, tgt)
            if enable_lang_ids and tgt_lang_dictionary is not None
            else None,
            langpairs_sharing_datasets=langpairs_sharing_datasets,
        )

        # TODO: handle modified lang toks for mined data and dae data
        if self.args.lang_tok_replacing_bos_eos:
            ds = self.alter_dataset_langtok(
                langpair_ds,
                src_eos=self.dicts[src if src else tgt].eos(),
                src_lang=src,
                tgt_eos=self.dicts[tgt].eos(),
                tgt_lang=tgt,
                src_langtok_spec=src_langtok_spec,
                tgt_langtok_spec=tgt_langtok_spec,
            )
        else:
            ds = langpair_ds
        return ds

    def load_split_langpair_datasets(self, split, data_param_list):
        datasets = []
        langpairs_sharing_datasets = (
            {} if self.args.enable_reservsed_directions_shared_datasets else None
        )
        for param in data_param_list:
            ds = self.load_a_dataset(
                split=split,
                langpairs_sharing_datasets=langpairs_sharing_datasets,
                **param,
            )
            datasets.append(ds)
        return datasets

    def get_data_paths_and_lang_pairs(self, split):
        datapaths = {"main": self.args.data}
        lang_pairs = {"main": self.lang_pairs}
        if split == getattr(self.args, "train_subset", None):
            # only training data can have extra data and extra language pairs
            if self.args.extra_data:
                extra_datapaths = self.args.extra_data
                datapaths.update(extra_datapaths)
            if self.args.extra_lang_pairs:
                extra_lang_pairs = {
                    k: v.split(",") for k, v in self.args.extra_lang_pairs.items()
                }
                lang_pairs.update(extra_lang_pairs)
        return datapaths, lang_pairs

    @classmethod
    def get_dataset_key(cls, data_category, src, tgt):
        return f"{data_category}:{src}-{tgt}"

    def get_split_num_data_shards(self, split):
        if split in self._num_shards_dict:
            return self._num_shards_dict[split]
        num_shards_dict = {}
        data_paths, lang_pairs = self.get_data_paths_and_lang_pairs(split)

        for data_category, paths in data_paths.items():
            if data_category not in lang_pairs:
                continue
            paths = utils.split_paths(paths)
            lang_dirs = [
                lang_pair.split("-") for lang_pair in lang_pairs[data_category]
            ]
            lang_dirs = [x if len(x) > 1 else (x[0], x[0]) for x in lang_dirs]
            for src, tgt in lang_dirs:
                # monolingual data ruqires tgt only
                assert src is not None or "mono_" in data_category, (
                    f"error: src={src}, " "tgt={tgt} for data_category={data_category}"
                )
                key = self.get_dataset_key(data_category, src, tgt)
                if "mono_" in data_category:
                    num_shards_dict[key] = self.get_mono_split_num_shards(
                        split, tgt, paths, self.args.dataset_impl
                    )
                else:
                    num_shards_dict[key] = self.get_split_num_shards(
                        split, src, tgt, paths, self.args.dataset_impl
                    )
        self._num_shards_dict[split] = num_shards_dict
        logger.info(f"[{split}] num of shards: {num_shards_dict}")
        return num_shards_dict

    def get_split_data_path(self, paths, epoch, shard_epoch, num_shards):
        shard = epoch if shard_epoch is None else shard_epoch
        shard = (shard - 1) % num_shards
        path = paths[shard]
        return path

    def get_split_data_param_list(self, split, epoch, shard_epoch=None):
        # TODO: to extend with extra datasets and keys and loop over different shard data paths
        param_list = []
        data_paths, lang_pairs = self.get_data_paths_and_lang_pairs(split)
        logger.info(f"langtoks settings: {self.args.langtoks}")
        split_num_shards_dict = self.get_split_num_data_shards(split)
        for data_category, paths in data_paths.items():
            if data_category not in lang_pairs:
                continue
            paths = utils.split_paths(paths)
            assert len(paths) > 0
            if len(paths) > 1:
                self._has_sharded_data = True
            if split != getattr(self.args, "train_subset", None):
                # if not training data set, use the first shard for valid and test
                paths = paths[:1]

            if data_category in self.args.langtoks:
                lang_tok_spec = self.args.langtoks[data_category]
            else:
                # default to None
                lang_tok_spec = (None, None)

            # infer langcode
            lang_dirs = [
                lang_pair.split("-") for lang_pair in lang_pairs[data_category]
            ]
            lang_dirs = [x if len(x) > 1 else (x[0], x[0]) for x in lang_dirs]
            for src, tgt in lang_dirs:
                assert src is not None or data_category == "mono_dae", (
                    f"error: src={src}, " "tgt={tgt} for data_category={data_category}"
                )
                # logger.info(f"preparing param for {data_category}: {src} - {tgt}")
                key = self.get_dataset_key(data_category, src, tgt)
                data_path = self.get_split_data_path(
                    paths, epoch, shard_epoch, split_num_shards_dict[key]
                )
                param_list.append(
                    {
                        "key": key,
                        "data_path": data_path,
                        "split": split,
                        "src": src,
                        "src_dict": self.dicts[src]
                        if src and data_category != "mono_dae"
                        else None,
                        "tgt": tgt,
                        "tgt_dict": self.dicts[tgt],
                        "data_category": data_category,
                        "langtok_spec": lang_tok_spec,
                    }
                )
        return param_list

    def get_train_dataset_sizes(self, data_param_list, datasets):
        num_shards = [
            self.get_split_num_data_shards(param["split"])[param["key"]]
            for param in data_param_list
        ]
        data_sizes = [
            (key, len(d) * num_shard)
            for (key, d), num_shard in zip(datasets, num_shards)
        ]
        logger.info(
            f"data sizes multiplied by num_shards used in sampling ratios: {data_sizes}"
        )
        return [s for _, s in data_sizes]

    def get_ordered_train_dataset_sizes(self, data_param_list, datasets):
        num_shards = [
            (self.get_split_num_data_shards(param["split"])[param["key"]], param["src"], param["tgt"])
            for param in data_param_list
        ]
        data_sizes = [
            (src, tgt, len(d) * num_shard)
            for (key, d), (num_shard, src, tgt) in zip(datasets, num_shards)
        ]
        ordered_data_sizes = np.zeros(len(datasets))
        for src, tgt, s in data_sizes:
            ll = src if self.target_group == "source_lang" else tgt
            ordered_data_sizes[_lang_id(self.lang_dict, ll)] = s
        return np.array(ordered_data_sizes)

    def get_train_sampling_ratios(self, data_param_list, datasets, epoch=1):
        data_sizes = self.get_train_dataset_sizes(data_param_list, datasets)
        sampling_func = self.sampling_method.sampling_method_selector()
        sample_ratios = sampling_func(data_sizes) if sampling_func is not None else None
        return sample_ratios

    def get_sampling_ratios(self, data_param_list, datasets, epoch):
        if self.args.sampling_weights_from_file:
            weights = load_sampling_weights(self.args.sampling_weights_from_file)
            sample_ratios = [weights[k] for k, _ in datasets]
            logger.info(
                "| ignoring --sampling-weights when loadding sampling weights "
                f"from file {self.args.sampling_weights_from_file}"
            )
        elif self.args.sampling_weights:
            sample_ratios = [self.args.sampling_weights[k] for k, _ in datasets]
        else:
            sample_ratios = self.get_train_sampling_ratios(
                data_param_list, datasets, epoch
            )

        if sample_ratios is not None:
            logger.info(
                "| Upsample ratios: {}".format(
                    list(zip(map(lambda x: x["key"], data_param_list), sample_ratios))
                )
            )
            assert len(sample_ratios) == len(datasets)
        return sample_ratios

    def load_split_datasets(
        self, split, training, epoch=1, combine=False, shard_epoch=None, **kwargs
    ):
        data_param_list = self.get_split_data_param_list(
            split, epoch, shard_epoch=shard_epoch
        )
        langpairs_sharing_datasets = (
            {} if self.args.enable_reservsed_directions_shared_datasets else None
        )
        datasets = [
            (
                param["key"],
                self.load_a_dataset(
                    combine=combine,
                    langpairs_sharing_datasets=langpairs_sharing_datasets,
                    **param,
                ),
            )
            for param in data_param_list
        ]
        return datasets, data_param_list

    def load_into_concat_dataset(self, split, datasets, data_param_list):
        if self.args.lang_tok_replacing_bos_eos:
            # TODO: to investigate why TransformEosLangPairDataset doesn't work with ConcatDataset
            return SampledMultiDataset(
                OrderedDict(datasets),
                sampling_ratios=None,
                eval_key=None,
                collate_format=CollateFormat.single,
                virtual_size=None,
                split=split,
            )
        return ConcatDataset([d for _, d in datasets])

    def load_sampled_multi_epoch_dataset(
        self, split, training, epoch=0, combine=False, shard_epoch=None, **kwargs
    ):
        datasets, data_param_list = self.load_split_datasets(
            split, training, epoch, combine, shard_epoch=shard_epoch, **kwargs
        )

        if training and split == getattr(self.args, "train_subset", None):
            sample_ratios = self.get_sampling_ratios(data_param_list, datasets, epoch)
            data_sizes = self.get_ordered_train_dataset_sizes(data_param_list, datasets)
            self.data_ratios = data_sizes / sum(data_sizes)
            datasets = OrderedDict(datasets)
            idx = 1 if self.target_group == "target_lang" else 0
            my_lang_ids = [_lang_id(self.lang_dict, key.split(":")[-1].split("-")[idx]) for key in list(datasets.keys())]
            logger.info("Mapped lang ids = {}".format(my_lang_ids))
            return SampledMultiDataset(
                args=self.args,
                datasets=datasets,
                sampling_ratios=sample_ratios,
                seed=2,
                epoch=epoch,
                eval_key=None,
                collate_format=CollateFormat.single,
                split=split,
                shared_collater=self._shared_collater(),
                virtual_size=self.args.virtual_data_size,
                remapped_lang_ids=np.array(my_lang_ids),
            )
            # return SampledMultiEpochDataset(
            #     datasets,
            #     epoch=epoch,
            #     shard_epoch=shard_epoch,
            #     # valid and test datasets will be degenerate to concating datasets:
            #     sampling_ratios=sample_ratios,
            #     eval_key=None,
            #     collate_format=CollateFormat.single,
            #     virtual_size=self.args.virtual_data_size,
            #     split=split,
            #     virtual_epoch_size=self.args.virtual_epoch_size,
            #     # if not using lang_tok altering, simplified to use the same collater
            #     shared_collater=self._shared_collater(),
            #     remapped_lang_ids=np.array(my_lang_ids)
            # )
        else:
            return self.load_into_concat_dataset(split, datasets, data_param_list)

    def load_sampled_multi_epoch_dataset_concat(
        self, split, training, epoch=0, combine=False, shard_epoch=None, **kwargs
    ):
        logger.info("load concat train dataset!")
        datasets, data_param_list = self.load_split_datasets(
            split, training, epoch, combine, shard_epoch=shard_epoch, **kwargs
        )

        data_sizes = self.get_ordered_train_dataset_sizes(data_param_list, datasets)
        self.data_ratios = data_sizes / sum(data_sizes)
        datasets = OrderedDict(datasets)
        idx = 1 if self.target_group == "target_lang" else 0
        my_lang_ids = [_lang_id(self.lang_dict, key.split(":")[-1].split("-")[idx]) for key in list(datasets.keys())]
        logger.info("Mapped lang ids = {}".format(my_lang_ids))
        return SampledMultiDataset(
            args=self.args,
            datasets=datasets,
            seed=2,
            epoch=epoch,
            eval_key=None,
            collate_format=CollateFormat.single,
            virtual_size=self.args.virtual_data_size,
            split=split,
            shared_collater=False,
            shuffle=False,
            remapped_lang_ids=np.array(my_lang_ids),
        )
