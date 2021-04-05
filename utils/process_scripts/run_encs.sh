#!/bin/bash

SCRIPTS=/jet/home/chuntinz/work/data/wmt/mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl  # clean corpus by min/max lengths and ratios; used after bpe

workdir=/jet/home/chuntinz/work/data/wmt4/19_encs

lang=cs
python deduplicate.py \
  --src-file ${workdir}/para.cs \
  --tgt-file ${workdir}/para.en \
  --src-file-out ${workdir}/para.cs.dedup \
  --tgt-file-out ${workdir}/para.en.dedup


python subsample_data.py ${workdir} \
  ${workdir}/para.cs.dedup ${workdir}/para.en.dedup 4000000 "3.5M"

mv ${workdir}/para.en.dedup.3.5M ${workdir}/temp.en
mv ${workdir}/para.cs.dedup.3.5M ${workdir}/temp.cs

perl ${CLEAN} -ratio 1.5 ${workdir}/temp ${lang} en ${workdir}/para.subset 2 250

wc -l ${workdir}/para.subset*

cat ${workdir}/euro.cs ${workdir}/news.cs ${workdir}/para.subset.cs > ${workdir}/train.en-cs.cs
cat ${workdir}/euro.en ${workdir}/news.en ${workdir}/para.subset.en > ${workdir}/train.en-cs.en