#!/bin/bash

SCRIPTS=/jet/home/chuntinz/work/data/wmt/mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl  # clean corpus by min/max lengths and ratios; used after bpe

workdir=/jet/home/chuntinz/work/data/wmt/14_enfr

for lang in en fr; do
  bash clean.sh ${workdir}/train.tags.en-fr.${lang} ${lang}
done

lang=fr
python deduplicate.py \
  --src-file ${workdir}/train.tags.en-fr.${lang}.clean \
  --tgt-file ${workdir}/train.tags.en-fr.en.clean \
  --src-file-out ${workdir}/train.en-fr.${lang}.dedup \
  --tgt-file-out ${workdir}/train.en-fr.en.dedup


python subsample_data.py ${workdir} \
  ${workdir}/train.en-fr.${lang}.dedup ${workdir}/train.en-fr.en.dedup 2500000 "2.5M"

mv ${workdir}/train.en-fr.en.dedup.2.5M ${workdir}/temp.en
mv ${workdir}/train.en-fr.fr.dedup.2.5M ${workdir}/temp.fr

perl ${CLEAN} -ratio 1.5 ${workdir}/temp ${lang} en ${workdir}/train.en-fr 1 250