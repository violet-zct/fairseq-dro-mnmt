#!/bin/bash

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
  ${workdir}/train.en-fr.${lang}.dedup ${workdir}/train.en-fr.en.dedup 4000000 "4M"