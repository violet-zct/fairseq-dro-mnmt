#!/bin/bash

workdir=/jet/home/chuntinz/work/data/wmt/19_encs

lang=cs
python deduplicate.py \
  --src-file ${workdir}/para.cs \
  --tgt-file ${workdir}/para.en \
  --src-file-out ${workdir}/para.cs.dedup \
  --tgt-file-out ${workdir}/para.en.dedup


python subsample_data.py ${workdir} \
  ${workdir}/para.cs.dedup ${workdir}/para.en.dedup 3500000 "3.5M"

mv ${workdir}/para.en.dedup.3.5M ${workdir}/temp.en
mv ${workdir}/para.cs.dedup.3.5M ${workdir}/temp.fr

perl ${CLEAN} -ratio 1.5 ${workdir}/temp ${lang} en ${workdir}/para.subset 1 250

wc -l ${workdir}/para.subset*

cat ${workdir}/euro.cs ${workdir}/news.cs ${workdir}/para.subset.cs > ${workdir}/train.en-cs.cs
cat ${workdir}/euro.en ${workdir}/news.en ${workdir}/para.subset.en > ${workdir}/train.en-cs.en