#!/bin/bash

workdir=/jet/home/chuntinz/work/data/wmt/19_encs

lang=cs
python deduplicate.py \
  --src-file ${workdir}/para.cs \
  --tgt-file ${workdir}/para.en \
  --src-file-out ${workdir}/para.cs.dedup \
  --tgt-file-out ${workdir}/para.en.dedup


python subsample_data.py ${workdir} \
  ${workdir}/para.cs.dedup ${workdir}/para.en.dedup 1000000 "1M"


cat euro.cs news.cs para.cs.dedup.1M > train.en-cs.cs
cat euro.en news.en para.en.dedup.1M > train.en-cs.en