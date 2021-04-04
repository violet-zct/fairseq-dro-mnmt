#!/bin/bash

SCRIPTS=/jet/home/chuntinz/work/data/wmt4/mosesdecoder/scripts
DETOK=$SCRIPTS/tokenizer/detokenizer.perl  # clean corpus by min/max lengths and ratios; used after bpe

workdir=/jet/home/chuntinz/work/data/wmt4/14_ende

lang=de

for split in test valid train; do
  perl $DETOK -l en < $workdir/${split}.en-de.en > ${workdir}/${split}.en-de.en.detok
  mv ${workdir}/${split}.en-de.en.detok $workdir/${split}.en-de.en
done

for split in test valid train; do
  perl $DETOK -l de < $workdir/${split}.en-de.de > ${workdir}/${split}.en-de.de.detok
  mv ${workdir}/${split}.en-de.de.detok $workdir/${split}.en-de.de
done

mv $workdir/train.en-de.en $workdir/train.en
mv $workdir/train.en-de.de $workdir/train.de

python subsample_data.py $workdir train.de train.en 3000000 subset
mv $workdir/train.en.subset $workdir/train.en-de.en
mv $workdir/train.de.subset $workdir/train.en-de.de