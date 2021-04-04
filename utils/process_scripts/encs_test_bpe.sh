#!/bin/bash

# km: clean lang id
# cs: para1M + euro + news
# fr: clean + dedup + subsample 1.8M
# tr: original

langs="fr,tr,cs,ta"
opt_root=/jet/home/chuntinz/work/data/wmt
opt_data=${opt_root}/data
opt_bin=${opt_root}/data-bin

SPM_TRAIN=scripts/spm_train.py
SPM_ENCODE=scripts/spm_encode.py
BPE_SIZE=32000
EN_BPE_SIZE=24000

SCRIPTS=/jet/home/chuntinz/work/data/wmt/mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl  # clean corpus by min/max lengths and ratios;


for lang in ${langs//,/ }; do
  cat ${opt_data}/${lang}_en/spm.train.${lang} >> ${opt_data}/combine.spm.train.xx
  cat ${opt_data}/${lang}_en/spm.train.en >> ${opt_data}/combine.spm.train.en
done

python preprocess.py \
  --source-lang xx --target-lang en \
  --trainpref ${opt_data}/combine.spm.train \
  --thresholdsrc 0 --thresholdtgt 0 \
  --destdir ${opt_bin} --workers 20

for lang in ${langs//,/ }; do
  python preprocess.py \
  --source-lang ${lang} --target-lang en \
  --trainpref ${opt_data}/${lang}_en/spm.train \
  --validpref ${opt_data}/${lang}_en/spm.valid \
  --testpref ${opt_data}/${lang}_en/spm.test \
  --thresholdsrc 0 --thresholdtgt 0 \
  --srcdict ${opt_bin}/dict.xx.txt \
  --tgtdict ${opt_bin}/dict.en.txt \
  --destdir ${opt_bin} --workers 20
done

for lang in ${langs//,/ }; do
  python preprocess.py \
  --source-lang ${lang} --target-lang en \
  --validpref ${opt_data}/${lang}_en/spm.cap.valid \
  --optvalidpref cap.valid \
  --thresholdsrc 0 --thresholdtgt 0 \
  --srcdict ${opt_bin}/dict.xx.txt \
  --tgtdict ${opt_bin}/dict.en.txt \
  --destdir ${opt_bin} --workers 20
done
