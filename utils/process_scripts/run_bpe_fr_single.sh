#!/bin/bash

opt_root=/jet/home/chuntinz/work/data/wmt4
data_path=${opt_root}/14_enfr
opt_dir=${opt_root}/enfr_data
opt_bin=${opt_root}/enfr_bin

SPM_TRAIN=scripts/spm_train.py
SPM_ENCODE=scripts/spm_encode.py
BPE_SIZE=30000

cat ${opt_dir}/train.en-fr.en ${opt_dir}/train.en-fr.fr > ${opt_dir}/combine
python ${SPM_TRAIN} \
  --input=${opt_dir}/combine \
  --model_prefix=${opt_dir}/spm \
  --vocab_size=${BPE_SIZE} \
  --character_coverage=0.9995

for lang in en fr; do
  python ${SPM_ENCODE} \
      --model ${opt_dir}/spm.model \
      --inputs ${opt_dir}/train.en-fr.${lang} \
      --outputs ${opt_dir}/spm.train.${lang}

  for split in test valid; do
      python ${SPM_ENCODE} \
      --model ${opt_dir}/spm.model \
      --inputs ${data_path}/${split}.en-fr.${lang} \
      --outputs ${opt_dir}/spm.${split}.${lang}
  done
done

lang=fr
python preprocess.py \
  --source-lang ${lang} --target-lang en \
  --trainpref ${opt_dir}/spm.train \
  --validpref ${opt_dir}/spm.valid \
  --testpref ${opt_dir}/spm.test \
  --thresholdsrc 3 --thresholdtgt 3 \
  --joined-dictionary \
  --destdir ${opt_bin} --workers 20
