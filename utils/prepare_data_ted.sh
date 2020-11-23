#!/bin/bash

# following Cindy, spm separately and combine

langs=""
root="/checkpoint/chuntinz/data/mnmt_data/ted/raw"
opt_root="/checkpoint/chuntinz/data/mnmt_data/ted/ted_all"
opt_root="/checkpoint/chuntinz/data/mnmt_data/ted/ted8_related"
opt_root="/checkpoint/chuntinz/data/mnmt_data/ted/ted8_diverse"

opt_data=${opt_root}/data
opt_bin=${opt_root}/data-bin
mkdir -p ${opt_data}

SPM_TRAIN=scripts/spm_train.py
SPM_ENCODE=scripts/spm_encode.py
BPE_SIZE=8000
EN_BPE_SIZE=30000

for lang in ${langs//,/ }; do
  python ${SPM_TRAIN} \
    --input=${root}/${lang}_en/train.${lang} \
    --model_prefix=${root}/${lang}_en/spm.${lang} \
    --vocab_size=${BPE_SIZE} \
    --character_coverage=1.0

  mkdir ${opt_data}/${lang}_en
  for f in ${root}/${lang}_en/*${lang}; do
    python ${SPM_ENCODE} \
      --model ${root}/${lang}_en/spm.${lang}.model \
      --inputs ${f} --outputs ${opt_data}/${lang}_en/spm.${f}
  done

  cat ${root}/${lang}_en/train.en >> ${opt_data}/combine.en
done

python ${SPM_TRAIN} \
  --input=${root}/${lang}_en/train.en \
  --model_prefix=${opt_data}/spm.en.${EN_BPE_SIZE}\
  --vocab_size=${EN_BPE_SIZE} \
  --character_coverage=1.0

for lang in ${langs//,/ }; do
  for f in ${root}/${lang}_en/*en; do
    python ${SPM_ENCODE} \
      --model ${opt_data}/spm.en.${EN_BPE_SIZE}.model \
      --inputs ${f} --outputs ${opt_data}/${lang}_en/spm.${f}
  done
done

for lang in ${langs//,/ }; do
  cat ${opt_data}/${lang}_en/spm.train.${lang} >> ${opt_data}/combine.train.src
  cat ${opt_data}/${lang}_en/spm.train.en >> ${opt_data}/combine.train.en
done

python fairseq_cli/generate.py \
  --source-lang src --target-lang en \
  --trainpref ${opt_data}/combine.train \
  --thresholdsrc 0 --thresholdtgt 0 \
  --destdir ${opt_bin} --workers 20