#!/bin/bash

# following Cindy, spm separately and combine

root="/home/chuntinz/tir5/data/mnmt_data/opus10/raw"
langs="yi,mr,oc,be,ta,ka,gl,ur,bg,is"

opt_root="/home/chuntinz/tir5/data/mnmt_data/opus10"
opt_data=${opt_root}/data
opt_bin=${opt_root}/data-bin

rm -rf ${opt_bin}
rm -rf ${opt_data}
mkdir -p ${opt_data}

SPM_TRAIN=scripts/spm_train.py
SPM_ENCODE=scripts/spm_encode.py
BPE_SIZE=32000
EN_BPE_SIZE=16000

python utils/opus_upsample_cat_and_valid_cap.py

python ${SPM_TRAIN} \
  --input=${opt_data}/raw.combine.en \
  --model_prefix=${opt_data}/spm.en.${EN_BPE_SIZE}\
  --vocab_size=${EN_BPE_SIZE} \
  --character_coverage=1.0

python ${SPM_TRAIN} \
  --input=${opt_data}/raw.combine.xx \
  --model_prefix=${opt_data}/spm.xx.${BPE_SIZE}\
  --vocab_size=${BPE_SIZE} \
  --character_coverage=1.0

for lang in ${langs//,/ }; do
  for f in ${root}/${lang}_en/*en; do
    temp=$(basename $f)
    python ${SPM_ENCODE} \
      --model ${opt_data}/spm.en.${EN_BPE_SIZE}.model \
      --inputs ${f} --outputs ${opt_data}/${lang}_en/spm.${temp}
    python ${SPM_ENCODE} \
      --model ${opt_data}/spm.en.${EN_BPE_SIZE}.model \
      --inputs ${opt_data}/${lang}_en/cap.raw.valid.en \
      --outputs ${opt_data}/${lang}_en/spm.cap.valid.en
  done

  for f in ${root}/${lang}_en/*${lang}; do
    temp=$(basename $f)
    python ${SPM_ENCODE} \
      --model ${opt_data}/spm.xx.${BPE_SIZE}.model \
      --inputs ${f} --outputs ${opt_data}/${lang}_en/spm.${temp}

    python ${SPM_ENCODE} \
      --model ${opt_data}/spm.xx.${BPE_SIZE}.model \
      --inputs ${opt_data}/${lang}_en/cap.raw.valid.${lang} \
      --outputs ${opt_data}/${lang}_en/spm.cap.valid.${lang}
  done
done

for lang in ${langs//,/ }; do
  cat ${opt_data}/${lang}_en/spm.train.${lang} >> ${opt_data}/combine.train.xx
  cat ${opt_data}/${lang}_en/spm.train.en >> ${opt_data}/combine.train.en
done

python preprocess.py \
  --source-lang xx --target-lang en \
  --trainpref ${opt_data}/combine.train \
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

