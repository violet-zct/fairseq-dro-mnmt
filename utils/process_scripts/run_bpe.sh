#!/bin/bash

# km: clean lang id
# cs: para1M + euro + news
# fr: clean + dedup + subsample 1.8M
# tr: original

langs="fr,tr,de,ta"
opt_root=/jet/home/chuntinz/work/data/wmt4
opt_data=${opt_root}/data_de_v2
opt_bin=${opt_root}/data-bin-v2

rm -rf ${opt_bin}
rm -rf ${opt_data}
mkdir -p ${opt_data}

SPM_TRAIN=scripts/spm_train.py
SPM_ENCODE=scripts/spm_encode.py
BPE_SIZE=34000
EN_BPE_SIZE=24000

SCRIPTS=/jet/home/chuntinz/work/data/wmt4/mosesdecoder/scripts
CLEAN=$SCRIPTS/training/clean-corpus-n.perl  # clean corpus by min/max lengths and ratios;

python utils/process_scripts/wmt_upsample_cat_data_and_cap_valid.py

python ${SPM_TRAIN} \
  --input=${opt_data}/raw.combine.en \
  --model_prefix=${opt_data}/spm.en.${EN_BPE_SIZE}\
  --vocab_size=${EN_BPE_SIZE} \
  --character_coverage=0.9995

python ${SPM_TRAIN} \
  --input=${opt_data}/raw.combine.xx \
  --model_prefix=${opt_data}/spm.xx.${BPE_SIZE}\
  --vocab_size=${BPE_SIZE} \
  --character_coverage=0.9995

for lang in ${langs//,/ }; do
  mkdir -p ${opt_data}/${lang}_en

  if [ ${lang} = "fr" ]; then
    langdir="14_enfr"
  elif [ ${lang} = "de" ]; then
    langdir="14_ende"
  elif [ ${lang} = "tr" ]; then
    langdir="18_entr"
  elif [ ${lang} = "ta" ]; then
    langdir="20_enta"
  else
    echo "wrong lang id!"
    exit
  fi
  for lid in en ${lang}; do
    if [ ${lid} = "en" ]; then
      model="en"
      size=${EN_BPE_SIZE}
    else
      model="xx"
      size=${BPE_SIZE}
    fi
    python ${SPM_ENCODE} \
      --model ${opt_data}/spm.${model}.${size}.model \
      --inputs ${opt_root}/${langdir}/train.en-${lang}.${lid} \
      --outputs ${opt_root}/${langdir}/spm.train.en-${lang}.${lid}

    for split in test valid; do
      python ${SPM_ENCODE} \
      --model ${opt_data}/spm.${model}.${size}.model \
      --inputs ${opt_root}/${langdir}/${split}.en-${lang}.${lid} \
      --outputs ${opt_data}/${lang}_en/spm.${split}.${lid}
    done
    python ${SPM_ENCODE} \
      --model ${opt_data}/spm.${model}.${size}.model \
      --inputs ${opt_root}/${langdir}/cap.valid.${lid} \
      --outputs ${opt_data}/${lang}_en/spm.cap.valid.${lid}
    done

  perl ${CLEAN} -ratio 1.5 ${opt_root}/${langdir}/spm.train.en-${lang} ${lid} en ${opt_data}/${lang}_en/spm.train 5 250

  if [ ${lang} = "fr" ]; then
    python utils/process_scripts/subsample_data.py ${opt_data}/${lang}_en spm.train.en spm.train.fr 2000000 subset
    mv ${opt_data}/${lang}_en/spm.train.en.subset ${opt_data}/${lang}_en/spm.train.en
    mv ${opt_data}/${lang}_en/spm.train.fr.subset ${opt_data}/${lang}_en/spm.train.fr
  elif [ ${lang} = "de" ]; then
    python utils/process_scripts/subsample_data.py ${opt_data}/${lang}_en spm.train.en spm.train.de 2500000 subset
    mv ${opt_data}/${lang}_en/spm.train.en.subset ${opt_data}/${lang}_en/spm.train.en
    mv ${opt_data}/${lang}_en/spm.train.de.subset ${opt_data}/${lang}_en/spm.train.de
  fi

done

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
