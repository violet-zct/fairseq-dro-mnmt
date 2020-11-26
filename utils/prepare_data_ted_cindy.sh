#!/bin/bash

# following Cindy, spm separately and combine

root="/checkpoint/chuntinz/data/mnmt_data/ted/raw"

#langs="ara,aze,bel,ben,bos,bul,ces,cmn,dan,deu,ell,epo,est,eus,fas,fin,fra,glg,heb,hin,hrv,hun,hye,ind,ita,jpn,kat,kaz,kor,kur,lit,mar,mkd,mon,msa,mya,nld,nob,pol,por,ron,rus,slk,slv,spa,sqi,srp,swe,tam,tha,tur,ukr,urd,vie,XXfr_ca,XXpt_pt,XXzh,XXzh_tw"
#opt_root="/checkpoint/chuntinz/data/mnmt_data/ted/ted_all"

langs="aze,bel,glg,slk,tur,rus,por,ces"
opt_root="/checkpoint/chuntinz/data/mnmt_data/ted/ted8_related_sep"

langs="bos,mar,hin,mkd,ell,bul,fra,kor"
opt_root="/checkpoint/chuntinz/data/mnmt_data/ted/ted8_diverse_sep"

if [ ! -d ${opt_root} ]; then
  mkdir ${opt_root}
fi
opt_data=${opt_root}/data
opt_bin=${opt_root}/data-bin

rm -rf ${opt_bin}
rm -rf ${opt_data}
mkdir -p ${opt_data}

SPM_TRAIN=scripts/spm_train.py
SPM_ENCODE=scripts/spm_encode.py
BPE_SIZE=8000
EN_BPE_SIZE=8000

for lang in ${langs//,/ }; do
  python ${SPM_TRAIN} \
    --input=${root}/${lang}_en/train.${lang} \
    --model_prefix=${root}/${lang}_en/spm.${lang} \
    --vocab_size=${BPE_SIZE} \
    --character_coverage=1.0

  mkdir ${opt_data}/${lang}_en
  for f in ${root}/${lang}_en/*${lang}; do
    temp=$(basename $f)
    python ${SPM_ENCODE} \
      --model ${root}/${lang}_en/spm.${lang}.model \
      --inputs ${f} --outputs ${opt_data}/${lang}_en/spm.${temp}
  done

  cat ${root}/${lang}_en/train.en >> ${opt_data}/combine.en
done

python ${SPM_TRAIN} \
  --input=${opt_data}/combine.en \
  --model_prefix=${opt_data}/spm.en.${EN_BPE_SIZE}\
  --vocab_size=${EN_BPE_SIZE} \
  --character_coverage=1.0

for lang in ${langs//,/ }; do
  for f in ${root}/${lang}_en/*en; do
    temp=$(basename $f)
    python ${SPM_ENCODE} \
      --model ${opt_data}/spm.en.${EN_BPE_SIZE}.model \
      --inputs ${f} --outputs ${opt_data}/${lang}_en/spm.${temp}
  done
done

for lang in ${langs//,/ }; do
  cat ${opt_data}/${lang}_en/spm.train.${lang} >> ${opt_data}/combine.train.src
  cat ${opt_data}/${lang}_en/spm.train.en >> ${opt_data}/combine.train.en
done

python preprocess.py \
  --source-lang src --target-lang en \
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
  --srcdict ${opt_bin}/dict.src.txt \
  --tgtdict ${opt_bin}/dict.en.txt \
  --destdir ${opt_bin} --workers 20
done