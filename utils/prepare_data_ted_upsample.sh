#!/bin/bash

# following Cindy, spm separately and combine

root="/checkpoint/chuntinz/data/mnmt_data/ted/raw"
option=$1

if [ $option = "1" ]; then
  langs="ara,aze,bel,ben,bos,bul,ces,cmn,dan,deu,ell,epo,est,eus,fas,fin,fra,glg,heb,hin,hrv,hun,hye,ind,ita,jpn,kat,kaz,kor,kur,lit,mar,mkd,mon,msa,mya,nld,nob,pol,por,ron,rus,slk,slv,spa,sqi,srp,swe,tam,tha,tur,ukr,urd,vie,XXfr_ca,XXpt_pt,XXzh,XXzh_tw"
  target="ted_all"
  BPE_SIZE=50000
  lang_file="/checkpoint/chuntinz/data/mnmt_data/ted/lang_lists/all.langs.list"
elif [ $option = "2" ]; then
  langs="aze,bel,glg,slk,tur,rus,por,ces"
  target="ted8_related"
  BPE_SIZE=30000
  lang_file="/checkpoint/chuntinz/data/mnmt_data/ted/lang_lists/8re.langs.list"
else
  langs="bos,mar,hin,mkd,ell,bul,fra,kor"
  target="ted8_diverse"
  BPE_SIZE=30000
  lang_file="/checkpoint/chuntinz/data/mnmt_data/ted/lang_lists/8di.langs.list"
fi

opt_root="/checkpoint/chuntinz/data/mnmt_data/ted/${target}"
opt_data=${opt_root}/data
opt_bin=${opt_root}/data-bin

rm -rf ${opt_bin}
rm -rf ${opt_data}
mkdir -p ${opt_data}

SPM_TRAIN=scripts/spm_train.py
SPM_ENCODE=scripts/spm_encode.py
EN_BPE_SIZE=10000

python utils/upsample_cat_data_and_cap_valid.py ${target}

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

cp ${lang_file} ${opt_bin}/langs.list