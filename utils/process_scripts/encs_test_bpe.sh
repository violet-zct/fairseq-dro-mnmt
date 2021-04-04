#!/bin/bash

# km: clean lang id
# cs: para1M + euro + news
# fr: clean + dedup + subsample 1.8M
# tr: original

opt_root=/jet/home/chuntinz/work/data/wmt4
opt_data=${opt_root}/data
opt_bin=${opt_root}/data-bin

rm ${opt_bin}/test.cs-en*

for lang in cs; do
  python preprocess.py \
  --source-lang ${lang} --target-lang en \
  --validpref ${opt_data}/${lang}_en/spm.test \
  --optvalidpref test \
  --thresholdsrc 0 --thresholdtgt 0 \
  --srcdict ${opt_bin}/dict.xx.txt \
  --tgtdict ${opt_bin}/dict.en.txt \
  --destdir ${opt_bin} --workers 20
done
