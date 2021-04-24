#!/bin/bash

root="/home/chuntinz/tir5/data/opus_wmt14/wmt14_train_dynamics_bpe"
dict_bin="/home/chuntinz/tir5/data/mnmt_data/wmt14_ende"
opt_root="/home/chuntinz/tir5/data/opus_wmt14/wmt14_train_dynamics_bin"

for dirname in min_probs_var_0.5 avg_probs_var_0.5 med_probs_var_0.5; do
  opt_data=$root/${dirname}
  opt_bin=${opt_root}/${dirname}
  python preprocess.py \
  --source-lang en --target-lang de \
  --trainpref ${opt_data}/train \
  --validpref ${opt_data}/valid \
  --testpref ${opt_data}/test \
  --thresholdsrc 0 --thresholdtgt 0 \
  --srcdict ${dict_bin}/dict.en.txt \
  --tgtdict ${dict_bin}/dict.de.txt \
  --destdir ${opt_bin} --workers 20
done

