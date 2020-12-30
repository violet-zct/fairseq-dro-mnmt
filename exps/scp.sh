#! /bin/bash

dir="1_analyze_ema0.1_alpha0.5_wu_ub_lang_dro_ted8 2_analyze_hier_ema0.1_alpha0.5_beta0.5_wu_ub_ted8 3_sanity_hier_ema0.1_alpha0.5_beta1.0_wu_ub_ted8"

for prefix in $dir; do
    for pp in related_o2m related_m2o diverse_o2m diverse_m2o; do
        dd=saved_models/${prefix}_${pp}
        if [ ! -d  $dd ]; then
            continue
        fi
        scp $dd/test_* tir:/home/chuntinz/tir5/logs/${prefix}_${pp}/
        if [ -f "${dd}/inner_log.txt" ]; then
            scp $dd/inner_log.txt tir:/home/chuntinz/tir5/logs/${prefix}_${pp}/
        fi
    done
done
