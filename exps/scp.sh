#! /bin/bash

#dir="4_baseline_ema0.1_alpha0.5_wu_ub_lang_dro_ted8 5_hier_baseline_ema0.1_alpha0.5_beta0.5_wu_ub_lang_dro_ted8 6_erm_ted8 7_outer_ub_hier_ema0.1_alpha0.5_beta0.5_wu_lang_dro_ted8 8_lr1e-4_ema0.1_alpha0.5_wu_ub_lang_dro_ted8 9_lr4e-4_ema0.1_alpha0.5_wu_ub_lang_dro_ted8"
#
#for prefix in $dir; do
#    for pp in related_o2m related_m2o diverse_o2m diverse_m2o; do
#        dd=saved_models/${prefix}_${pp}
#        if [ ! -d  $dd ]; then
#            continue
#        fi
#        ssh tir "mkdir /home/chuntinz/tir5/logs/${prefix}_${pp}/"
#        scp $dd/run.sh tir:/home/chuntinz/tir5/logs/${prefix}_${pp}/
#        scp $dd/*log.txt tir:/home/chuntinz/tir5/logs/${prefix}_${pp}/
#        scp $dd/test_*log tir:/home/chuntinz/tir5/logs/${prefix}_${pp}/
##        if [ -f "${dd}/inner_log.txt" ]; then
##            scp $dd/inner_log.txt tir:/home/chuntinz/tir5/logs/${prefix}_${pp}/
##        fi
#    done
#done

for exp_name in 55_erm_temp_1_wmt4_o2m 55_erm_temp_5_wmt4_o2m 55_erm_temp_100_wmt4_o2m 55_erm_temp_1_wmt4_m2o 55_erm_temp_5_wmt4_m2o 55_erm_temp_100_wmt4_m2o; do
  scp saved_models/${exp_name}/checkpoint_best.pt /home/chuntinz/tir5/logs/${exp_name}/
done

for exp_name in 57_erm_temp_1_wmt4_de_m2o 57_erm_temp_5_wmt4_de_m2o 57_erm_temp_100_wmt4_de_m2o 57_erm_temp_1_wmt4_de_o2m 57_erm_temp_5_wmt4_de_o2m 57_erm_temp_100_wmt4_de_o2m; do
  scp saved_models/${exp_name}/checkpoint_best.pt /home/chuntinz/tir5/logs/${exp_name}/
done