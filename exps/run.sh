#! /bin/bash

#rm -rf /checkpoint/xianl/space/dro_mnt/64*
#rm -rf /checkpoint/xianl/space/dro_mnt/65*
#rm -rf /checkpoint/xianl/space/dro_mnt/66*
#rm -rf /checkpoint/xianl/space/dro_mnt/67*


#rm -rf /checkpoint/xianl/space/dro_mnt/73*
#rm -rf /checkpoint/xianl/space/dro_mnt/74*
#rm -rf /checkpoint/xianl/space/dro_mnt/75*

#sbatch exps/73_cl_wmt_de_erm_temp5_baseline.sh
#sbatch exps/74_cl_hardness_sum_warmup5_baselines_chi_square_wmt.sh
#sbatch exps/75_cl_hardness_min_prob_warmup5_baselines_chi_square_wmt.sh

## keeps updating train dynamics; fix bugs; try sample based selection
#sbatch exps/69_warmup1_burnout20_td_c0.05_select_baselines_chi_square_wmt.sh
#sbatch exps/70_warmup20_burnout20_td_c0.05_select_baselines_chi_square_wmt.sh
#sbatch exps/71_sample_warmup1_burnout20_td_c0.05_select_baselines_chi_square_wmt.sh
#sbatch exps/72_sample_warmup20_burnout20_td_c0.05_select_baselines_chi_square_wmt.sh

## use CL with competence to select data for DRO; run baseline DRO, ERM with CL, and DRO with CL
#sbatch exps/62_baselines_chi_square_resample_wmt4_de_o2m.sh
#sbatch exps/73_cl_wmt_de_erm_temp5_baseline.sh
#sbatch exps/74_cl_hardness_sum_warmup5_baselines_chi_square_wmt.sh
#sbatch exps/75_cl_hardness_min_prob_warmup5_baselines_chi_square_wmt.sh

#scancel 40411227
#scancel 40367139
#sbatch exps/69_warmup1_burnout20_td_c0.05_select_baselines_chi_square_wmt.sh
#sbatch exps/76_baselines_chi_square_resample_wmt4_de_o2m.sh
#
#scp /checkpoint/xianl/space/dro_mnt/68_erm_train_dynamics_wmt14_ende_ende/checkpoint*pt tir:/home/chuntinz/tir5/logs/68_erm_train_dynamics_wmt14_ende_ende/
#scp /checkpoint/xianl/space/dro_mnt/68_erm_train_dynamics_wmt14_ende_deen/checkpoint*pt tir:/home/chuntinz/tir5/logs/68_erm_train_dynamics_wmt14_ende_deen/

#scp -r tir:/home/chuntinz/tir5/data/opus_wmt14/wmt14_train_dynamics_bin /checkpoint/xianl/space/dro_mnt/
#sbatch exps/76_baselines_chi_square_resample_wmt4_de_o2m.sh
#sbatch exps/77_subset_wmt14_ende_train_dynamics.sh
#sbatch exps/78_stale_td_select_baselines_chi_square_wmt.sh

mkdir -p /checkpoint/xianl/space/dro_mnt/data
rm -rf /checkpoint/xianl/space/dro_mnt/wmt14_train_dynamics_bin

datapath=/checkpoint/xianl/space/dro_mnt/data
root=/checkpoint/xianl/space/dro_mnt
tirroot=/home/chuntinz/tir5/logs

# copy data
scp -r tir:/home/chuntinz/tir5/data/mnmt_data/wmt4/data-bin-14de ${datapath}/
scp -r tir:/home/chuntinz/tir5/data/mnmt_data/wmt4/enfr_bin ${datapath}/
#for exp in avg_probs_var_0.5 min_probs_var_0.5 med_probs_var_0.5 random_0.5; do
#  scp ${root}/77_erm_train_dynamics_wmt14_ende_${exp}/test*log tir:${tirroot}/77_erm_train_dynamics_wmt14_ende_${exp}/
#done
#sbatch exps/76_baselines_chi_square_resample_wmt4_de_o2m.sh

# TED ERM to obtain models for DRO
sbatch exps/6_analyze_erm.sh
# resume training from previous runs; a run that was disrupted previously
sbatch exps/78_stale_td_select_baselines_chi_square_wmt.sh
# bilingual experiments for step A
sbatch exps/68_wmt14_ende_train_dynamics.sh
sbatch exps/79_enfr_train_dynamics.sh
# baseline experiments on the new created dataset: multilingual experiments for step B
sbatch exps/80_new_wmt_erm_baseline.sh
# multilingual experiments with multilingual TD selection for step B
sbatch exps/81_burnout20_td_select_wmt_new_oversample.sh



