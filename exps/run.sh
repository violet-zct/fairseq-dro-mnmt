#! /bin/bash

#rm -rf /checkpoint/xianl/space/dro_mnt/64*
#rm -rf /checkpoint/xianl/space/dro_mnt/65*
#rm -rf /checkpoint/xianl/space/dro_mnt/66*
#rm -rf /checkpoint/xianl/space/dro_mnt/67*


rm -rf /checkpoint/xianl/space/dro_mnt/73*
rm -rf /checkpoint/xianl/space/dro_mnt/74*
rm -rf /checkpoint/xianl/space/dro_mnt/75*

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

scancel 40411227
scancel 40367139
sbatch exps/69_warmup1_burnout20_td_c0.05_select_baselines_chi_square_wmt.sh
sbatch exps/76_baselines_chi_square_resample_wmt4_de_o2m.sh

scp /checkpoint/xianl/space/dro_mnt/68_erm_train_dynamics_wmt14_ende_ende/checkpoint*pt tir:/home/chuntinz/tir5/logs/68_erm_train_dynamics_wmt14_ende_ende/
scp /checkpoint/xianl/space/dro_mnt/68_erm_train_dynamics_wmt14_ende_deen/checkpoint*pt tir:/home/chuntinz/tir5/logs/68_erm_train_dynamics_wmt14_ende_deen/
