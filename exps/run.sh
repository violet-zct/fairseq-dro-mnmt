#! /bin/bash

rm -rf /checkpoint/xianl/space/dro_mnt/64*
rm -rf /checkpoint/xianl/space/dro_mnt/65*
rm -rf /checkpoint/xianl/space/dro_mnt/67*

# keeps updating train dynamics; fix bugs; try sample based selection
sbatch exps/69_warmup1_burnout20_td_c0.0_select_baselines_chi_square_wmt.sh
sbatch exps/70_warmup20_burnout20_td_c0.0_select_baselines_chi_square_wmt.sh
sbatch exps/71_sample_warmup1_burnout20_td_c0.0_select_baselines_chi_square_wmt.sh
sbatch exps/72_sample_warmup20_burnout20_td_c0.0_select_baselines_chi_square_wmt.sh