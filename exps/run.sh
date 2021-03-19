#! /bin/bash

sbatch exps/41_new_opus_erm_baseline.sh
sbatch exps/42_aug_erm_ted.sh
sbatch exps/43_aug_chi_square_resampling_ted.sh
sbatch exps/44_per_group_aug_erm_ted.sh
sbatch exps/45_per_group_aug_chi_square_resampling_ted.sh
sbatch exps/46_0.05_0.2_chi_square_resample_new_opus.sh
sbatch exps/47_aug_erm_opus.sh
sbatch exps/48_aug_0.1_chi_square_resample_opus.sh
sbatch exps/49_per_group_aug_erm_opus.sh
sbatch exps/50_per_group_aug_chi_square_resample_opus.sh