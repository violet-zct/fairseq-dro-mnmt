#! /bin/bash


datapath=/private/home/ghazvini/chunting/data/mnmt_data
root=/private/home/ghazvini/chunting/fairseq-dro-mnmt
tirroot=/home/chuntinz/tir5/logs

#scp  tir:${tirroot}/57_erm_temp_100_wmt4_de_o2m/train_sents_o2m.loss ${datapath}/wmt4/data-bin-v2/

sbatch exps/88_sent_baselined_chi_square_batch_dro_wmt4_de.sh
sbatch exps/89_group_baselined_chi_square_batch_dro_wmt4_de.sh

