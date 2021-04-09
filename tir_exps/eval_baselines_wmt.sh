#!/bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH --time=0
#SBATCH --array=0-3

source activate mnmt
models=(57_erm_temp_1_wmt4_de_m2o 57_erm_temp_100_wmt4_de_m2o 57_erm_temp_1_wmt4_de_o2m 57_erm_temp_100_wmt4_de_o2m)
model=${models[${SLURM_ARRAY_TASK_ID}]}

SAVE=/home/chuntinz/tir5/logs/${model}
datadir=/home/chuntinz/tir5/data/mnmt_data/
DATA=${datadir}/wmt4/data-bin-v2
langs="de,fr,ta,tr"

if [[ "${model}" == *m2o ]]; then
    lang_pairs="de-en,fr-en,ta-en,tr-en"
    ename="m2o"
    gtgt="en"
    etok="src"
    glevel="source_lang"
elif [[ "${model}" == *o2m ]]; then
    lang_pairs="en-de,en-fr,en-ta,en-tr"
    ename="o2m"
    gtgt="xx"
    etok="tgt"
    glevel="target_lang"
else
  exit
fi

python -u valid.py ${DATA}\
	  --task translation_multi_simple_epoch \
	  --path ${SAVE}/checkpoint_last.pt \
	  --valid-subset train \
	  --encoder-langtok ${etok} \
    --lang-pairs ${lang_pairs} \
    --lang-dict ${DATA}/langs.list \
	  --max-tokens 4096 --log-interval 100 | tee ${SAVE}/valid_log.txt
