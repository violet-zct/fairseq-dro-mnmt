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
#SBATCH --exclude=compute-0-31,compute-1-7

source activate hal
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
arch=transformer_wmt_en_de

#python -u valid.py ${DATA}\
#	  --task translation_multi_simple_epoch \
#	  --path ${SAVE}/checkpoint_last.pt \
#	  --valid-subset train \
#	  --encoder-langtok ${etok} \
#    --lang-pairs ${lang_pairs} \
#    --lang-dict ${DATA}/langs.list \
#	  --max-tokens 4096 --log-interval 100 --log-format simple | tee ${SAVE}/valid_log.txt

python -u fairseq_cli/compute_baseline_loss.py ${DATA}\
	  --task translation_multi_simple_epoch --restore-file checkpoint_best.pt \
	  --arch ${arch} --valid-subset train --skip-invalid-size-inputs-valid-test \
	  --sampling-method "temperature" --sampling-temperature 1 \
	  --encoder-langtok ${etok} --group-level ${glevel} \
	  --max-update 100 --layernorm-embedding \
    --lang-pairs ${lang_pairs} \
    --lang-dict ${DATA}/langs.list \
	  --no-epoch-checkpoints \
	  --share-decoder-input-output-embed \
	  --dropout 0.3 --attention-dropout 0.3 --activation-dropout 0.3 --weight-decay 1e-4 \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 5e-4 --min-lr -1 \
	  --criterion 'logged_label_smoothed_cross_entropy' --label-smoothing 0.1 \
	  --max-tokens 11192 \
	  --seed 222 \
  	--max-source-positions 512 --max-target-positions 512 \
  	--save-dir ${SAVE} \
    --encoder-normalize-before --decoder-normalize-before \
	  --log-interval 100 --log-format simple | tee ${SAVE}/valid_log.txt