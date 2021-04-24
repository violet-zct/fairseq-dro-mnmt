#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=learnfair
##SBATCH --partition=priority
##SBATCH --comment="TACL 4.20"
#SBATCH --job-name=77
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=100g
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=10
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320
#SBATCH --array=0-3

source activate mnmt

SAVE_ROOT=/checkpoint/xianl/space/dro_mnt
DATA=/checkpoint/xianl/space/dro_mnt/wmt14_train_dynamics_bin
langs="de"
log=1
lang_pairs="en-de"
gtgt="xx"
glevel="target_lang"

direction=$SLURM_ARRAY_TASK_ID

if [ $direction = 0 ]; then
  subdir="avg_probs_var_0.5"
elif [ $direction = 1 ]; then
  subdir="min_probs_var_0.5"
elif [ $direction = 2 ]; then
  subdir="med_probs_var_0.5"
elif [ $direction = 3 ]; then
  subdir="random_0.5"
else
    exit
fi

DATA=${DATA}/${subdir}
model=transformer_wmt_en_de
exp_name=77_erm_train_dynamics_wmt14_ende_${subdir}

SAVE=${SAVE_ROOT}/${exp_name}
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh
rm ${SAVE}/END
send_dir=/home/chuntinz/tir5/logs/${exp_name}
if [ ${log} = 1 ]; then
  bash v1_exps/send.sh ${exp_name} &
fi

python -u train.py ${DATA}\
	  --task translation_multi_simple_epoch \
	  --arch ${model} --valid-subset valid --skip-invalid-size-inputs-valid-test \
	  --group-level ${glevel} --max-tokens-valid 28268 \
	  --max-update 200000 --layernorm-embedding \
    --lang-pairs ${lang_pairs} \
	  --no-epoch-checkpoints \
	  --share-decoder-input-output-embed \
	  --dropout 0.3 --attention-dropout 0.3 --activation-dropout 0.3 --weight-decay 1e-4 \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 5e-4 --min-lr -1 \
	  --criterion 'train_dynamics_label_smoothed_cross_entropy' --label-smoothing 0.1 \
	  --compute-train-dynamics 0 --analyze 0 \
	  --max-tokens 8192 \
	  --seed 222 \
  	--max-source-positions 512 --max-target-positions 512 \
  	--save-dir ${SAVE} \
    --encoder-normalize-before --decoder-normalize-before \
	  --log-interval 100 --log-format simple | tee -a ${SAVE}/log.txt

date
echo "end" | tee ${SAVE}/END

for lang in ${langs//,/ }; do
    if [ $gtgt = "en" ]; then
        gsrc=${lang}
    else
        gsrc="en"
        gtgt=${lang}
    fi
    for cpt in best last; do
      python fairseq_cli/generate.py ${DATA} \
            --task translation_multi_simple_epoch  \
            --gen-subset test --skip-invalid-size-inputs-valid-test \
            --path ${SAVE}/checkpoint_${cpt}.pt \
            --batch-size 300 \
            --remove-bpe "@@ " --max-len-a 2 --max-len-b 0 \
            --lang-pairs ${lang_pairs} \
            --source-lang ${gsrc} --target-lang ${gtgt} \
            --quiet --beam 5 | tee ${SAVE}/test_${cpt}_${lang}_en.log
    done
done

scp ${SAVE}/log.txt tir:${send_dir}/

scp slurm_logs/slurm-${SLURM_JOB_ID}-$SLURM_ARRAY_TASK_ID.out tir:${send_dir}/
scp slurm_logs/slurm-${SLURM_JOB_ID}-$SLURM_ARRAY_TASK_ID.err tir:${send_dir}/