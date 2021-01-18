#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=priority
#SBATCH --job-name=mt-22_opus
#SBATCH --comment="ACL 2021 deadline Jan 26th."
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=100g
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=10
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320
#SBATCH --array=0-4

source activate mnmt2

split=$SLURM_ARRAY_TASK_ID
seeds=(1 524287 65537 101 8191)
seed=${seeds[$split]}

# The ENV below are only used in distributed training with env:// initialization
#export MASTER_ADDR=${SLURM_NODELIST:0:9}${SLURM_NODELIST:10:3}
#export MASTER_PORT=15213

SAVE_ROOT=/private/home/ghazvini/chunting/fairseq-dro-mnmt/saved_models
DATA=/private/home/ghazvini/chunting/data/marjan_data/mnmt_data/wmt14_ende
model=transformer_wmt_en_de
exp_name=22_opus_run${split}

SAVE=${SAVE_ROOT}/${exp_name}
rm -rf ${SAVE}
mkdir -p ${SAVE}
src=en
tgt=de
cp $0 ${SAVE}/run.sh

send_dir=/home/chuntinz/tir5/logs/${exp_name}
bash v1_exps/send.sh ${exp_name} &

python -u train.py ${DATA} \
    --seed ${seed} \
    --valid-subset valid \
    --eval-bleu --eval-bleu-remove-bpe '@@ ' --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-tokenized-bleu --eval-bleu-detok "space" \
    -a ${model} --optimizer adabelief --lr 0.0005 --clip-norm 1.0 --clip-mode 'total' -s $src -t $tgt \
    --label-smoothing 0.1 --max-tokens 8192 --share-all-embeddings \
    --attention-dropout 0.1 --activation-dropout 0.1 \
    --lr-scheduler milestone --lr-decay-rate 0.1 --milestones 250000 450000 \
    --weight-decay 0.0001 --weight-decay-type 'decoupled' \
    --criterion label_smoothed_cross_entropy --max-update 500000 \
    --warmup-updates 1000 --warmup-init-lr 1e-07 --dropout 0.1 \
    --adabelief-betas '(0.9, 0.999)' --adabelief-eps 1e-16 --save-dir ${SAVE} \
    --keep-last-epochs 10 --keep-interval-updates 1 --update-freq 1 --save-interval-updates 5000 \
    --log-format simple --log-interval 100 --num-workers 0 | tee ${SAVE}/log.txt

date
echo "end" | tee ${SAVE}/END

opt=${SAVE}/test_best.log
python -u fairseq_cli/generate.py $DATA --gen-subset test -s $src -t $tgt --path ${SAVE}/checkpoint_best.pt --batch-size 300 --remove-bpe "@@ "  --beam 5 --max-len-a 2 --max-len-b 0 --quiet | tee ${opt}
scp ${opt} tir:${send_dir}/

opt=${SAVE}/test_last.log
python -u fairseq_cli/generate.py $DATA --gen-subset test -s $src -t $tgt --path ${SAVE}/checkpoint_last.pt --batch-size 300 --remove-bpe "@@ "  --beam 5 --max-len-a 2 --max-len-b 0 --quiet | tee ${opt}
scp ${opt} tir:${send_dir}/

python scripts/average_checkpoints.py --inputs ${SAVE} --output ${SAVE}/checkpoint_last10.pt --num-epoch-checkpoints 10
rm -f ${SAVE}/checkpoint2*.pt
rm -f ${SAVE}/checkpoint_254_500000.pt

opt=${SAVE}/test_last10.log
python -u fairseq_cli/generate.py $DATA --gen-subset test -s $src -t $tgt --path ${SAVE}/checkpoint_last10.pt --batch-size 300 --remove-bpe "@@ " --beam 5 --max-len-a 2 --max-len-b 0 --quiet | tee ${opt}
scp ${opt} tir:${send_dir}/

scp ${SAVE}/log.txt tir:${send_dir}/
scp slurm_logs/slurm-${SLURM_JOB_ID}-$SLURM_ARRAY_TASK_ID.out tir:${send_dir}/
scp slurm_logs/slurm-${SLURM_JOB_ID}-$SLURM_ARRAY_TASK_ID.err tir:${send_dir}/
