#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=learnfair
##SBATCH --partition=priority
##SBATCH --comment="TACL 3.20"
#SBATCH --job-name=47
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=200g
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=10
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320
#SBATCH --array=0-1

source activate mnmt2

trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID
     # SLURM_JOB_ID is a unique representation of the job, equivalent
     # to above
     scontrol requeue $SLURM_JOB_ID
   fi
}


# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

savedir=/private/home/ghazvini/chunting/fairseq-dro-mnmt
datadir=/private/home/ghazvini/chunting/data/mnmt_data
DATA=${datadir}/opus10/data-bin
langs="yi,mr,oc,be,ta,ka,gl,hi,bg,is"
log=1

SAVE_ROOT=${savedir}/saved_models

if [ $SLURM_ARRAY_TASK_ID = 0 ]; then
    lang_pairs="en-yi,en-mr,en-oc,en-be,en-ta,en-ka,en-gl,en-hi,en-bg,en-is"
    ename="o2m"
    gtgt="xx"
    etok="tgt"
    glevel="target_lang"
    obfile="enxx_outer_baselines"
    ibfile="enxx_inner_baselines"
    aug="in_group"
elif [ $SLURM_ARRAY_TASK_ID = 1 ]; then
    lang_pairs="yi-en,mr-en,oc-en,be-en,ta-en,ka-en,gl-en,hi-en,bg-en,is-en"
    ename="m2o"
    gtgt="en"
    etok="src"
    glevel="source_lang"
    obfile="xxen_outer_baselines"
    ibfile="xxen_inner_baselines"
    aug="global"
else
    exit
fi

model=transformer_wmt_en_de
exp_name=47_aug_0.1_chi_square_resample_opus10_${ename}

SAVE=${SAVE_ROOT}/${exp_name}
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh
rm ${SAVE}/END
send_dir=/home/chuntinz/tir5/logs/${exp_name}
if [ ${log} = 1 ]; then
  bash v1_exps/send.sh ${exp_name} &
fi

python train.py ${DATA} \
    --warmup-epochs 1 \
    --task translation_multi_simple_epoch --ddp-backend=no_c10d \
    --arch ${model} --valid-subset valid --skip-invalid-size-inputs-valid-test \
    --aug-option ${aug} --mix-beta-type "fixed" --beta-dist-alpha 0.1 \
    --encoder-langtok ${etok} --enable-lang-ids \
    --criterion 'chi_square_resample' --label-smoothing 0.1 \
    --rho 0.05 --min-prob 0.2 --group-level ${glevel} --ema 0.1 \
    --max-update 300000 --layernorm-embedding \
    --lang-pairs ${lang_pairs} \
    --lang-dict ${DATA}/langs.list \
    --no-epoch-checkpoints \
    --share-decoder-input-output-embed \
    --dropout 0.3 --attention-dropout 0.3 --activation-dropout 0.3 --weight-decay 1e-4 \
    --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'step' \
    --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 5e-4 --lr-decay-rate 0.5 --lr-decay-steps 100000 \
    --max-tokens 8192 \
    --update-freq 1 \
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
    python fairseq_cli/generate.py ${DATA} \
          --task translation_multi_simple_epoch  \
          --gen-subset test --skip-invalid-size-inputs-valid-test \
          --path ${SAVE}/checkpoint_best.pt \
          --batch-size 300 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece --scoring sacrebleu \
          --lang-pairs ${lang_pairs} --lang-dict ${DATA}/langs.list \
          --encoder-langtok ${etok} \
          --source-lang ${gsrc} --target-lang ${gtgt} \
          --quiet --beam 5 | tee ${SAVE}/test_${lang}_en.log
    scp ${SAVE}/test_${lang}_en.log tir:${send_dir}/

done

for lang in ${langs//,/ }; do
    if [ $gtgt = "en" ]; then
        gsrc=${lang}
    else
        gsrc="en"
        gtgt=${lang}
    fi
    python fairseq_cli/generate.py ${DATA} \
          --task translation_multi_simple_epoch  \
          --gen-subset test \
          --path ${SAVE}/checkpoint_last.pt \
          --batch-size 300 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece --scoring sacrebleu \
          --lang-pairs ${lang_pairs} --lang-dict ${DATA}/langs.list \
          --encoder-langtok ${etok} \
          --source-lang ${gsrc} --target-lang ${gtgt} \
          --quiet --beam 5 | tee -a ${SAVE}/log.txt ${SAVE}/test_${lang}_en_last.log
    scp ${SAVE}/test_${lang}_en_last.log tir:${send_dir}/
done

scp ${SAVE}/log.txt tir:${send_dir}/
scp slurm_logs/slurm-${SLURM_JOB_ID}-$SLURM_ARRAY_TASK_ID.out tir:${send_dir}/
scp slurm_logs/slurm-${SLURM_JOB_ID}-$SLURM_ARRAY_TASK_ID.err tir:${send_dir}/