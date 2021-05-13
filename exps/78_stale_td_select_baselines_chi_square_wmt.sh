#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=learnfair
##SBATCH --partition=priority
##SBATCH --comment="TACL 4.20"
#SBATCH --job-name=78
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=100g
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=10
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320
#SBATCH --array=0-1
#SBATCH --exclude=learnfair5107,learnfair5199,learnfair5138,learnfair5033,learnfair5037,learnfair5030,learnfair5038,learnfair5078,learnfair5212,learnfair5072,learnfair5119,learnfair5216

source activate mnmt

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

SAVE_ROOT=/checkpoint/xianl/space/dro_mnt
datadir=/private/home/ghazvini/chunting/data/mnmt_data
DATA=${datadir}/wmt4/data-bin-v2
langs="de,fr,ta,tr"
log=1

rhos=(0.2)
direction=$(($SLURM_ARRAY_TASK_ID % 2))  # 0,1,0,1
tempid=$(($SLURM_ARRAY_TASK_ID / 2))  # 0,0,1,1
rho=${rhos[$tempid]}

if [ $direction = 0 ]; then
    lang_pairs="en-de,en-fr,en-ta,en-tr"
    ename="o2m"
    gtgt="xx"
    etok="tgt"
    glevel="target_lang"
    baselines="de:3.104,fr:2.64,ta:3.226,tr:2.195"
elif [ $direction = 1 ]; then
    lang_pairs="de-en,fr-en,ta-en,tr-en"
    ename="m2o"
    gtgt="en"
    etok="src"
    glevel="source_lang"
    baselines="de:2.944,fr:2.774,ta:3.07,tr:2.055"
else
    exit
fi

model=transformer_wmt_en_de
exp_name=78_least_mu_stale_td_select_warmup_20_baselines_ema_0.1_ch_0_rho_${rho}_min_0.2_chi_square_resample_wmt4_de_${ename}

SAVE=${SAVE_ROOT}/${exp_name}
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh
rm ${SAVE}/END
send_dir=/home/chuntinz/tir5/logs/${exp_name}
if [ ${log} = 1 ]; then
  bash v1_exps/send.sh ${exp_name} &
fi

python train.py ${DATA} \
    --warmup-epochs 20 --burnout-epochs -1 --exclude-c 0.05 --max-scale-up 1.0 --compute-train-dynamics 1 \
    --task translation_multi_simple_epoch --max-tokens-valid 30268 --selection-criterion avg \
    --arch ${model} --valid-subset valid --skip-invalid-size-inputs-valid-test \
    --encoder-langtok ${etok} --enable-lang-ids \
    --criterion 'chi_square_resample' --label-smoothing 0.1 --baselines ${baselines}\
    --rho ${rho} --min-prob 0.2 --group-level ${glevel} --ema 0.1 --clear-history 0 \
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

    for cpt in ${SAVE}/checkpoint*; do
        if [[ $cpt == *"last"* ]]; then
          cpt_name=test_${lang}_en_last.log
        elif [[ $cpt == *"best"* ]]; then
          cpt_name=test_${lang}_en.log
        else
          cpt_name=test_${lang}_en_160k.log
        fi

        python fairseq_cli/generate.py ${DATA} \
          --task translation_multi_simple_epoch  \
          --gen-subset test \
          --path ${cpt} \
          --batch-size 150 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece --scoring sacrebleu \
          --lang-pairs ${lang_pairs} --lang-dict ${DATA}/langs.list \
          --encoder-langtok ${etok} \
          --source-lang ${gsrc} --target-lang ${gtgt} \
          --quiet --beam 5 | tee ${SAVE}/${cpt_name}
        scp ${SAVE}/${cpt_name} tir:${send_dir}/
    done
done

rm ${SAVE}/dynamics.tar.gz
tar -cvzf ${SAVE}/dynamics.tar.gz ${SAVE}/*npy
scp ${SAVE}/dynamics.tar.gz tir:${send_dir}/
#rm ${SAVE}/*npy

scp ${SAVE}/log.txt tir:${send_dir}/
scp slurm_logs/slurm-${SLURM_JOB_ID}-$SLURM_ARRAY_TASK_ID.out tir:${send_dir}/
scp slurm_logs/slurm-${SLURM_JOB_ID}-$SLURM_ARRAY_TASK_ID.err tir:${send_dir}/