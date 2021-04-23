#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=learnfair
##SBATCH --partition=priority
##SBATCH --comment="TACL 4.20"
#SBATCH --job-name=73
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=300g
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=10
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320
#SBATCH --array=0-1

source activate mnmt

SAVE_ROOT=/checkpoint/xianl/space/dro_mnt
datadir=/private/home/ghazvini/chunting/data/mnmt_data
DATA=${datadir}/wmt4/data-bin-v2
langs="de,fr,ta,tr"
log=1

direction=$SLURM_ARRAY_TASK_ID
temp=5

if [ $direction = 0 ]; then
    lang_pairs="en-de,en-fr,en-ta,en-tr"
    ename="o2m"
    gtgt="xx"
    etok="tgt"
    glevel="target_lang"
elif [ $direction = 1 ]; then
    lang_pairs="de-en,fr-en,ta-en,tr-en"
    ename="m2o"
    gtgt="en"
    etok="src"
    glevel="source_lang"
else
    exit
fi

model=transformer_wmt_en_de
exp_name=73_cl_sum_log_probs_erm_temp_${temp}_wmt4_de_${ename}

SAVE=${SAVE_ROOT}/${exp_name}
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh
rm ${SAVE}/END
send_dir=/home/chuntinz/tir5/logs/${exp_name}
if [ ${log} = 1 ]; then
  bash v1_exps/send.sh ${exp_name} &
fi
echo $SLURM_ARRAY_JOB_ID $SLURM_ARRAY_TASK_ID $SLURM_JOB_ID > ${SAVE}/log.txt

python -u train.py ${DATA}\
	  --task translation_multi_simple_epoch --selection-method sample --max-scale-up 1.0 \
	  --compute-train-dynamics 1 --warmup-epochs 5 --competent-cl 1 --hardness 'sum_log_prob' \
	  --arch ${model} --valid-subset valid --skip-invalid-size-inputs-valid-test \
	  --sampling-method "temperature" --sampling-temperature ${temp} \
	  --encoder-langtok ${etok} --group-level ${glevel} \
	  --max-update 300000 --layernorm-embedding \
    --lang-pairs ${lang_pairs} \
    --lang-dict ${DATA}/langs.list \
	  --no-epoch-checkpoints \
	  --share-decoder-input-output-embed \
	  --dropout 0.3 --attention-dropout 0.3 --activation-dropout 0.3 --weight-decay 1e-4 \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 5e-4 --min-lr -1 \
	  --criterion 'logged_label_smoothed_cross_entropy' --label-smoothing 0.1 \
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
          --batch-size 300 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece --scoring sacrebleu \
          --lang-pairs ${lang_pairs} --lang-dict ${DATA}/langs.list \
          --encoder-langtok ${etok} \
          --source-lang ${gsrc} --target-lang ${gtgt} \
          --quiet --beam 5 | tee ${SAVE}/${cpt_name}
        scp ${SAVE}/${cpt_name} tir:${send_dir}/
    done
done

scp ${SAVE}/log.txt tir:${send_dir}/
scp slurm_logs/slurm-${SLURM_JOB_ID}-$SLURM_ARRAY_TASK_ID.out tir:${send_dir}/
scp slurm_logs/slurm-${SLURM_JOB_ID}-$SLURM_ARRAY_TASK_ID.err tir:${send_dir}/