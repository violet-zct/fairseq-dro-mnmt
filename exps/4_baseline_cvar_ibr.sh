#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=30g
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=5
#SBATCH --array=0-1

source activate mnmt

SAVE_ROOT=/jet/home/chuntinz/work/fairseq-dro-mnmt/saved_models
datadir=/jet/home/chuntinz/work/data
DATA=${datadir}/wmt4/data-bin-v2
langs="de,fr,ta,tr"

direction=${SLURM_ARRAY_TASK_ID}

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
exp_name=cvar_ibr_wmt4_${ename}

SAVE=${SAVE_ROOT}/${exp_name}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

python -u train.py ${DATA}\
	  --task translation_multi_simple_epoch --max-scale-up 1.0 \
	  --arch ${model} --valid-subset valid --skip-invalid-size-inputs-valid-test \
    --dro-alpha 0.5 --baselines ${baselines} --resampling 1 \
	  --encoder-langtok ${etok} --group-level ${glevel} \
	  --max-update 300000 --layernorm-embedding \
    --lang-pairs ${lang_pairs} \
    --lang-dict ${DATA}/langs.list \
	  --no-epoch-checkpoints \
	  --share-decoder-input-output-embed \
	  --dropout 0.3 --attention-dropout 0.3 --activation-dropout 0.3 --weight-decay 1e-4 \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 5e-4 --min-lr -1 \
	  --criterion 'plain_dro_label_smoothed_cross_entropy' --label-smoothing 0.1 \
	  --max-tokens 8192 \
	  --seed 222 \
  	--max-source-positions 512 --max-target-positions 512 \
  	--save-dir ${SAVE} \
    --encoder-normalize-before --decoder-normalize-before \
	  --log-interval 100 --log-format simple | tee -a ${SAVE}/log.txt

date

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
      --path ${SAVE}/checkpoint_best.pt \
      --batch-size 300 \
      --lenpen 1.0 \
      --remove-bpe sentencepiece --scoring sacrebleu \
      --lang-pairs ${lang_pairs} --lang-dict ${DATA}/langs.list \
      --encoder-langtok ${etok} \
      --source-lang ${gsrc} --target-lang ${gtgt} \
      --quiet --beam 5 | tee ${SAVE}/test_${lang}_en.log
done