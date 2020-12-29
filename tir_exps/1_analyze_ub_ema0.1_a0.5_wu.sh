#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
##SBATCH --partition=learnfair
#SBATCH --partition=priority
#SBATCH --comment="TACL 12.18"
#SBATCH --job-name=28.lang.dro.e0.1.a0.5.wu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=700g
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=30
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320
#SBATCH --array=1,3

source activate mnmt

SLURM_ARRAY_TASK_ID=1
savedir=/home/chuntinz/tir5/fairseq-dro-mnmt
datadir=/home/chuntinz/tir5/data/mnmt_data
log=2

SAVE_ROOT=${savedir}/saved_models

if [ $SLURM_ARRAY_TASK_ID = 0 ]; then
    langs="aze,bel,glg,slk,tur,rus,por,ces"
    lang_pairs="en-aze,en-bel,en-glg,en-slk,en-tur,en-rus,en-por,en-ces"
    DATA=${datadir}/ted/ted8_related/data-bin
    ename="related_o2m"
    gtgt="xx"
    etok="tgt"
    glevel="target_lang"
elif [ $SLURM_ARRAY_TASK_ID = 1 ]; then
    langs="aze,bel,glg,slk,tur,rus,por,ces"
    lang_pairs="aze-en,bel-en,glg-en,slk-en,tur-en,rus-en,por-en,ces-en"
    DATA=${datadir}/ted/ted8_related/data-bin
    ename="related_m2o"
    gtgt="en"
    etok="src"
    glevel="source_lang"
elif [ $SLURM_ARRAY_TASK_ID = 2 ]; then
    langs="bos,mar,hin,mkd,ell,bul,fra,kor"
    lang_pairs="en-bos,en-mar,en-hin,en-mkd,en-ell,en-bul,en-fra,en-kor"
    DATA=${datadir}/ted/ted8_diverse/data-bin
    ename="diverse_o2m"
    gtgt="xx"
    etok="tgt"
    glevel="target_lang"
elif [ $SLURM_ARRAY_TASK_ID = 3 ]; then
    langs="bos,mar,hin,mkd,ell,bul,fra,kor"
    lang_pairs="bos-en,mar-en,hin-en,mkd-en,ell-en,bul-en,fra-en,kor-en"
    DATA=${datadir}/ted/ted8_diverse/data-bin
    ename="diverse_m2o"
    gtgt="en"
    etok="src"
    glevel="source_lang"
else
    exit
fi

model=transformer_iwslt_de_en
exp_name=1_analyze_ema0.1_alpha0.5_wu_ub_lang_dro_ted8_${ename}

SAVE=${SAVE_ROOT}/${exp_name}
rm -rf ${SAVE}
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh

send_dir=/home/chuntinz/tir5/logs/${exp_name}
if [ ${log} = 1 ]; then
  bash v1_exps/send.sh ${exp_name} &
fi

python train.py ${DATA}\
    --start-ft-steps 200 \
    --task translation_multi_simple_epoch \
    --arch ${model} --valid-subset cap.valid \
    --encoder-langtok ${etok} --enable-lang-ids \
    --criterion 'upper_bound_plain_dro_label_smoothed_cross_entropy' --label-smoothing 0.1 \
    --dro-alpha 0.5 --update-dro-freq 1 --group-level ${glevel} --ema 0.1 \
    --max-update 400 --validate-interval-updates 150 --layernorm-embedding \
    --lang-pairs ${lang_pairs} \
    --lang-dict ${DATA}/langs.list \
    --no-epoch-checkpoints \
    --share-decoder-input-output-embed \
    --dropout 0.3 --attention-dropout 0.3 --activation-dropout 0.3 --weight-decay 0.0 \
    --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'step' \
    --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4 --lr-decay-rate 0.5 --lr-decay-steps 50000 \
    --max-tokens 8192 \
    --update-freq 1 \
    --seed 222 \
    --max-source-positions 512 --max-target-positions 512 \
    --save-dir ${SAVE} \
    --encoder-normalize-before --decoder-normalize-before \
    --log-interval 100 --log-format simple | tee ${SAVE}/log.txt

exit
date
wait

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
          --remove-bpe sentencepiece \
	        --scoring sacrebleu \
          --lang-pairs ${lang_pairs} --lang-dict ${DATA}/langs.list \
          --encoder-langtok ${etok} \
          --source-lang ${gsrc} --target-lang ${gtgt} \
          --beam 5  | tee ${SAVE}/test_${lang}_en.log
    if [ ${log} = 1 ]; then
      scp ${SAVE}/test_${lang}_en.log tir:${send_dir}/
    fi
done