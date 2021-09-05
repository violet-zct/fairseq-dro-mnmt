#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=exp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=50g
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=5
#SBATCH --time=0
#SBATCH --array=0-3

source activate mnmt

SAVE_ROOT=saved_models
datadir=data/mnmt_data

if [ $SLURM_ARRAY_TASK_ID = 0 ]; then
    langs="aze,bel,glg,slk,tur,rus,por,ces"
    lang_pairs="en-aze,en-bel,en-glg,en-slk,en-tur,en-rus,en-por,en-ces"
    DATA=${datadir}/ted/ted8_related/data-bin
    ename="related_o2m"
    gtgt="xx"
    etok="tgt"
    glevel="target_lang"
    baselines="aze:3.457,bel:3.365,glg:2.805,slk:2.687,tur:2.976,rus:2.832,por:2.368,ces:2.771"
elif [ $SLURM_ARRAY_TASK_ID = 1 ]; then
    langs="aze,bel,glg,slk,tur,rus,por,ces"
    lang_pairs="aze-en,bel-en,glg-en,slk-en,tur-en,rus-en,por-en,ces-en"
    DATA=${datadir}/ted/ted8_related/data-bin
    ename="related_m2o"
    gtgt="en"
    etok="src"
    glevel="source_lang"
    baselines="aze:2.111,bel:1.893,glg:2.006,slk:2.155,tur:2.392,rus:2.472,por:2.037,ces:2.23"
elif [ $SLURM_ARRAY_TASK_ID = 2 ]; then
    langs="bos,mar,hin,mkd,ell,bul,fra,kor"
    lang_pairs="en-bos,en-mar,en-hin,en-mkd,en-ell,en-bul,en-fra,en-kor"
    DATA=${datadir}/ted/ted8_diverse/data-bin
    ename="diverse_o2m"
    gtgt="xx"
    etok="tgt"
    glevel="target_lang"
    baselines="bos:3.111,mar:3.432,hin:3.151,mkd:2.714,ell:2.466,bul:2.429,fra:2.335,kor:3.271"
elif [ $SLURM_ARRAY_TASK_ID = 3 ]; then
    langs="bos,mar,hin,mkd,ell,bul,fra,kor"
    lang_pairs="bos-en,mar-en,hin-en,mkd-en,ell-en,bul-en,fra-en,kor-en"
    DATA=${datadir}/ted/ted8_diverse/data-bin
    ename="diverse_m2o"
    gtgt="en"
    etok="src"
    glevel="source_lang"
    baselines="bos:1.868,mar:2.227,hin:2.188,mkd:2.129,ell:2.148,bul:2.144,fra:2.195,kor:2.664"
else
    exit
fi

rho=0.05 # constraint size
model=transformer_wmt_en_de
exp_name=baseline_t1_${rho}_chi_square_step_ted8_${ename}

SAVE=${SAVE_ROOT}/${exp_name}
mkdir -p ${SAVE}
cp $0 ${SAVE}/run.sh

python train.py ${DATA}\
    --warmup-epochs 1 --baselines ${baselines} \
    --task translation_multi_simple_epoch \
    --arch ${model} --valid-subset cap.valid \
    --encoder-langtok ${etok} --enable-lang-ids \
    --criterion 'chi_square_resample' --label-smoothing 0.1 \
    --group-level ${glevel} --rho ${rho} --ema 0.1 --clamp-q-to-min 1 \
    --max-update 200000 --layernorm-embedding \
    --lang-pairs ${lang_pairs} \
    --lang-dict ${DATA}/langs.list \
    --no-epoch-checkpoints \
    --share-decoder-input-output-embed \
    --dropout 0.3 --attention-dropout 0.3 --activation-dropout 0.3 --weight-decay 1e-4 \
    --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'step' \
    --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4 --lr-decay-rate 0.5 --lr-decay-steps 100000 \
    --max-tokens 16384 \
    --update-freq 1 \
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
    for cpt in best last; do
    	python fairseq_cli/generate.py ${DATA} \
          --task translation_multi_simple_epoch  \
          --gen-subset test --skip-invalid-size-inputs-valid-test \
          --path ${SAVE}/checkpoint_${cpt}.pt \
          --batch-size 300 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece --scoring sacrebleu \
          --lang-pairs ${lang_pairs} --lang-dict ${DATA}/langs.list \
          --encoder-langtok ${etok} \
          --source-lang ${gsrc} --target-lang ${gtgt} \
          --quiet --beam 5 | tee ${SAVE}/test_${cpt}_${lang}_en.log
    done
done