#!/bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10g
#SBATCH --time=0
##SBATCH --array=0-1
#SBATCH --array=0-1

source activate mnmt
models=(erm_ted8_t1_diverse_m2o erm_ted8_t1_diverse_o2m)
models=(erm_ted8_t1_related_m2o erm_ted8_t1_related_o2m)
model=${models[${SLURM_ARRAY_TASK_ID}]}

SAVE=/home/chuntinz/tir5/logs/${model}
datadir=/home/chuntinz/tir5/data/mnmt_data/

DATA=${datadir}/ted/ted8_related/data-bin
langs="aze,bel,glg,slk,tur,rus,por,ces"

#DATA=${datadir}/ted/ted8_diverse/data-bin
#langs="bos,mar,hin,mkd,ell,bul,fra,kor"

if [[ "${model}" == *m2o ]]; then
    lang_pairs="aze-en,bel-en,glg-en,slk-en,tur-en,rus-en,por-en,ces-en"
    #lang_pairs="bos-en,mar-en,hin-en,mkd-en,ell-en,bul-en,fra-en,kor-en"
    etok="src"
    glevel="source_lang"
elif [[ "${model}" == *o2m ]]; then
    lang_pairs="en-aze,en-bel,en-glg,en-slk,en-tur,en-rus,en-por,en-ces"
    #lang_pairs="en-bos,en-mar,en-hin,en-mkd,en-ell,en-bul,en-fra,en-kor"
    etok="tgt"
    glevel="target_lang"
else
  exit
fi

arch=transformer_iwslt_de_en

python -u fairseq_cli/compute_baseline_loss.py ${DATA}\
    --task translation_multi_simple_epoch --restore-file ${SAVE}/checkpoint_best.pt \
    --arch ${arch} --valid-subset train --skip-invalid-size-inputs-valid-test \
    --encoder-langtok ${etok} --group-level ${glevel} \
    --max-update 100 --layernorm-embedding \
    --lang-pairs ${lang_pairs} \
    --lang-dict ${DATA}/langs.list \
    --no-epoch-checkpoints \
    --share-decoder-input-output-embed \
    --dropout 0.3 --attention-dropout 0.3 --activation-dropout 0.3 --weight-decay 0.0 \
    --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
    --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4 --min-lr -1 \
    --criterion 'logged_label_smoothed_cross_entropy' --label-smoothing 0.1 \
    --max-tokens 11192 \
    --seed 222 \
    --max-source-positions 512 --max-target-positions 512 \
    --save-dir ${SAVE} \
    --encoder-normalize-before --decoder-normalize-before \
    --log-interval 100 --log-format simple | tee ${SAVE}/valid_log.txt
