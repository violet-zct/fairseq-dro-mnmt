#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
##SBATCH --partition=learnfair
#SBATCH --partition=priority
#SBATCH --comment="TACL 12.14"
#SBATCH --job-name=18.o2m.related.tgt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=700g
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=30
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320
#SBATCH --array=0

source activate mnmt

#data_names=(ted8_related)
#split=${data_names[$SLURM_ARRAY_TASK_ID]}
SAVE_ROOT=/private/home/chuntinz/work/fairseq-dro-mnmt/saved_models
DATA=/checkpoint/chuntinz/data/mnmt_data/ted/ted8_related/data-bin

langs="aze,bel,glg,slk,tur,rus,por,ces"
lang_pairs="en-aze,en-bel,en-glg,en-slk,en-tur,en-rus,en-por,en-ces"
#lang_pairs="aze-en,bel-en,glg-en,slk-en,tur-en,rus-en,por-en,ces-en"
model=transformer_iwslt_de_en
exp_name=18_bleu_ub_alpha_0.5_step_lr_warmup_50k_tgt_lang_dro_ted8_related_o2m

SAVE=${SAVE_ROOT}/${exp_name}
rm -rf ${SAVE}
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh

python train.py ${DATA} \
    --start-ft-steps 25000 \
	  --task translation_multi_simple_epoch \
	  --arch ${model} --valid-subset cap.valid \
	  --encoder-langtok "tgt" --enable-lang-ids \
	  --criterion 'upper_bound_plain_dro_label_smoothed_cross_entropy' --label-smoothing 0.1 \
	  --dro-alpha 0.5 --update-dro-freq 1 --group-level "target_lang"\
	  --max-update 300000 --layernorm-embedding \
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
    --eval-bleu --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --eval-bleu-remove-bpe sentencepiece \
	  --log-interval 100 --log-format simple | tee ${SAVE}/log.txt

date
wait

for lang in ${langs//,/ }; do
      python fairseq_cli/generate.py ${DATA} \
          --task translation_multi_simple_epoch  \
          --gen-subset test \
          --path ${SAVE}/checkpoint_best.pt \
          --batch-size 300 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece \
	        --scoring sacrebleu \
          --lang-pairs ${lang_pairs} --lang-dict ${DATA}/langs.list \
          --encoder-langtok "tgt" \
          --source-lang en --target-lang ${lang} \
          --beam 5  | tee ${SAVE}/test_${lang}_en.log
done