#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
##SBATCH --partition=learnfair
#SBATCH --partition=priority
#SBATCH --comment="TACL 12.2"
#SBATCH --job-name=mt.ted8.diverse.m2o
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

SAVE_ROOT=/private/home/chuntinz/work/fairseq-dro-mnmt/saved_models
DATA=/checkpoint/chuntinz/data/mnmt_data/ted/ted8_diverse/data-bin

langs="bos,mar,hin,mkd,ell,bul,fra,kor"
lang_pairs="bos-en,mar-en,hin-en,mkd-en,ell-en,bul-en,fra-en,kor-en"
model=transformer_iwslt_de_en
exp_name=5_baseline_temp_5_ted8_diverse_m2o

SAVE=${SAVE_ROOT}/${exp_name}
rm -rf ${SAVE}
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh

python train.py ${DATA}\
	  --task translation_multi_simple_epoch \
	  --arch ${model} --valid-subset cap.valid \
	  --sampling-method "temperature" --sampling-temperature 5 \
	  --encoder-langtok "src" \
	  --max-update 150000 --layernorm-embedding \
    --lang-pairs ${lang_pairs} \
    --lang-dict ${DATA}/langs.list \
	  --no-epoch-checkpoints \
	  --share-decoder-input-output-embed \
	  --dropout 0.3 --attention-dropout 0.1 --activation-dropout 0.1 --weight-decay 0.0 \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 3e-5 --min-lr -1 \
	  --criterion 'label_smoothed_cross_entropy' --label-smoothing 0.1 \
	  --max-tokens 8192 \
	  --update-freq 1 \
	  --seed 222 \
  	--max-source-positions 512 --max-target-positions 512 \
  	--save-dir ${SAVE} \
    --encoder-normalize-before --decoder-normalize-before \
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
      --lang-pairs ${lang_pairs} \
      --encoder-langtok "src" \
      --source-lang ${lang} --target-lang en \
      --beam 5  | tee ${SAVE}/test_${lang}_en.log
done
