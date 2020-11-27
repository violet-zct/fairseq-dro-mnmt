#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
##SBATCH --partition=learnfair
#SBATCH --partition=priority
#SBATCH --comment="TACL 12.2"
#SBATCH --job-name=mt.ted8.diverse.o2m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=700g
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=30
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320
#SBATCH --array=0-2

source activate mnmt

data_names=(ted8_diverse_30k ted8_diverse ted8_diverse_sep)
split=${data_names[$SLURM_ARRAY_TASK_ID]}
SAVE_ROOT=/private/home/chuntinz/work/fairseq-dro-mnmt/saved_models
DATA=/checkpoint/chuntinz/data/mnmt_data/ted/${split}/data-bin

langs="bos,mar,hin,mkd,ell,bul,fra,kor"
lang_pairs="en-bos,en-mar,en-hin,en-mkd,en-ell,en-bul,en-fra,en-kor"
model=transformer_iwslt_de_en
exp_name=6_baseline_temp_5_${split}_o2m

SAVE=${SAVE_ROOT}/${exp_name}
rm -rf ${SAVE}
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh

python train.py ${DATA}\
	  --task translation_multi_simple_epoch \
	  --arch ${model} --valid-subset cap.valid \
	  --sampling-method "temperature" --sampling-temperature 5 \
	  --encoder-langtok "tgt" \
	  --max-update 150000 --layernorm-embedding \
    --lang-pairs ${lang_pairs} \
    --lang-dict ${DATA}/langs.list \
	  --no-epoch-checkpoints \
	  --share-decoder-input-output-embed \
	  --dropout 0.3 --attention-dropout 0.3 --activation-dropout 0.3 --weight-decay 0.0 \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4 --min-lr -1 \
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
          --lang-pairs ${lang_pairs} --lang-dict ${DATA}/langs.list \
          --encoder-langtok "tgt" \
          --source-lang en --target-lang ${lang} \
          --beam 5  | tee ${SAVE}/test_${lang}_en.log
done
