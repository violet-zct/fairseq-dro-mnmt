#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
##SBATCH --partition=learnfair
#SBATCH --partition=priority
#SBATCH --comment="TACL 12.2"
#SBATCH --job-name=mt.all.o2m
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

SAVE_ROOT=/private/home/chuntinz/work/fairseq-gdro/saved_models
DATA=/checkpoint/chuntinz/data/mnmt_data/ted/ted_all/data-bin

langs="ara,aze,bel,ben,bos,bul,ces,cmn,dan,deu,ell,epo,est,eus,fas,fin,fra,glg,heb,hin,hrv,hun,hye,ind,ita,jpn,kat,kaz,kor,kur,lit,mar,mkd,mon,msa,mya,nld,nob,pol,por,ron,rus,slk,slv,spa,sqi,srp,swe,tam,tha,tur,ukr,urd,vie,XXfr_ca,XXpt_pt,XXzh,XXzh_tw"
lang_pairs="en-ara,en-aze,en-bel,en-ben,en-bos,en-bul,en-ces,en-cmn,en-dan,en-deu,en-ell,en-epo,en-est,en-eus,en-fas,en-fin,en-fra,en-glg,en-heb,en-hin,en-hrv,en-hun,en-hye,en-ind,en-ita,en-jpn,en-kat,en-kaz,en-kor,en-kur,en-lit,en-mar,en-mkd,en-mon,en-msa,en-mya,en-nld,en-nob,en-pol,en-por,en-ron,en-rus,en-slk,en-slv,en-spa,en-sqi,en-srp,en-swe,en-tam,en-tha,en-tur,en-ukr,en-urd,en-vie,en-XXfr_ca,en-XXpt_pt,en-XXzh,en-XXzh_tw"
model=transformer
exp_name=2_baseline_temp_5_ted_all_o2m

SAVE=${SAVE_ROOT}/${exp_name}
rm -rf ${SAVE}
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh

python train.py ${DATA}\
	  --task translation_multi_simple_epoch \
	  --arch ${model} \
	  --sampling-method "temperature" --sampling-temperature 5 \
	  --encoder-langtok "tgt" \
	  --max-update 300000 --layernorm-embedding \
    --lang-pairs ${lang_pairs} \
    --lang-dict ${DATA}/langs.list \
	  --no-epoch-checkpoints \
	  --distributed-world-size 1 \
	  --share-decoder-input-output-embed --share-decoders --share-encoders \
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
  if [ $lang = "XXzh_tw" ] || [ $lang = "XXzh" ] || [ $lang = "jpn" ]; then
    python fairseq_cli/generate.py ${DATA} \
            --task translation_multi_simple_epoch  \
            --gen-subset test \
            --path ${SAVE}/checkpoint_best.pt \
            --batch-size 300 \
            --lenpen 1.0 \
            --lang-pairs ${lang_pairs} \
            --source-lang en --target-lang ${lang} \
            --encoder-langtok "tgt" \
            --beam 5  | tee ${SAVE}/test_${lang}_en.log
  else
        python fairseq_cli/generate.py ${DATA} \
            --task translation_multi_simple_epoch  \
            --gen-subset test \
            --path ${SAVE}/checkpoint_best.pt \
            --batch-size 300 \
            --lenpen 1.0 \
            --remove-bpe sentencepiece \
	          --sacrebleu \
            --lang-pairs ${lang_pairs} \
            --encoder-langtok "tgt" \
            --source-lang en --target-lang ${lang} \
            --beam 5  | tee ${SAVE}/test_${lang}_en.log
  fi
done