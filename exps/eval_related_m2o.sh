#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
##SBATCH --partition=learnfair
#SBATCH --partition=priority
#SBATCH --comment="TACL 12.14"
#SBATCH --job-name=49.m2o.relate
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
langs="aze"
lang_pairs="aze-en,bel-en,glg-en,slk-en,tur-en,rus-en,por-en,ces-en"
model=transformer_iwslt_de_en
exp_name=49_step_lr_warmup_50k_hier_dro_ted8_related_m2o

SAVE=${SAVE_ROOT}/${exp_name}
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
          --encoder-langtok "src" \
          --target-lang en --source-lang ${lang} \
          --beam 5  | tee ${SAVE}/test_${lang}_en.log
done
