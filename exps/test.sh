#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
##SBATCH --partition=learnfair
#SBATCH --partition=priority
#SBATCH --comment="TACL 1.10"
#SBATCH --job-name=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=700g
##SBATCH -C volta32gb
#SBATCH --cpus-per-task=30
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320
#SBATCH --array=0-3

source activate mnmt

savedir=/private/home/ghazvini/chunting/fairseq-dro-mnmt
datadir=/private/home/ghazvini/chunting/data/marjan_data/mnmt_data

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

for exp in 1_analyze_ema0.1_alpha0.5_wu_ub_lang_dro_ted8 2_analyze_hier_ema0.1_alpha0.5_beta0.5_wu_ub_ted8 3_sanity_hier_ema0.1_alpha0.5_beta1.0_wu_ub_ted8; do
exp_name=${exp}_${ename}
send_dir=/home/chuntinz/tir5/logs/${exp_name}
SAVE=${SAVE_ROOT}/${exp_name}

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
          --beam 5 | tee -a ${SAVE}/test_${lang}_en.log
    scp ${SAVE}/test_${lang}_en.log tir:${send_dir}/
done
done
