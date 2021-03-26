#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
##SBATCH --partition=learnfair
#SBATCH --partition=priority
#SBATCH --comment="TACL 3.30"
#SBATCH --job-name=test.m2o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=200g
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=10
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320
#SBATCH --array=0

source activate mnmt2

savedir=/private/home/ghazvini/chunting/fairseq-dro-mnmt
datadir=/private/home/ghazvini/chunting/data/mnmt_data
DATA=${datadir}/opus10/data-bin
langs="yi,mr,oc,be,ta,hi,gl,ur,bg,is"

SAVE_ROOT=${savedir}/saved_models

lang_pairs="yi-en,mr-en,oc-en,be-en,ta-en,hi-en,gl-en,ur-en,bg-en,is-en"
ename="m2o"
gtgt="en"
etok="src"

for exp_name in 46_ch_0_rho_0.05_min_0.2_chi_square_resample_opus10_${ename} 41_medium_erm_temp_1_opus10_${ename} 41_medium_erm_temp_5_opus10_${ename} \
41_medium_erm_temp_100_opus10_${ename} 48_ch_0_aug_0.1_chi_square_resample_opus10_${ename} 49_per_group_aug_0.1_erm_opus10_${ename} \
50_ch_0_per_group_aug_0.1_chi_square_resample_opus10_${ename} 47_aug_0.1_opus10_${ename}; do

SAVE=${SAVE_ROOT}/${exp_name}
send_dir=/home/chuntinz/tir5/logs/${exp_name}

for lang in ${langs//,/ }; do
    if [ $gtgt = "en" ]; then
        gsrc=${lang}
    else
        gsrc="en"
        gtgt=${lang}
    fi

    for cpt in ${SAVE}/checkpoint*; do
        if [[ $cpt == *"last"* ]]; then
          cpt_name=test_${lang}_en_last.log
        elif [[ $cpt == *"best"* ]]; then
          cpt_name=test_${lang}_en.log
        else
          cpt_name=test_${lang}_en_160k.log
        fi

        python fairseq_cli/generate.py ${DATA} \
          --task translation_multi_simple_epoch  \
          --gen-subset test \
          --path ${cpt} \
          --batch-size 150 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece --scoring sacrebleu \
          --lang-pairs ${lang_pairs} --lang-dict ${DATA}/langs.list \
          --encoder-langtok ${etok} \
          --source-lang ${gsrc} --target-lang ${gtgt} \
          --quiet --beam 5 | tee ${SAVE}/${cpt_name}
        scp ${SAVE}/${cpt_name} tir:${send_dir}/
    done
done

done