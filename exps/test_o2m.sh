#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
##SBATCH --partition=learnfair
#SBATCH --partition=priority
#SBATCH --comment="TACL 3.20"
#SBATCH --job-name=test.o2m
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
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
langs="yi,mr,oc,be,ta,ka,gl,ur,bg,is"

SAVE_ROOT=${savedir}/saved_models

lang_pairs="en-yi,en-mr,en-oc,en-be,en-ta,en-ka,en-gl,en-ur,en-bg,en-is"
ename="o2m"
gtgt="xx"
etok="tgt"
glevel="target_lang"
obfile="enxx_outer_baselines"
ibfile="enxx_inner_baselines"

model=transformer_wmt_en_de

for exp_name in 32_rho_0.1_min_0.2_chi_square_resample_opus10_${ename} 33_rho_0.05_min_0.2_chi_square_resample_opus10_${ename} 34_rho_0.05_min_0.5_chi_square_resample_opus10_${ename}; do

SAVE=${SAVE_ROOT}/${exp_name}
send_dir=/home/chuntinz/tir5/logs/${exp_name}

for lang in ${langs//,/ }; do
    if [ $gtgt = "en" ]; then
        gsrc=${lang}
    else
        gsrc="en"
        gtgt=${lang}
    fi
    python fairseq_cli/generate.py ${DATA} \
          --task translation_multi_simple_epoch  \
          --gen-subset test --skip-invalid-size-inputs-valid-test \
          --path ${SAVE}/checkpoint_last.pt \
          --batch-size 300 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece --scoring sacrebleu \
          --lang-pairs ${lang_pairs} --lang-dict ${DATA}/langs.list \
          --encoder-langtok ${etok} \
          --source-lang ${gsrc} --target-lang ${gtgt} \
          --quiet --beam 5 | tee ${SAVE}/test_${lang}_en_last.log
    scp ${SAVE}/test_${lang}_en.log tir:${send_dir}/

done

done