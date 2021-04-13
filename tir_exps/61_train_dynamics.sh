#! /bin/bash
#SBATCH --output=slurm_logs/slurm-%A-%a.out
#SBATCH --error=slurm_logs/slurm-%A-%a.err
#SBATCH --partition=learnfair
##SBATCH --partition=priority
##SBATCH --comment="TACL 4.20"
#SBATCH --job-name=61
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --mem=300g
#SBATCH -C volta32gb
#SBATCH --cpus-per-task=10
##SBATCH --signal=B:USR1@60 #Signal is sent to batch script itself
##SBATCH --open-mode=append
#SBATCH --time=4320
#SBATCH --array=0-1

source activate mnmt

savedir=/home/chuntinz/tir5/fairseq-dro-mnmt
datadir=/home/chuntinz/tir5/data/mnmt_data

SAVE_ROOT=${savedir}/saved_models

langs="bos,mar,hin,mkd,ell,bul"
lang_pairs="en-bos,en-mar,en-hin,en-mkd,en-ell,en-bul"
DATA=${datadir}/ted/ted8_diverse/data-bin
ename="diverse_o2m"
gtgt="xx"
etok="tgt"
glevel="target_lang"

model=transformer_iwslt_de_en
exp_name=61_debug_${ename}

SAVE=${SAVE_ROOT}/${exp_name}
mkdir -p ${SAVE}

cp $0 ${SAVE}/run.sh
send_dir=/home/chuntinz/tir5/logs/${exp_name}

python -u train.py ${DATA}\
	  --task translation_multi_simple_epoch \
	  --arch ${model} --valid-subset valid --skip-invalid-size-inputs-valid-test \
	  --encoder-langtok ${etok} --group-level ${glevel} --max-tokens-valid 12868 \
	  --max-update 300000 --layernorm-embedding \
    --lang-pairs ${lang_pairs} \
    --lang-dict ${DATA}/langs.list \
	  --no-epoch-checkpoints \
	  --share-decoder-input-output-embed \
	  --dropout 0.3 --attention-dropout 0.3 --activation-dropout 0.3 --weight-decay 1e-4 \
	  --optimizer 'adam' --adam-betas '(0.9, 0.98)' --lr-scheduler 'inverse_sqrt' \
	  --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 5e-4 --min-lr -1 \
	  --criterion 'train_dynamics_label_smoothed_cross_entropy' --label-smoothing 0.1 --compute-train-dynamics 1 \
	  --max-tokens 5892 \
	  --seed 222 \
  	--max-source-positions 512 --max-target-positions 512 \
  	--save-dir ${SAVE} \
    --encoder-normalize-before --decoder-normalize-before \
	  --log-interval 100 --log-format simple | tee -a ${SAVE}/log.txt

tar -cvzf ${SAVE}/dynamics.tar.gz ${SAVE}/*npy ${SAVE}/*opt
exit

date
echo "end" | tee ${SAVE}/END

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
          --path ${SAVE}/checkpoint_best.pt \
          --batch-size 150 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece --scoring sacrebleu \
          --lang-pairs ${lang_pairs} --lang-dict ${DATA}/langs.list \
          --encoder-langtok ${etok} \
          --source-lang ${gsrc} --target-lang ${gtgt} \
          --quiet --beam 5 | tee ${SAVE}/test_${lang}_en.log
    scp ${SAVE}/test_${lang}_en.log tir:${send_dir}/
done

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
          --batch-size 150 \
          --lenpen 1.0 \
          --remove-bpe sentencepiece --scoring sacrebleu \
          --lang-pairs ${lang_pairs} --lang-dict ${DATA}/langs.list \
          --encoder-langtok ${etok} \
          --source-lang ${gsrc} --target-lang ${gtgt} \
          --quiet --beam 5 | tee ${SAVE}/test_${lang}_en_last.log
    scp ${SAVE}/test_${lang}_en_last.log tir:${send_dir}/
done

scp ${SAVE}/log.txt tir:${send_dir}/

tar -cvzf ${SAVE}/dynamics.tar.gz ${SAVE}/*npy ${SAVE}/*opt
scp ${SAVE}/dynamics.tar.gz tir:${send_dir}/

rm ${SAVE}/*npy
rm ${SAVE}/*opt
scp slurm_logs/slurm-${SLURM_JOB_ID}-$SLURM_ARRAY_TASK_ID.out tir:${send_dir}/
scp slurm_logs/slurm-${SLURM_JOB_ID}-$SLURM_ARRAY_TASK_ID.err tir:${send_dir}/