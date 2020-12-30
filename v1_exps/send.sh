dirname=$1 # exp name

end=$((SECONDS+114400))
saved_models=saved_models/${dirname}
send_dir=/home/chuntinz/tir5/logs/${dirname}
ssh tir "mkdir ${send_dir}"

scp $saved_models/run.sh tir:${send_dir}/
echo ${send_dir}
while [ $SECONDS -lt $end ]; do
    scp $saved_models/log.txt tir:${send_dir}/
    if [[ -f "$saved_models/END" ]]; then
        break
    fi
    sleep 1800s
done
