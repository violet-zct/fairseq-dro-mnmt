#! /bin/bash

dirname=$1 # exp name

end=$((SECONDS+259200))
saved_models=/checkpoint/xianl/space/dro_mnt/${dirname}
send_dir=/home/chuntinz/tir5/logs/${dirname}
end_file=$saved_models/END
ssh tir "mkdir ${send_dir}"

scp $saved_models/run.sh tir:${send_dir}/
echo ${send_dir}
while [ $SECONDS -lt $end ]; do
    scp $saved_models/log.txt tir:${send_dir}/
    if [[ -f ${end_file} ]]; then
        break
    fi
    sleep 1800s
done
