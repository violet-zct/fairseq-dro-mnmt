dirname=$1 # exp name

#end=$((SECONDS+259200))
end=$((SECONDS+7200))
saved_models="saved_models/${dirname}"
send_dir=/home/chuntinz/tir5/logs/${dirname}
ssh tir "mkdir ${send_dir}"


scp $saved_models/run.sh tir:${send_dir}/
echo ${send_dir}
while [ $SECONDS -lt $end ]; do
    scp $saved_models/log.txt tir:${send_dir}/
#    scp ${saved_models}/*log tir:${send_tir}/
#    ls ${saved_models}
    if [[ -f $saved_models/END ]]; then
        echo "transfer"
        scp $saved_models/*log tir:${send_dir}/
        wait
        scp $saved_models/*log.txt tir:${send_dir}/
        wait
        wait
        break
    fi
    sleep 100s
done
