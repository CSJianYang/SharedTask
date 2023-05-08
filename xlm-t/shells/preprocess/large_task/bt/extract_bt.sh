OUTPUT_DIR=/mnt/input/SharedTask/large-scale/MonolingualData/all_spm_split_bt/
for file in $(ls $OUTPUT_DIR/*.log); do
    new_file=${file%.log}
    echo ${new_file}
    exit
    cat $file | grep -P "^H" | cut -f 3- > $new_file
done