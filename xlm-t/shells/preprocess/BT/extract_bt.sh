INPUT_DIR=/mnt/input/SharedTask/large-scale/MonolingualData/all_spm_split_bt/
for file in $(ls $INPUT_DIR/*.log); do
    new_file=${file%.log}
    #if [ ! -f $new_file ]; then
    echo "$file -> $new_file"
    cat $file | grep -P "^H" | cut -f 3- > $new_file
    #fi
done

OUTPUT_DIR=/mnt/input/SharedTask/large-scale/MonolingualData/4M/
mkdir -p $OUTPUT_DIR
LANGS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
for lang in ${LANGS[@]}; do
    #if [ ! -f $OUTPUT_DIR/train.${lang} ]; then
    cat $INPUT_DIR/en000.2${lang} $INPUT_DIR/en001.2${lang} > $OUTPUT_DIR/train.${lang}
    wc -l $OUTPUT_DIR/train.${lang}
    #fi
done