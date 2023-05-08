TEXT=/mnt/input/SharedTask/thunder/small_task2/download/small_task2_filt_spm/
TRAIN=/mnt/input/SharedTask/thunder/small_task2/download/train/
LANG_PAIRS=(en-id en-jv en-ms en-ta en-tl id-jv id-ms id-ta id-tl jv-ms jv-ta jv-tl ms-ta ms-tl ta-tl)
mkdir -p $TRAIN
for LANG_PAIR in ${LANG_PAIRS[@]}; do
    src=${LANG_PAIR:0:2}
    tgt=${LANG_PAIR:3:2}
    echo "${src}->${tgt}"
    count=$(ls $TEXT/*.${src}-${tgt}.${src} 2>/dev/null | wc -l)
    if [ $count -gt 0 ]; then
        for file in $(ls $TEXT/*.${src}-${tgt}.${src}); do
            file=$(basename $file)
            echo "$TEXT/$file -> $TRAIN/train.${src}-${tgt}.${src}"            
            cat $TEXT/$file >> $TRAIN/train.${src}-${tgt}.${src}
        done
        for file in $(ls $TEXT/*.${src}-${tgt}.${tgt}); do
            file=$(basename $file)
            echo "$TEXT/$file -> $TRAIN/train.${src}-${tgt}.${tgt}"        
            cat $TEXT/$file >> $TRAIN/train.${src}-${tgt}.${tgt}
        done
    fi
done
