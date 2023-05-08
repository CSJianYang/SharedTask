echo "small-task1-train"
TRAIN=/mnt/input/SharedTask/thunder/small_task2/Filter_v1/bt_iter2_train/
DATA_BIN=/mnt/input/SharedTask/thunder/small_task2/Filter_v1/bt_data-bin/
DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
LANG_PAIRS=(en-id en-ta en-tl en-jv en-ms)
for LANG_PAIR in ${LANG_PAIRS[@]}; do
    src=${LANG_PAIR:0:2}
    tgt=${LANG_PAIR:3:2}
    echo "${src}-${tgt}"
    echo "Start binarizing $TRAIN/train.${src}-${tgt}..."    
    python ./fairseq_cli/preprocess.py  \
        --trainpref $TRAIN/train.${src}-${tgt} \
        --source-lang $src --target-lang $tgt \
        --destdir $DATA_BIN \
        --srcdict $DICT \
        --tgtdict $DICT \
        --workers 40
done