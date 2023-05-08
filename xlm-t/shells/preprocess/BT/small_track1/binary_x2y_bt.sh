echo "small-task1-train"
TRAIN=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/bt_train/
DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
DATA_BIN=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/bt_data-bin/
LANG_PAIRS=(et-hr et-hu et-sr et-mk hr-hu hr-sr hr-mk hu-sr hu-mk sr-mk hr-et hu-et sr-et mk-et hu-hr sr-hr mk-hr sr-hu mk-hu mk-sr)
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
