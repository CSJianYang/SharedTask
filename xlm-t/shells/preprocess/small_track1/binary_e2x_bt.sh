echo "small-task1-train"
TRAIN=/mnt/input/SharedTask/thunder/small_task1/all_spm_e2x_bt/
DATA_BIN=/mnt/input/SharedTask/thunder/small_task1/all_spm_e2x_bt/bt_data-bin/
DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
LANG_PAIRS=(en-et en-hr en-hu en-mk en-sr)
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