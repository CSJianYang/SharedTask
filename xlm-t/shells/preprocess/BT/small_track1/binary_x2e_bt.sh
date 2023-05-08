TRAIN=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/parallel_bt_train/
DATA_BIN=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/parallel_bt_data-bin/
DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
LANGS=(en et hr hu mk sr)
for LANG in ${LANGS[@]}; do
    echo "Start binarizing $TRAIN/train.${LANG}..."    
    python ./fairseq_cli/preprocess.py  \
        --trainpref $TRAIN/train --only-source \
        --source-lang ${LANG} \
        --destdir $DATA_BIN \
        --srcdict $DICT \
        --workers 40
done