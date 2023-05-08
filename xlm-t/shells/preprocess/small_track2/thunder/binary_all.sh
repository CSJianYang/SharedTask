DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
TRAIN=/mnt/input/SharedTask/thunder/small_task2/download/train/
VALID=/mnt/input/SharedTask/thunder/flores101_dataset/devtest-code_spm/
DATA_BIN=/mnt/input/SharedTask/thunder/small_task2/download/data-bin/
LANG_PAIRS=(en-id en-jv en-ms en-ta en-tl id-jv id-ms id-ta id-tl jv-ms jv-ta jv-tl ms-ta ms-tl ta-tl)
for LANG_PAIR in ${LANG_PAIRS[@]}; do
    src=${LANG_PAIR:0:2}
    tgt=${LANG_PAIR:3:2}
    echo "Start binarizing $TRAIN/train.${src}-${tgt}..."  
    python ./fairseq_cli/preprocess.py  \
        --trainpref $TRAIN/train.${src}-${tgt} \
        --source-lang $src --target-lang $tgt \
        --destdir $DATA_BIN \
        --srcdict $DICT \
        --tgtdict $DICT \
        --workers 40
    python ./fairseq_cli/preprocess.py  \
        --validpref $VALID/valid \
        --source-lang $src --target-lang $tgt \
        --destdir $DATA_BIN \
        --srcdict $DICT \
        --tgtdict $DICT \
        --workers 40
done
