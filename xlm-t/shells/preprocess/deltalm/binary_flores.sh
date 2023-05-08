DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
LANG_PAIRS=(en-et en-hr en-hu en-mk en-sr et-hr et-hu et-mk et-sr hr-hu hr-mk hr-sr hu-mk hu-sr mk-sr sr-mk)
TRAIN=/mnt/input/SharedTask/thunder/flores101_dataset/dev-code_spm/
DATA_BIN=/mnt/input/SharedTask/thunder/small_task1/download/data-bin/
for LANG_PAIR in ${LANG_PAIRS[@]}; do
    src=$(echo $LANG_PAIR | cut -d'-' -f1)
    tgt=$(echo $LANG_PAIR | cut -d'-' -f2)
    echo "${src}-${tgt}"
    echo "Start binarizing $TRAIN/valid.${src}-${tgt}..."    
    python ./fairseq_cli/preprocess.py  \
        --validpref $TRAIN/valid \
        --source-lang $src --target-lang $tgt \
        --destdir $DATA_BIN \
        --srcdict $DICT \
        --tgtdict $DICT \
        --workers 40
done