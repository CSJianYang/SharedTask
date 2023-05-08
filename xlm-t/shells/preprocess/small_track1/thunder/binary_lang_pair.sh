src=$1
tgt=$2
DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
VALID=/mnt/input/SharedTask/thunder/flores101_dataset/devtest-code_spm/
echo "small-task1"
TRAIN=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/train/
DATA_BIN=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/data-bin/
echo "${src}-${tgt}"
if [ -f $TRAIN/train.${src}-${tgt}.${src} -a -f $TRAIN/train.${src}-${tgt}.${tgt} ]; then
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
fi
