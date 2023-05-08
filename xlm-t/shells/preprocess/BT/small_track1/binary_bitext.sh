src=$1
tgt=$2
N=$3
echo "$src $tgt $N"
VALID=/mnt/input/SharedTask/thunder/flores101_dataset/devtest-code_spm/
DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
for ((INDEX = 0 ; INDEX < $N ; INDEX++ )); do
    TRAIN=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/split${N}/train${INDEX}/
    DATA_BIN=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/split-data-bin${N}/data-bin${INDEX}/
    echo "Start binarizing $TRAIN/train.${tgt}-${src}..."    
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
