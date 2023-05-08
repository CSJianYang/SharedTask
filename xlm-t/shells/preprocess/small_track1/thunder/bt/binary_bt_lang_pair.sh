src=$1
tgt=$2
N=$3
START=$4
END=$5
echo "$src $tgt $N"
DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
for ((INDEX = $START ; INDEX < $END ; INDEX++ )); do
    TRAIN=/mnt/input/SharedTask/thunder/small_task1/data/bt_split${N}/train${INDEX}/
    DATA_BIN=/mnt/input/SharedTask/thunder/small_task2/data/bt_split-data-bin${N}/data-bin${INDEX}/
    echo "Start binarizing $TRAIN/train.{tgt}-${src}..."    
    python ./fairseq_cli/preprocess.py  \
        --trainpref $TRAIN/train.${src}-${tgt} \
        --source-lang $src --target-lang $tgt \
        --destdir $DATA_BIN \
        --srcdict $DICT \
        --tgtdict $DICT \
        --workers 40
done
