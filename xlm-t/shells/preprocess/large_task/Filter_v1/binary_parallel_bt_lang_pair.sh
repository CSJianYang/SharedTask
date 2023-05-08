src=$1
N=$2
START=$3
END=$4
echo "$src $N"
DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
for ((INDEX = $START ; INDEX < $END ; INDEX++ )); do
    TRAIN=/mnt/input/SharedTask/thunder/large_track/data/Filter_v1/parallel_bt_split${N}/train${INDEX}/
    DATA_BIN=/mnt/input/SharedTask/thunder/large_track/data/Filter_v1/parallel_bt_split-data-bin${N}/data-bin${INDEX}/
    echo "Start binarizing $TRAIN/train.${src}..."    
    python ./fairseq_cli/preprocess.py  \
        --trainpref $TRAIN/train --only-source \
        --source-lang ${src} \
        --destdir $DATA_BIN \
        --srcdict $DICT \
        --workers 20
done
