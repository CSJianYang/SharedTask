src=$1
N=$2
START=$3
END=$4
echo "$src $N"
VALID=/mnt/input/SharedTask/thunder/flores101_dataset/devtest-code_spm/
DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
for ((INDEX = $START ; INDEX < $END ; INDEX++ )); do
    TRAIN=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/parallel_bt2.5-split${N}/train${INDEX}/
    DATA_BIN=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/parallel_bt2.5_split-data-bin${N}/data-bin${INDEX}/
    echo "Start binarizing $TRAIN/train.${src}..."    
    python ./fairseq_cli/preprocess.py  \
        --trainpref $TRAIN/train --only-source \
        --source-lang ${src} \
        --destdir $DATA_BIN \
        --srcdict $DICT \
        --workers 40
done
