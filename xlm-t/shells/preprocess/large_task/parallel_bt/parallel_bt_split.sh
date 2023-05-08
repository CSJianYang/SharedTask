LANG=$1
SPLIT_NUM=$2
INPUT=/mnt/input/SharedTask/thunder/BT/4M-all-spm/train.$LANG
OUTPUT=/mnt/input/SharedTask/thunder/large_track/data/parallel_bt_split${SPLIT_NUM}/
echo "INPUT | $INPUT"
python ./scripts/SharedTask/SplitTrainingData.py --input $INPUT --output $OUTPUT --split-num $SPLIT_NUM


