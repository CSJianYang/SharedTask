LANG_PAIR=$1
SPLIT_NUM=$2
INPUT=/mnt/input/SharedTask/thunder/large_track/data/bt_spm/$LANG_PAIR/
OUTPUT=/mnt/input/SharedTask/thunder/large_track/data/bt_split${SPLIT_NUM}/
echo "INPUT | $INPUT"
python ./scripts/SharedTask/SplitTrainingData.py --input $INPUT --output $OUTPUT --split-num $SPLIT_NUM

