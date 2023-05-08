src=$1
tgt=$2
INPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/all_spm_x2e_bt/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/small_task2/Filter_v1/parallel_bt_train/
FILTER=./scripts/SharedTask/small_track2/filter_parallel_bt.py
mkdir -p $$OUTPUR_DIR
python $FILTER -input $INPUT_DIR -output $OUTPUT_DIR