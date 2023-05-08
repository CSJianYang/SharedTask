src=$1
tgt=$2
INPUT_DIR=/mnt/input/SharedTask/thunder/small_task1/all_spm_x2e_bt/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/parallel_bt_train/
FILTER=./scripts/SharedTask/filter_with_length.py
mkdir -p $$OUTPUR_DIR
python $FILTER -src $INPUT_DIR/train.${src}-${tgt}.${src} -tgt $INPUT_DIR/train.${src}-${tgt}.${tgt} -new-src $OUTPUT_DIR/train.${src}-${tgt}.${src} -new-tgt $OUTPUT_DIR/train.${src}-${tgt}.${tgt}