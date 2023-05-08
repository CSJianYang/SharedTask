LANG_PAIR=$1
INPUT_DIR=/mnt/input/SharedTask/thunder/share_task_final_BT/$LANG_PAIR/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/large_track/data/bt_spm/$LANG_PAIR/
SPM_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/spm.model
mkdir -p $OUTPUT_DIR
for file in $(ls $INPUT_DIR); do
    echo "SPM $file..."
    if [ ! -f $OUTPUT_DIR/$file ]; then
        INPUT=$INPUT_DIR/$file
        OUTPUT=$OUTPUT_DIR/$file
        echo "$INPUT -> $OUTPUT"
        cat $INPUT | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT
    fi
done