LANG=$1
INPUT_DIR=/mnt/input/SharedTask/thunder/BT/4M-all/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/BT/4M-all-spm/
SPM_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/spm.model
mkdir -p $OUTPUT_DIR
INPUT=$INPUT_DIR/train.$LANG
OUTPUT=$OUTPUT_DIR/train.$LANG
echo "$INPUT -> $OUTPUT"
cat $INPUT | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT