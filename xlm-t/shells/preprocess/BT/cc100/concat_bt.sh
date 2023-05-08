ORIG_DIR=/mnt/input/SharedTask/thunder/MonolingualData/cc100/all_spm_split_10W/
INPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/cc100/all_spm_split_10W_bt/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/cc100/final/
SPM_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/spm.model
mkdir -p $$OUTPUR_DIR
LANGS=(or)
#LANGS=(be ff ku lo my ns om or)
mkdir -p $OUTPUT_DIR
for lang in ${LANGS[@]}; do
    echo "Concating en to $OUTPUT_DIR/train.${lang}-en.en..."
    cat $INPUT_DIR/${lang}*.2en | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT_DIR/train.${lang}-en.en
    #echo "Concating $lang to $OUTPUT_DIR/train.${lang}-en.${lang}..."
    #cat $ORIG_DIR/train.${lang}* >  $OUTPUT_DIR/train.${lang}-en.${lang}
done
wc -l $OUTPUT_DIR/*