ORIG_DIR=/mnt/input/SharedTask/thunder/MonolingualData/all_spm_split_10W/
INPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/all_spm_split_10W_bt/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/all_spm_e2x_bt/
SPM_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/spm.model
mkdir -p $$OUTPUR_DIR
LANGS=(id jv ms ta tl)
mkdir -p $OUTPUT_DIR
for lang in ${LANGS[@]}; do
    echo "Concating $lang.."
    cat $INPUT_DIR/${lang}*.2en | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT_DIR/train.en-${lang}.en
    cat $ORIG_DIR/${lang}* > $OUTPUT_DIR/train.en-${lang}.${lang}
done
wc -l $OUTPUT_DIR/*