ORIG_DIR=/mnt/input/SharedTask/thunder/MonolingualData/all_spm/
INPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/all_spm_split_10W_bt_iter2/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/all_spm_x2y_bt/
SPM_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/spm.model
mkdir -p $$OUTPUR_DIR
src=$1
LANGS=(id jv ms ta tl)
mkdir -p $OUTPUT_DIR
for tgt in ${LANGS[@]}; do
    if [ ${src} != ${tgt} ]; then
        echo "Concating $INPUT_DIR/${src}.2${tgt} -> $OUTPUT_DIR/train.${tgt}-${src}.${tgt}"
        cat $INPUT_DIR/${src}*.2${tgt} | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT_DIR/train.${tgt}-${src}.${tgt}
        echo "Copying $ORIG_DIR/${src}.tsv -> $OUTPUT_DIR/train.${tgt}-${src}.${src}"
        cat $ORIG_DIR/${src}.tsv > $OUTPUT_DIR/train.${tgt}-${src}.${src}
    fi
done
wc -l $OUTPUT_DIR/*