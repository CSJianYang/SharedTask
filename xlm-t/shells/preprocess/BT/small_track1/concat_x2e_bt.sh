ORIG_DIR=/mnt/input/SharedTask/thunder/MonolingualData/all/
INPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/all_spm_split_10W_bt/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/all_spm_x2e_bt/
SPM_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/spm.model
mkdir -p $$OUTPUR_DIR
LANGS=(et hr hu mk sr)
mkdir -p $OUTPUT_DIR
for lang in ${LANGS[@]}; do
    echo "Concating $lang.."
    cat $INPUT_DIR/en*.2${lang} | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT_DIR/train.en-${lang}.${lang}
    head $ORIG_DIR/en.tsv.filt -n 6000000 > $OUTPUT_DIR/train.en-${lang}.en
done
wc -l $OUTPUT_DIR/*