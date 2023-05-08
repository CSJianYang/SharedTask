SPM_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/spm.model
INPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/cc100/final-filter/
ORIG_DIR=/mnt/input/SharedTask/thunder/Bt_v1/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/large_track/data/bt_spm/
#LANGS=(be ff ku lo my ns om or)
LANGS=(or)
for lang in ${LANGS[@]}; do
    echo "$INPUT_DIR/train.${lang}-en.* -> $OUTPUT_DIR/${lang}en/train.${lang}-en.*"
    mkdir -p $OUTPUT_DIR/${lang}en
    cp $INPUT_DIR/train.${lang}-en.* $OUTPUT_DIR/${lang}en/
    mkdir -p $ORIG_DIR/${lang}en
    echo "DESPM: $OUTPUT_DIR/${lang}en/train.${lang}-en.${lang} -> $ORIG_DIR/${lang}en/train.${lang}-en.${lang}"
    cat $OUTPUT_DIR/${lang}en/train.${lang}-en.${lang} | spm_decode --model=$SPM_MODEL --input_format=piece > $ORIG_DIR/${lang}en/train.${lang}-en.${lang}
    echo "DESPM: $OUTPUT_DIR/${lang}en/train.${lang}-en.en -> $ORIG_DIR/${lang}en/train.${lang}-en.en"
    cat $OUTPUT_DIR/${lang}en/train.${lang}-en.en | spm_decode --model=$SPM_MODEL --input_format=piece > $ORIG_DIR/${lang}en/train.${lang}-en.en
done
