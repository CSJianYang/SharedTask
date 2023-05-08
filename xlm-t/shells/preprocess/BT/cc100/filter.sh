INPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/cc100/final/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/cc100/filter/
FILTER=./scripts/SharedTask/filter_with_length.py
mkdir -p $$OUTPUR_DIR
LANGS=(be ff ku lo my ns om or)
for lang in ${LANGS[@]}; do
    echo "Filter $lang.."
    python $FILTER -src $INPUT_DIR/train.${lang}-en.${lang} -tgt $INPUT_DIR/train.${lang}-en.en -new-src $OUTPUT_DIR/train.${lang}-en.${lang} -new-tgt $OUTPUT_DIR/train.${lang}-en.en
done
wc -l $OUTPUT_DIR/*