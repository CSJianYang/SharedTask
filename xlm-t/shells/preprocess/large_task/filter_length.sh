FILTER=./scripts/SharedTask/filter_with_length.py
INPUT_DIR=/mnt/input/SharedTask/thunder/share_task_scarce_data/
ORIG_DIR=/mnt/input/SharedTask/thunder/share_task_final_data_v1/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/Bitext_v1/
PYTHON=/home/v-jiaya/miniconda3/envs/amlt8/bin/python
DEDUP=/home/v-jiaya/SharedTask/xlm-t/scripts/deduplicate_lines.py
LANGS=(be ff ku mi my ns om or wo yo zu)
for lang in ${LANGS[@]}; do
    INPUT_SRC=$INPUT_DIR/en-$lang.en.snt
    INPUT_TGT=$INPUT_DIR/en-$lang.$lang.snt
    OUTPUT_SRC=$INPUT_DIR/train.$lang-en.en
    OUTPUT_TGT=$INPUT_DIR/train.$lang-en.$lang
    echo "$INPUT_SRC -> $OUTPUT_SRC"
    echo "$INPUT_TGT -> $OUTPUT_TGT"
    $PYTHON $FILTER -src $INPUT_SRC -tgt $INPUT_TGT -new-src $OUTPUT_SRC -new-tgt $OUTPUT_TGT
    cat $INPUT_DIR/train.$lang-en.en $ORIG_DIR/${lang}en/train.$lang-en.en > $OUTPUT_DIR/${lang}en/train.$lang-en.en
    cat $INPUT_DIR/train.$lang-en.$lang $ORIG_DIR/${lang}en/train.$lang-en.$lang > $OUTPUT_DIR/${lang}en/train.$lang-en.$lang
    wc -l $OUTPUT_DIR/${lang}en/*
done


