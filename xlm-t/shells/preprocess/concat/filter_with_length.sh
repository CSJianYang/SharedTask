FILTER=./scripts/SharedTask/filter_with_length.py
DEDUP=./scripts/SharedTask/deduplicate_pairs.py
INPUT_DIR=/mnt/input/SharedTask/large-scale/ZC50-Updated/
ORIG_DIR=/mnt/input/SharedTask/thunder/Bitext_v2/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/Bitext_v1_sup/
OUTPUT_DIR2=/mnt/input/SharedTask/thunder/Bitext_v1/
#LANGS=(am bn el fa hu it ja ka lv ml pl pt ro sv uk ur)
lang=$1
INPUT_SRC=$INPUT_DIR/train/train.$lang-en.en
INPUT_TGT=$INPUT_DIR/train/train.$lang-en.$lang
OUTPUT_SRC=$INPUT_DIR/train-filter/train.$lang-en.en
OUTPUT_TGT=$INPUT_DIR/train-filter/train.$lang-en.$lang
echo "$INPUT_SRC -> $OUTPUT_SRC"
echo "$INPUT_TGT -> $OUTPUT_TGT"
python $FILTER -src $INPUT_SRC -tgt $INPUT_TGT -new-src $OUTPUT_SRC -new-tgt $OUTPUT_TGT
mkdir -p $OUTPUT_DIR/${lang}en/
wc -l $INPUT_DIR/train-filter/*
wc -l $ORIG_DIR/${lang}en/*
cat $INPUT_DIR/train-filter/train.$lang-en.en $ORIG_DIR/${lang}en/train.$lang-en.en > $OUTPUT_DIR/${lang}en/train.$lang-en.en
cat $INPUT_DIR/train-filter/train.$lang-en.$lang $ORIG_DIR/${lang}en/train.$lang-en.$lang > $OUTPUT_DIR/${lang}en/train.$lang-en.$lang
wc -l $OUTPUT_DIR/${lang}en/*
python $DEDUP -src $OUTPUT_DIR/${lang}en/train.$lang-en.en -tgt $OUTPUT_DIR/${lang}en/train.$lang-en.$lang -new-src $OUTPUT_DIR2/${lang}en/train.$lang-en.en -new-tgt $OUTPUT_DIR2/${lang}en/train.$lang-en.$lang
wc -l $OUTPUT_DIR2/${lang}en/*