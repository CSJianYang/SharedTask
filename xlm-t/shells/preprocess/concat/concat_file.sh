INPUT_DIR=$1
OUTPUT_DIR=$2
src=$3
tgt=$4
new_src=$5
new_tgt=$6
mkdir -p $OUTPUT_DIR
if [ -f $OUTPUT_DIR/train.${new_src}-${new_tgt}.${new_src} ]; then
    rm $OUTPUT_DIR/train.${new_src}-${new_tgt}.${new_src}
fi
if [ -f $OUTPUT_DIR/train.${new_src}-${new_tgt}.${new_tgt} ]; then
    rm $OUTPUT_DIR/train.${new_src}-${new_tgt}.${new_tgt}
fi

for dir in $(ls $INPUT_DIR); do
    echo "$INPUT_DIR/$dir/"
    cat $INPUT_DIR/$dir/*.${src}.snt >> $OUTPUT_DIR/train.${new_src}-${new_tgt}.${new_src}
    cat $INPUT_DIR/$dir/*.${tgt}.snt >> $OUTPUT_DIR/train.${new_src}-${new_tgt}.${new_tgt}
done
wc -l $OUTPUT_DIR/train.${new_src}-${new_tgt}.${new_src}
wc -l $OUTPUT_DIR/train.${new_src}-${new_tgt}.${new_tgt}
head $OUTPUT_DIR/train.${new_src}-${new_tgt}.${new_src}
head $OUTPUT_DIR/train.${new_src}-${new_tgt}.${new_tgt}
tail $OUTPUT_DIR/train.${new_src}-${new_tgt}.${new_src}
tail $OUTPUT_DIR/train.${new_src}-${new_tgt}.${new_tgt}