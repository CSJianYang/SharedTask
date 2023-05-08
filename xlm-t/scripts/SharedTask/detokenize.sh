INPUT_DIR=/mnt/input/SharedTask/thunder/share_task_final_data_v1/loen/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/Bitext_v1/loen/
DETOKENIZER=/home/v-jiaya/mosesdecoder/scripts/tokenizer/detokenizer.perl
for file in $(ls $INPUT_DIR); do
    echo "$INPUT_DIR/$file -> $OUTPUT_DIR/$file"
    l=$(echo $file | cut -d'.' -f 3)
    echo "$l"
    cat $INPUT_DIR/$file | $DETOKENIZER -threads 20 -l $l -no-escape > $OUTPUT_DIR/$file
done