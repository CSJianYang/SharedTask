echo "small-task2"
TEXT=/mnt/input/SharedTask/thunder/small_task2/download/small_task2_filt/
NEWTEXT=/mnt/input/SharedTask/thunder/small_task2/download/small_task2_filt_spm/
mkdir -p $NEWTEXT
for file in $(ls $TEXT); do
    if [ ! -f $NEWTEXT/$file ]; then
        INPUT=$TEXT/$file
        OUTPUT=$NEWTEXT/$file
        echo "$INPUT -> $OUTPUT"
        cat $INPUT | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT
    fi
done
   

