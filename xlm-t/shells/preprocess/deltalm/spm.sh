op=$1
SPM_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/spm.model
if [ "$op" == "small-task1-train" ]; then 
    echo "small-task1"
    TEXT=/mnt/input/SharedTask/thunder/small_task1/download/small_task1_filt/
    NEWTEXT=/mnt/input/SharedTask/thunder/small_task1/download/small_task1_filt_spm/
    mkdir -p $NEWTEXT
    for file in $(ls $TEXT); do
        if [ ! -f $NEWTEXT/$file ]; then
            INPUT=$TEXT/$file
            OUTPUT=$NEWTEXT/$file
            echo "$INPUT -> $OUTPUT"
            cat $INPUT | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT
        fi
    done
elif [ "$op" == "small-task2-train" ]; then
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
elif [ "$op" == "dev" ]; then
    echo "dev"
    TEXT=/mnt/input/SharedTask/thunder/flores101_dataset/dev-code/
    NEWTEXT=/mnt/input/SharedTask/thunder/flores101_dataset/dev-code_spm/
    mkdir -p $NEWTEXT
    for file in $(ls $TEXT); do
        if [ ! -f $NEWTEXT/$file ]; then
            INPUT=$TEXT/$file
            OUTPUT=$NEWTEXT/$file
            echo "$INPUT -> $OUTPUT"
            cat $INPUT | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT
        fi
    done
elif [ "$op" == "devtest" ]; then
    echo "devtest"
    TEXT=/mnt/input/SharedTask/thunder/flores101_dataset/devtest-code/
    NEWTEXT=/mnt/input/SharedTask/thunder/flores101_dataset/devtest-code_spm/
    mkdir -p $NEWTEXT
    for file in $(ls $TEXT); do
        if [ ! -f $NEWTEXT/$file ]; then
            INPUT=$TEXT/$file
            OUTPUT=$NEWTEXT/$file
            echo "$INPUT -> $OUTPUT"
            cat $INPUT | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT
        fi
    done
elif [ "$op" == "monolingual" ]; then
    echo "monolingual"
    TEXT=/mnt/input/SharedTask/thunder/MonolingualData/all/
    NEWTEXT=/mnt/input/SharedTask/thunder/MonolingualData/all_spm/
    mkdir -p $NEWTEXT
    for file in $(ls $TEXT); do
        if [ ! -f $NEWTEXT/$file ]; then
            INPUT=$TEXT/$file
            OUTPUT=$NEWTEXT/$file
            echo "$INPUT -> $OUTPUT"
            cat $INPUT | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT
        fi
    done
else
    exit
    echo "large"
    TEXT=/mnt/input/SharedTask/thunder/small_task1/download/small_task1_filt/
    NEWTEXT=/mnt/input/SharedTask/thunder/small_task1/download/small_task1_filt_spm/
    mkdir -p $NEWTEXT
    for file in $(ls $TEXT); do
        if [ ! -f $NEWTEXT/$file ]; then
            INPUT=$TEXT/$file
            OUTPUT=$NEWTEXT/$file
            echo "$INPUT -> $OUTPUT"
            cat $INPUT | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT
        fi
    done
fi

