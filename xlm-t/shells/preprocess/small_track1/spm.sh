op=$1

if [ "$op" == "small-task1-train" ]; then
    echo "small-task1"
    TEXT=/mnt/input/SharedTask/large-scale/small_task1/download/small_task1_filt_spm/
    NEWTEXT=/mnt/input/SharedTask/large-scale/small_task1/download/train/
    mkdir -p $NEWTEXT
    SPM_MODEL=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_615M/flores101_mm100_615M/sentencepiece.bpe.model
    for file in $(ls $TEXT); do
        if [ ! -f $NEWTEXT/$file ]; then
            INPUT=$TEXT/$file
            OUTPUT=$NEWTEXT/$file
            echo "$INPUT -> $OUTPUT"
            cat $INPUT | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT
        fi
    done
elif [ "$op" == "small-task1-test" ]; then
    echo "small-task1"
    TEXT=/mnt/input/SharedTask/large-scale/small_task1/download/ProdTestData/
    NEWTEXT=/mnt/input/SharedTask/large-scale/small_task1/download/ProdTestData_spm/
    mkdir -p $NEWTEXT
    SPM_MODEL=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_615M/flores101_mm100_615M/sentencepiece.bpe.model
    for file in $(ls $TEXT); do
        if [ ! -f $NEWTEXT/$file ]; then
            INPUT=$TEXT/$file
            OUTPUT=$NEWTEXT/$file
            echo "$INPUT -> $OUTPUT"
            cat $INPUT | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT
        fi
    done
elif [ "$op" == "small-task2" ]; then
    echo "small-task2"
    TEXT=/mnt/input/SharedTask/large-scale/small_task2/download/small_task2_filt/
    NEWTEXT=/mnt/input/SharedTask/large-scale/small_task2/download/small_task2_filt_spm/
    mkdir -p $NEWTEXT
    SPM_MODEL=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_615M/flores101_mm100_615M/sentencepiece.bpe.model
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
    TEXT=/mnt/input/SharedTask/large-scale/MonolingualData/all/
    NEWTEXT=/mnt/input/SharedTask/large-scale/MonolingualData/all_spm/
    mkdir -p $NEWTEXT
    SPM_MODEL=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_615M/flores101_mm100_615M/sentencepiece.bpe.model
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
    TEXT=/mnt/input/SharedTask/large-scale/small_task1/download/small_task1_filt/
    NEWTEXT=/mnt/input/SharedTask/large-scale/small_task1/download/small_task1_filt_spm/
    mkdir -p $NEWTEXT
    SPM_MODEL=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_615M/flores101_mm100_615M/sentencepiece.bpe.model
    for file in $(ls $TEXT); do
        if [ ! -f $NEWTEXT/$file ]; then
            INPUT=$TEXT/$file
            OUTPUT=$NEWTEXT/$file
            echo "$INPUT -> $OUTPUT"
            cat $INPUT | spm_encode --model=$SPM_MODEL --output_format=piece > $OUTPUT
        fi
    done
fi

