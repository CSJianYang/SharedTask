op=$1
N_LINES=2000000

if [ "$op" == "monolingual" ]; then
    echo "monolingual"
    TEXT=/mnt/input/SharedTask/large-scale/MonolingualData/all_spm/
    NEWTEXT=/mnt/input/SharedTask/large-scale/MonolingualData/all_spm_split/
    mkdir -p $NEWTEXT
    for file in $(ls $TEXT); do
        if [ ! -f $NEWTEXT/$file ]; then
            INPUT=$TEXT/$file
            OUTPUT=$NEWTEXT/$file
            echo "$INPUT -> $OUTPUT"
            split -l $N_LINES -d -a 3 $INPUT $OUTPUT
        fi
    done
    wc -l $NEWTEXT/*
else
    exit
fi