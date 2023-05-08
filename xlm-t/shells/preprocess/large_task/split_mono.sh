op=$1
N_LINES=100000

if [ "$op" == "monolingual" ]; then
    echo "monolingual"
    TEXT=/mnt/input/SharedTask/large-scale/MonolingualData/all_spm/
    NEWTEXT=/mnt/input/SharedTask/large-scale/MonolingualData/all_spm_split_10W/
    mkdir -p $NEWTEXT
    for file in $(ls $TEXT); do
        if [ ! -f $NEWTEXT/$file ]; then
            INPUT=$TEXT/$file
            OUTPUT=$NEWTEXT/$file
            echo "$INPUT -> $OUTPUT"
            split -l $N_LINES -d -a 4 $INPUT $OUTPUT
        fi
    done
    wc -l $NEWTEXT/*
elif [ "$op" == "monolingual-cc100" ]; then
    echo "monolingual-cc100"
    TEXT=/mnt/input/SharedTask/thunder/MonolingualData/cc100/
    LANGS=(be ff ku lo my ns om or)
    mkdir -p $TEXT/all_spm/
    for lg in ${LANGS[@]}; do
        cat $TEXT/raw/$lg/* > $TEXT/all_spm/train.$lg
    done
    wc -l $TEXT/all_spm/*
    NEWTEXT=$TEXT/all_spm_split_10W/
    mkdir -p $NEWTEXT
    for file in $(ls $TEXT/all_spm/); do
        if [ ! -f $NEWTEXT/$file ]; then
            INPUT=$TEXT/all_spm/$file
            OUTPUT=$NEWTEXT/$file
            echo "$INPUT -> $OUTPUT"
            split -l $N_LINES -d -a 4 $INPUT $OUTPUT
        fi
    done
    wc -l $NEWTEXT/*
else
    exit
fi