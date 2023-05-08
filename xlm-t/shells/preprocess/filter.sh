op=$1

if [ "$op" == "monolingual-remove-empty-lines" ]; then
    echo "monolingual-remove-empty-lines"
    TEXT=/mnt/input/SharedTask/large-scale/MonolingualData/all_spm/
    NEWTEXT=/mnt/input/SharedTask/large-scale/MonolingualData/all_spm_no_empty_lines/
    mkdir -p $NEWTEXT
    for file in $(ls $TEXT); do
        if [ ! -f $NEWTEXT/$file ]; then
            cat $TEXT/$file | tr -s "\n" > $NEWTEXT/$file
        fi
        wc -l $TEXT/$file
        wc -l $NEWTEXT/$file
    done
else
    exit
fi



