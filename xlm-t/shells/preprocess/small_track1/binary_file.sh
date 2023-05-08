op=$1
src=$2
tgt=$3
DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
VALID=/mnt/input/SharedTask/thunder/flores101_dataset/devtest-code_spm/
if [ "$op" == "small-task1-train" ]; then
    echo "small-task1-train"
    TRAIN=/mnt/input/SharedTask/thunder/small_task1/train-filter/
    DATA_BIN=/mnt/input/SharedTask/thunder/small_task1/data-bin/
    echo "${src}-${tgt}"
    #if [ ! -f $DATA_BIN/train.${src}-${tgt}.${src}.bin -a ! -f $DATA_BIN/train.${src}-${tgt}.${tgt}.bin ]; then
    if [ -f $TRAIN/train.${src}-${tgt}.${src} -a -f $TRAIN/train.${src}-${tgt}.${tgt} ]; then
        echo "Start binarizing $TRAIN/train.${src}-${tgt}..."    
        python ./fairseq_cli/preprocess.py  \
            --trainpref $TRAIN/train.${src}-${tgt} \
            --source-lang $src --target-lang $tgt \
            --destdir $DATA_BIN \
            --srcdict $DICT \
            --tgtdict $DICT \
            --workers 40
        python ./fairseq_cli/preprocess.py  \
            --validpref $VALID/valid \
            --source-lang $src --target-lang $tgt \
            --destdir $DATA_BIN \
            --srcdict $DICT \
            --tgtdict $DICT \
            --workers 40
    fi
    #fi
elif [ "$op" == "small-task1-test" ]; then
    echo "small-task1-test"
    TRAIN=/mnt/input/SharedTask/large-scale/small_task1/download/train/
    DATA_BIN=/mnt/input/SharedTask/large-scale/small_task1/download/data-bin/
    for src in "${LANGS[@]}"; do
        for tgt in "${LANGS[@]}"; do
            echo "${src}-${tgt}"
            if [ -f $TRAIN/valid.${src}-${tgt}.${src} -a -f $TRAIN/valid.${src}-${tgt}.${tgt} ]; then
                echo "Start binarizing $TRAIN/valid.${src}-${tgt}..."    
                python ./fairseq_cli/preprocess.py  \
                    --validpref $TRAIN/valid \
                    --source-lang $src --target-lang $tgt \
                    --destdir $DATA_BIN \
                    --srcdict $DICT \
                    --tgtdict $DICT \
                    --workers 40
            fi
        done
    done
elif [ "$op" == "small-task2-train" ]; then
    TRAIN=/mnt/input/SharedTask/large-scale/small_task2/download/train/
    DATA_BIN=/mnt/input/SharedTask/large-scale/small_task2/download/data-bin/
    for src in "${LANGS[@]}"; do
        for tgt in "${LANGS[@]}"; do
            if [ -f $TRAIN/train.${src}-${tgt}.${src} -a -f $TRAIN/train.${src}-${tgt}.${tgt} ]; then
                echo "Start binarizing $TRAIN/train.{src}-${tgt}..."    
                python ./fairseq_cli/preprocess.py  \
                    --trainpref $TRAIN/train.${src}-${tgt} \
                    --source-lang $src --target-lang $tgt \
                    --destdir $DATA_BIN \
                    --srcdict $DICT \
                    --tgtdict $DICT \
                    --workers 40
            fi
        done
    done
elif [ "$op" == "small-task2-test" ]; then
    TRAIN=/mnt/input/SharedTask/large-scale/small_task2/download/train/
    DATA_BIN=/mnt/input/SharedTask/large-scale/small_task2/download/data-bin/
    for src in "${LANGS[@]}"; do
        for tgt in "${LANGS[@]}"; do
            if [ -f $TRAIN/valid.${src}-${tgt}.${src} -a -f $TRAIN/valid.${src}-${tgt}.${tgt} ]; then
                echo "Start binarizing $TRAIN/train.{src}-${tgt}..."    
                python ./fairseq_cli/preprocess.py  \
                    --trainpref $TRAIN/valid.${src}-${tgt} \
                    --source-lang $src --target-lang $tgt \
                    --destdir $DATA_BIN \
                    --srcdict $DICT \
                    --tgtdict $DICT \
                    --workers 40
            fi
        done
    done
else
    TRAIN=/mnt/input/SharedTask/large-scale/small_task2/download/train/
    DATA_BIN=/mnt/input/SharedTask/large-scale/small_task2/download/data-bin/
    for src in "${LANGS[@]}"; do
        for tgt in "${LANGS[@]}"; do
            if [ -f $TRAIN/train.${src}-${tgt}.${src} -a -f $TRAIN/train.${src}-${tgt}.${tgt} ]; then
                #echo "Start binarizing $TRAIN/train.{src}-${tgt}..."    
                python ./fairseq_cli/preprocess.py  \
                    --trainpref $TRAIN/train.${src}-${tgt} \
                    --source-lang $src --target-lang $tgt \
                    --destdir $DATA_BIN \
                    --srcdict $DICT \
                    --tgtdict $DICT \
                    --workers 40
            fi
        done
    done
fi