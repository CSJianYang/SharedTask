op=$1

LANGS=(af am ar as ast ay az ba be bg bn br bs ca ceb cjk cs cy da de dyu el en es et fa ff fi fr fy ga gd gl gu ha he hi hr ht hu hy id ig ilo is it ja jv ka kac kam kea kg kk km kmb kmr kn ko ku ky lb lg ln lo lt luo lv mg mi mk ml mn mr ms mt my ne nl no ns ny oc om or pa pl ps pt qu ro ru sd shn si sk sl sn so sq sr ss su sv sw ta te tg th ti tl tn tr uk umb ur uz vi wo xh yi yo zh zu)
DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt

if [ "$op" == "small-task1-train" ]; then
    echo "small-task1-train"
    TRAIN=/mnt/input/SharedTask/thunder/small_task1/download/train/
    DATA_BIN=/mnt/input/SharedTask/thunder/small_task1/download/data-bin/
    for src in "${LANGS[@]}"; do
        for tgt in "${LANGS[@]}"; do
            echo "${src}-${tgt}"
            if [ ! -f $DATA_BIN/train.${src}-${tgt}.${src}.bin -a ! -f $DATA_BIN/train.${src}-${tgt}.${tgt}.bin ]; then
                if [ -f $TRAIN/train.${src}-${tgt}.${src} -a -f $TRAIN/train.${src}-${tgt}.${tgt} ]; then
                    echo "Start binarizing $TRAIN/train.${src}-${tgt}..."    
                    python ./fairseq_cli/preprocess.py  \
                        --trainpref $TRAIN/train.${src}-${tgt} \
                        --source-lang $src --target-lang $tgt \
                        --destdir $DATA_BIN \
                        --srcdict $DICT \
                        --tgtdict $DICT \
                        --workers 40
                fi
            fi
        done
    done
elif [ "$op" == "small-task1-valid" ]; then
    echo "small-task1-valid"
    LANGS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
    TRAIN=/mnt/input/SharedTask/thunder/flores101_dataset/dev-code_spm/
    DATA_BIN=/mnt/input/SharedTask/thunder/small_task1/download/data-bin/
    for src in "${LANGS[@]}"; do
        for tgt in "${LANGS[@]}"; do
            echo "${src}-${tgt}"
            if [ "${src}" != "${tgt}" ]; then
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
    echo "small-task2-train"
    TRAIN=/mnt/input/SharedTask/thunder/small_task2/download/train/
    DATA_BIN=/mnt/input/SharedTask/thunder/small_task2/download/data-bin/
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
elif [ "$op" == "small-task2-valid" ]; then
    echo "small-task2-valid"
    LANGS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
    TRAIN=/mnt/input/SharedTask/thunder/flores101_dataset/dev-code_spm/
    DATA_BIN=/mnt/input/SharedTask/thunder/small_task2/download/data-bin/
    for src in "${LANGS[@]}"; do
        for tgt in "${LANGS[@]}"; do
            if [ "${src}" != "${tgt}" ]; then
                echo "Start binarizing $TRAIN/train.{src}-${tgt}..."    
                python ./fairseq_cli/preprocess.py  \
                    --trainpref $TRAIN/valid \
                    --source-lang $src --target-lang $tgt \
                    --destdir $DATA_BIN \
                    --srcdict $DICT \
                    --tgtdict $DICT \
                    --workers 40
            fi
        done
    done
else
    TRAIN=/mnt/input/SharedTask/thunder/small_task2/download/train/
    DATA_BIN=/mnt/input/SharedTask/thunder/small_task2/download/data-bin/
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