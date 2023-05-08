INDEX=$1
TRAIN=/mnt/input/SharedTask/thunder/large_track/data/split/train${INDEX}/
VALID=/mnt/input/SharedTask/thunder/flores101_dataset/devtest-code_spm/
DATA_BIN=/mnt/input/SharedTask/thunder/large_track/data/split-data-bin/data-bin${INDEX}/
DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
LANGS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
for src in "${LANGS[@]}"; do
    for tgt in "${LANGS[@]}"; do
        if [ "${src}" != "${tgt}" ]; then
            if [ -f $TRAIN/train.${src}-${tgt}.${src} ]; then 
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
            elif [ -f $TRAIN/train.${tgt}-${src}.${src} ]; then 
                echo "Start binarizing $TRAIN/train.${src}-${tgt}..."    
                python ./fairseq_cli/preprocess.py  \
                    --trainpref $TRAIN/train.${tgt}-${src} \
                    --source-lang $tgt --target-lang $src \
                    --destdir $DATA_BIN \
                    --srcdict $DICT \
                    --tgtdict $DICT \
                    --workers 40
                python ./fairseq_cli/preprocess.py  \
                    --validpref $VALID/valid \
                    --source-lang $tgt --target-lang $src \
                    --destdir $DATA_BIN \
                    --srcdict $DICT \
                    --tgtdict $DICT \
                    --workers 40
            else
                echo "No ${src}-${tgt} Data"
            fi
        fi
    done
done