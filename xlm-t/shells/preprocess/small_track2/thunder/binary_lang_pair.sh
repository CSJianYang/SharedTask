src=$1
tgt=$2
DICT=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/dict.txt
TRAIN=/mnt/input/SharedTask/thunder/small_task2/download/train/
VALID=/mnt/input/SharedTask/thunder/flores101_dataset/devtest-code_spm/
DATA_BIN=/mnt/input/SharedTask/thunder/small_task2/download/data-bin/

echo "Start binarizing $TRAIN/train.${src}-${tgt}..."  
python ./fairseq_cli/preprocess.py  \
    --trainpref $TRAIN/train.${src}-${tgt} \
    --source-lang $src --target-lang $tgt \
    --destdir $DATA_BIN \
    --srcdict $DICT \
    --tgtdict $DICT \
    --workers 20
python ./fairseq_cli/preprocess.py  \
    --validpref $VALID/valid \
    --source-lang $src --target-lang $tgt \
    --destdir $DATA_BIN \
    --srcdict $DICT \
    --tgtdict $DICT \
    --workers 20

