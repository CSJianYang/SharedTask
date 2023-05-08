TEXT=/mnt/input/SharedTask/thunder/small_task2/download/small_task2_filt_spm/
TRAIN=/mnt/input/SharedTask/thunder/small_task2/download/train/
src=$1
tgt=$2
echo "Concat ${src}-${tgt}..."
cat $TEXT/*.${src}-${tgt}.${src} > $TRAIN/train.${src}-${tgt}.${src}       
cat $TEXT/*.${src}-${tgt}.${tgt} > $TRAIN/train.${src}-${tgt}.${tgt}