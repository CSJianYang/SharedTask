ALIGN=/home/v-jiaya/fast_align/build/fast_align
ATOOLS=/home/v-jiaya/fast_align/build/atools
PREPROCESS=/home/v-jiaya/DeepMNMT/xlm-t/fairseq_cli/preprocess.py
PYTHON=/home/v-jiaya/anaconda3/bin/python
ORIG_TRAIN=/home/v-jiaya/DeepMNMT/data/xlmr-data/spm-data/
TRAIN=/home/v-jiaya/DeepMNMT/data/xlmr-data/spm-data-filter/
NEW_TRAIN=/home/v-jiaya/DeepMNMT/alignment/train/
EM_STEP=10
FILTER=/home/v-jiaya/DeepMNMT/xlm-t/filter.py
mkdir -p $ORIG_TRAIN $TRAIN
for lp in en-fr en-de en-fi en-cs en-et en-tr en-lv en-ro en-hi en-gu; do
    src=$(echo $lp | cut -d "-" -f 1)
    tgt=$(echo $lp | cut -d "-" -f 2)
    for splt in valid train; do
        echo "Start aligning ${splt}: ${src}-${tgt} "
        $PYTHON $FILTER -src-set $ORIG_TRAIN/$splt.${src}-${tgt}.${src} -tgt-set $ORIG_TRAIN/$splt.${src}-${tgt}.${tgt} -new-src-set $TRAIN/$splt.${src}-${tgt}.${src} -new-tgt-set $TRAIN/$splt.${src}-${tgt}.${tgt}
        paste $TRAIN/$splt.${src}-${tgt}.${src} $TRAIN/$splt.${src}-${tgt}.${tgt} | awk -F '\t' '{print $1 " ||| " $2}'  > $NEW_TRAIN/$splt.${src}-${tgt}
        $ALIGN -i $NEW_TRAIN/$splt.${src}-${tgt} -d -o -v -I $EM_STEP > $NEW_TRAIN/$splt.${src}-${tgt}.forward.align
        $ALIGN -i $NEW_TRAIN/$splt.${src}-${tgt} -d -o -v -r -I $EM_STEP > $NEW_TRAIN/$splt.${src}-${tgt}.backward.align
        $ATOOLS -i $NEW_TRAIN/$splt.${src}-${tgt}.forward.align -j $NEW_TRAIN/$splt.${src}-${tgt}.backward.align -c grow-diag-final-and > $NEW_TRAIN/$splt.${src}-${tgt}.align
        rm $NEW_TRAIN/$splt.${src}-${tgt}.forward.align $NEW_TRAIN/$splt.${src}-${tgt}.backward.align $NEW_TRAIN/$splt.${src}-${tgt}
    done
    #"Start binarizing ${src}-${tgt} "
    #DATA_BIN=/home/v-jiaya/DeepMNMT/data/xlmr-data/alignment-data-bin/
    #$PYTHON $PREPROCESS --source-lang $src --target-lang $tgt \
    #--trainpref $TRAIN/train --validpref $TRAIN/valid \
    #--destdir $DATA_BIN \
    #--srcdict /home/v-jiaya/DeepMNMT/data/xlmr-data/data-bin/dict.en.txt \
    #--tgtdict /home/v-jiaya/DeepMNMT/data/xlmr-data/data-bin/dict.en.txt \
    #--align-suffix "align" \
    #--workers 40
done