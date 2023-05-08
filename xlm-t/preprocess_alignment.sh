RAW=/home/v-jiaya/DeepMNMT/data/xlmr-data/valid-set/
TRAIN=/home/v-jiaya/DeepMNMT/data/xlmr-data/valid-set-spm/
SPM_MODEL=/home/v-jiaya/DeepMNMT/premodels/xlmr.base/sentencepiece.bpe.model
src=en
for tgt in fr de fi cs et tr lv ro hi gu; do
    echo "Start binarize $TRAIN/train.${src}-${tgt}..."
    spm_encode --model=$SPM_MODEL --output_format=piece < $RAW/valid.${src}-${tgt}.${src} > $TRAIN/valid.${src}-${tgt}.${src}
    spm_encode --model=$SPM_MODEL --output_format=piece < $RAW/valid.${src}-${tgt}.${tgt} > $TRAIN/valid.${src}-${tgt}.${tgt}
 
    #/home/v-jiaya/anaconda3-fairseq-0.9.0/bin/python /home/v-jiaya/DeepMNMT/xlm-t/fairseq_cli/preprocess.py --source-lang ${src} --target-lang ${tgt} \
    #        --trainpref $TRAIN/train.$src-$tgt \
    #        --destdir $DATA_BIN \
    #        --srcdict /home/v-jiaya/DeepMNMT/data-bin/dict.en.txt \
    #        --tgtdict /home/v-jiaya/DeepMNMT/data-bin/dict.en.txt \
    #        --align-suffix align \
    #        --workers 40
done