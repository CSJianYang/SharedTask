cat $FOUT.out | sacrebleu -tok spm $FTGT | tee -a $BLEU_DIR/${src}-${tgt}.BLEU