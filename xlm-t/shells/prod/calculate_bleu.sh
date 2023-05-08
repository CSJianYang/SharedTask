LANGS=(af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu)
OUTPUT=/mnt/input/SharedTask/devtest_Product_Translation/Evaluation/Translation/
TARGET=/mnt/input/SharedTask/thunder/flores101_dataset/devtest-code/
BLEU_DIR=/mnt/input/SharedTask/devtest_Product_Translation/Evaluation/BLEU/
mkdir -p $BLEU_DIR
for src in ${LANGS[@]}; do
    for tgt in ${LANGS[@]}; do
        FOUT=$OUTPUT/${src}.2${tgt}
        FTGT=$TARGET/valid.${tgt}
        if [ "${src}" != "${tgt}" -a -f $FOUT ]; then
            echo "Saving BLEU to $BLEU_DIR/${src}-${tgt}.BLEU..."      
            cat $FOUT | sacrebleu -tok spm $FTGT | tee -a $BLEU_DIR/${src}-${tgt}.BLEU
        fi
    done
done

