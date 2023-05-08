export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES=0
DATA_BIN=/mnt/input/SharedTask/thunder/large_track/data/split-data-bin80/data-bin0/
TEXT=/mnt/input/SharedTask/thunder/flores101_dataset/devtest-code/
SPM_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/spm.model
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu"
LANG_PAIRS="en-et,et-en,en-hr,hr-en,en-hu,hu-en,en-mk,mk-en,en-sr,sr-en,et-hr,hr-et,et-hu,hu-et,et-mk,mk-et,et-sr,sr-et,hr-hu,hu-hr,hr-mk,mk-hr,hr-sr,sr-hr,hu-mk,mk-hu,hu-sr,sr-hu,mk-sr,sr-mk"
src=$1
tgt=$2
batchsize=$3
beam=$4
MODEL=$5


lenpen=1.0
INPUT=$TEXT/valid.${src}
FTGT=$TEXT/valid.${tgt}


echo "$INPUT | Beam: $beam | $MODEL"
FOUT=$INPUT.2${tgt}
cat $INPUT | python ./fairseq_cli/interactive.py $DATA_BIN \
    --path $MODEL \
    --encoder-langtok "tgt" --langtoks '{"main":("tgt",None)}' \
    --task translation_multi_simple_epoch \
    --langs $LANGS \
    --lang-pairs $LANG_PAIRS \
    --source-lang $src --target-lang $tgt --truncate-source \
    --buffer-size 10000 --batch-size $batchsize --beam $beam --lenpen $lenpen \
    --bpe sentencepiece --sentencepiece-model $SPM_MODEL --no-progress-bar --fp16 > $FOUT
cat $FOUT | grep -P "^D" | cut -f 3- > $FOUT.out

BLEU_DIR=/mnt/input/SharedTask/thunder/flores101_dataset/SmallTask1/BLEU/
mkdir -p $BLEU_DIR
evaluation_tok="spm"
echo "Saving BLEU to $BLEU_DIR/${src}-${tgt}.BLEU..."
echo "$MODEL" | tee -a $BLEU_DIR/${src}-${tgt}.BLEU
if [ $evaluation_tok == "spm" ]; then
    cat $FOUT.out | sacrebleu -tok spm $FTGT | tee -a $BLEU_DIR/${src}-${tgt}.BLEU
else
    cat $FOUT.out | sacrebleu -l $src-$tgt $FTGT | tee -a $BLEU_DIR/${src}-${tgt}.BLEU
fi