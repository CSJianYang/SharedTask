export CUDA_VISIBLE_DEVICES=0
DATA_BIN=/mnt/input/SharedTask/large-scale/small_task1/download/data-bin/
TEXT=/mnt/input/SharedTask/large-scale/small_task1/download/flores101_dataset/dev-code/
SPM_MODEL=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_175M/flores101_mm100_175M/sentencepiece.bpe.model
LANG_PAIRS=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_175M/flores101_mm100_175M/language_pairs.txt
src=$1
tgt=$2
beam=$3
MODEL=$4


lenpen=1.0
INPUT=$TEXT/${src}.dev
FTGT=$TEXT/${tgt}.dev



echo $MODEL
FOUT=$INPUT.2${tgt}
cat $INPUT | python ./fairseq_cli/interactive.py $DATA_BIN \
    --path $MODEL \
    --encoder-langtok "src" --decoder-langtok --langtoks '{"main":("src", "tgt")}' \
    --task translation_multi_simple_epoch \
    --langs "af,am,ar,as,ast,ay,az,ba,be,bg,bn,br,bs,ca,ceb,cjk,cs,cy,da,de,dyu,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kac,kam,kea,kg,kk,km,kmb,kmr,kn,ko,ku,ky,lb,lg,ln,lo,lt,luo,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,no,ns,ny,oc,om,or,pa,pl,ps,pt,qu,ro,ru,sd,shn,si,sk,sl,sn,so,sq,sr,ss,su,sv,sw,ta,te,tg,th,ti,tl,tn,tr,uk,umb,ur,uz,vi,wo,xh,yi,yo,zh,zu" \
    --lang-pairs $LANG_PAIRS \
    --source-lang $src --target-lang $tgt \
    --buffer-size 10000 --batch-size 32 --beam $beam --lenpen $lenpen \
    --bpe sentencepiece --sentencepiece-model $SPM_MODEL --no-progress-bar --fp16 > $FOUT
# --same-lang-per-batch

cat $FOUT | grep -P "^D" | cut -f 3- > $FOUT.out
cat $FOUT.out | sacrebleu -l $src-$tgt $FTGT