export CUDA_VISIBLE_DEVICES=0
DATA_BIN=/mnt/input/SharedTask/thunder/small_task1/data-bin/
INPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/all_spm_split_10W/
OUTPUT_DIR=/mnt/input/SharedTask/thunder/MonolingualData/all_spm_split_10W_bt/
mkdir -p $OUTPUT_DIR
src=$1
tgt=$2
beam=$3
MODEL=$4
INPUT_FILE=$5
OUTPUT_FILE=$6
BATCH_SIZE=$7
INPUT=$INPUT_DIR/$INPUT_FILE
OUTPUT=$OUTPUT_DIR/$OUTPUT_FILE
echo "${src}->${tgt} | beam $beam | MODEL $MODEL"
echo "INPUT $INPUT | OUTPUT $OUTPUT"
echo "BATCH_SIZE $BATCH_SIZE | BUFFER_SIZE $BUFFER_SIZE"
lenpen=1.0


cat $INPUT | python interactive.py $DATA_BIN \
    --path $MODEL \
    --encoder-langtok "tgt" --langtoks '{"main":("tgt", "None")}' \
    --task translation_multi_simple_epoch \
    --langs "af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu" --truncate-source \
    --lang-pairs "en-et,et-en,en-hr,hr-en,en-hu,hu-en,en-mk,mk-en,en-sr,sr-en,et-hr,hr-et,et-hu,hu-et,et-mk,mk-et,et-sr,sr-et,hr-hu,hu-hr,hr-mk,mk-hr,hr-sr,sr-hr,hu-mk,mk-hu,hu-sr,sr-hu,mk-sr,sr-mk" --max-len-b 512 \
    --source-lang $src --target-lang $tgt \
    --buffer-size 10000 --batch-size $BATCH_SIZE --beam $beam --lenpen $lenpen \
    --remove-bpe=sentencepiece --no-progress-bar --fp16 > $OUTPUT.log

cat $OUTPUT.log | grep -P "^H" | cut -f 3- > $OUTPUT
echo "Successfully saving $INPUT to $OUTPUT..."
