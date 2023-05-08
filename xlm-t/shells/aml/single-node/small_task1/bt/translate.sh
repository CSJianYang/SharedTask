export CUDA_VISIBLE_DEVICES=0
DATA_BIN=/mnt/input/SharedTask/large-scale/small_task1/download/data-bin/
INPUT_DIR=/mnt/input/SharedTask/large-scale/MonolingualData/all_spm_split/
OUTPUT_DIR=/mnt/input/SharedTask/large-scale/MonolingualData/all_spm_split_bt/
SPM_MODEL=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_615M/flores101_mm100_615M/sentencepiece.bpe.model
LANG_PAIRS=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_175M/flores101_mm100_175M/language_pairs.txt
mkdir -p $OUTPUT_DIR
src=$1
tgt=$2
beam=$3
MODEL=$4
INPUT_FILE=$5
OUTPUT_FIEL=$6
INPUT=$INPUT_DIR/$INPUT_FILE
OUTPUT=$OUTPUT_DIR/$OUTPUT_FILE

lenpen=1.0
echo $MODEL
FOUT=$OUTPUT_DIR/$OUTPUT_FILE.2$tgt
cat $INPUT | python ./fairseq_cli/interactive.py $DATA_BIN \
    --path $MODEL \
    --encoder-langtok "src" --decoder-langtok --langtoks '{"main":("src", "tgt")}' \
    --task translation_multi_simple_epoch \
    --langs "af,am,ar,as,ast,ay,az,ba,be,bg,bn,br,bs,ca,ceb,cjk,cs,cy,da,de,dyu,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kac,kam,kea,kg,kk,km,kmb,kmr,kn,ko,ku,ky,lb,lg,ln,lo,lt,luo,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,no,ns,ny,oc,om,or,pa,pl,ps,pt,qu,ro,ru,sd,shn,si,sk,sl,sn,so,sq,sr,ss,su,sv,sw,ta,te,tg,th,ti,tl,tn,tr,uk,umb,ur,uz,vi,wo,xh,yi,yo,zh,zu" \
    --lang-pairs $LANG_PAIRS \
    --source-lang $src --target-lang $tgt \
    --buffer-size 10000 --batch-size 32 --beam $beam --lenpen $lenpen \
    --no-progress-bar --fp16 > $FOUT

cat $FOUT | grep -P "^H" | cut -f 3- > $FOUT.out
