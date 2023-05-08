#Bitext
TEXT=/mnt/input/SharedTask/thunder/small_task2/Filter_v1/data-bin/
DATA_BIN=$TEXT

#Bitext
BT_TEXT=/mnt/input/SharedTask/thunder/small_task2/Filter_v1/bt_data-bin/
BT_DATA_BIN=$BT_TEXT

#Parallel BT
PARALLEL_BT_TEXT=/mnt/input/SharedTask/thunder/small_task2/Filter_v1/parallel_bt1.5_split-data-bin10/
PARALLEL_BT_DATA_BIN=$PARALLEL_BT_TEXT/data-bin0:$PARALLEL_BT_TEXT/data-bin1:$PARALLEL_BT_TEXT/data-bin2:$PARALLEL_BT_TEXT/data-bin3:$PARALLEL_BT_TEXT/data-bin4:$PARALLEL_BT_TEXT/data-bin5:$PARALLEL_BT_TEXT/data-bin6:$PARALLEL_BT_TEXT/data-bin7:$PARALLEL_BT_TEXT/data-bin8:$PARALLEL_BT_TEXT/data-bin9


MODEL=/mnt/input/SharedTask/thunder/small_task2/Filter_v1/model/36L-12L-FT-pivot/
PRETRAINED_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/model_state_update_300000.th

#LANGS
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu"

#LANG PAIRS
LANG_PAIRS="en-id,id-en,en-jv,jv-en,en-ms,ms-en,en-ta,ta-en,en-tl,tl-en,id-jv,jv-id,id-ms,ms-id,id-ta,ta-id,id-tl,tl-id,jv-ms,ms-jv,jv-ta,ta-jv,jv-tl,tl-jv,ms-ta,ta-ms,ms-tl,tl-ms,ta-tl,tl-ta"
BT_LANG_PAIRS="en-id,en-jv,en-ms,en-ta,en-tl,id-jv,id-ms,jv-id,jv-ms,ms-jv,tl-id,tl-jv,tl-ms,tl-ta"
PARALLEL_BT_LANG_PAIRS="id-en,jv-en,ms-en,ta-en,tl-en"

echo "Start Training ..."
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT train.py $TEXT \
    --save-dir $MODEL --arch xlmt_decoder_variant_large_from_deltalm_postnorm --pretrained-deltalm-checkpoint $PRETRAINED_MODEL --init-encoder-only --init-decoder-only --variant addffn \
    --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
    --encoder-langtok "tgt" --langtoks '{"main":("tgt",None),"bt":("tgt",None),"parallel_bt":("tgt",None)}' \
    --langs $LANGS \
    --lang-pairs $LANG_PAIRS --truncate-source --enable-reservsed-directions-shared-datasets --data-param-list-sampling-ratios "{\"main\":0.7,\"bt\":0.2,\"parallel_bt\":0.1}" \
    --share-all-embeddings --max-source-positions 512 --max-target-positions 512 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 5e-5 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \
    --max-update 400000 --max-epoch 100 --weight-decay 0.0 --max-tokens 1792 --encoder-layers 36 --update-freq 32 --clip-norm 1.0 \
    --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --ddp-backend=no_c10d \
    --extra-data "{\"bt\":\"${BT_DATA_BIN}\",\"parallel_bt\":\"${PARALLEL_BT_DATA_BIN}\"}" --extra-lang-pairs "{\"bt\":\"${BT_LANG_PAIRS}\",\"parallel_bt\":\"${PARALLEL_BT_LANG_PAIRS}\"}" 2>&1 | tee -a $MODEL/train.log 2>&1 | tee -a $MODEL/train.log

# --reset-optimizer --reset-dataloader --reset-meters --reset-lr-scheduler
#--same-lang-per-batch --enable-lang-ids

#--data-param-list-sampling-ratios
#"{\"main\":0.5,\"bt\":0.4,\"parallel_bt\":0.1}"
