#Bitext
TEXT=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/split-data-bin10/
DATA_BIN=$TEXT/data-bin0:$TEXT/data-bin1:$TEXT/data-bin2:$TEXT/data-bin3:$TEXT/data-bin4:$TEXT/data-bin5:$TEXT/data-bin6:$TEXT/data-bin7:$TEXT/data-bin8:$TEXT/data-bin9

#BT
BT_TEXT=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/bt_data-bin/
BT_DATA_BIN=$BT_TEXT

#Parallel BT
PARALLEL_BT_TEXT=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/parallel_bt1.5_split-data-bin10/
PARALLEL_BT_DATA_BIN=$PARALLEL_BT_TEXT/data-bin0:$PARALLEL_BT_TEXT/data-bin1:$PARALLEL_BT_TEXT/data-bin2:$PARALLEL_BT_TEXT/data-bin3:$PARALLEL_BT_TEXT/data-bin4:$PARALLEL_BT_TEXT/data-bin5:$PARALLEL_BT_TEXT/data-bin6:$PARALLEL_BT_TEXT/data-bin7:$PARALLEL_BT_TEXT/data-bin8:$PARALLEL_BT_TEXT/data-bin9


MODEL=/mnt/input/SharedTask/thunder/small_task1/Filter_v1/model/FT/
PRETRAINED_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/model_state_update_300000.th

#LANGS
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu"

#LANG PAIRS
LANG_PAIRS="en-et,et-en,en-hr,hr-en,en-hu,hu-en,en-mk,mk-en,en-sr,sr-en,et-hr,hr-et,et-hu,hu-et,et-mk,mk-et,et-sr,sr-et,hr-hu,hu-hr,hr-mk,mk-hr,hr-sr,sr-hr,hu-mk,mk-hu,hu-sr,sr-hu,mk-sr,sr-mk"
BT_LANG_PAIRS="en-et,en-hr,en-hu,en-mk,en-sr"
PARALLEL_BT_LANG_PAIRS="et-en,hr-en,hu-en,mk-en,sr-en,et-hr,hr-et,et-hu,hu-et,et-mk,mk-et,et-sr,sr-et,hr-hu,hu-hr,hr-mk,mk-hr,hr-sr,sr-hr,hu-mk,mk-hu,hu-sr,sr-hu,mk-sr,sr-mk"

echo "Start Training ..."
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=3 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT train.py $DATA_BIN \
    --save-dir $MODEL --arch xlmt_decoder_variant_large_from_deltalm_postnorm --pretrained-deltalm-checkpoint $PRETRAINED_MODEL --init-encoder-only --init-decoder-only --variant addffn \
    --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
    --encoder-langtok "tgt" --langtoks '{"main":("tgt",None),"bt":("tgt",None),"parallel_bt":("tgt",None)}' \
    --langs $LANGS --data-param-list-sampling-ratios "{\"main\":0.6,\"bt\":0.2,\"parallel_bt\":0.2}" \
    --lang-pairs $LANG_PAIRS --truncate-source --enable-reservsed-directions-shared-datasets \
    --share-all-embeddings --max-source-positions 512 --max-target-positions 512 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 1e-4 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \
    --max-update 400000 --max-epoch 100 --weight-decay 0.0 --max-tokens 1536 --update-freq 96 \
    --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --ddp-backend=no_c10d \
    --extra-data "{\"bt\":\"${BT_DATA_BIN}\",\"parallel_bt\":\"${PARALLEL_BT_DATA_BIN}\"}" --extra-lang-pairs "{\"bt\":\"${BT_LANG_PAIRS}\",\"parallel_bt\":\"${PARALLEL_BT_LANG_PAIRS}\"}" 2>&1 | tee -a $MODEL/train.log 2>&1 | tee -a $MODEL/train.log

#--reset-optimizer --reset-lr-scheduler --reset-dataloader --reset-meters
#--same-lang-per-batch --enable-lang-ids

#--data-param-list-sampling-ratios "{\"main\":0.6,\"bt\":0.2,\"parallel_bt\":0.2}" \

