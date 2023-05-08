TEXT=/mnt/input/SharedTask/thunder/small_task2/download/data-bin/
MODEL=/mnt/input/SharedTask/thunder/small_task2/download/model/deltalm/lr1e-4-deltalm-postnorm-36L-12L/
PRETRAINED_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/model_state_update_300000.th
echo "Start Training ..."
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT train.py $TEXT \
    --save-dir $MODEL --arch xlmt_decoder_variant_large_from_deltalm_postnorm --pretrained-deltalm-checkpoint $PRETRAINED_MODEL --init-encoder-only --init-decoder-only --variant addffn \
    --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
    --encoder-langtok "tgt" --langtoks '{"main":("tgt",None)}' --langs "af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu" \
    --lang-pairs "en-id,id-en,en-jv,jv-en,en-ms,ms-en,en-ta,ta-en,en-tl,tl-en,id-jv,jv-id,id-ms,ms-id,id-ta,ta-id,id-tl,tl-id,jv-ms,ms-jv,jv-ta,ta-jv,jv-tl,tl-jv,ms-ta,ta-ms,ms-tl,tl-ms,ta-tl,tl-ta" --truncate-source \
    --share-all-embeddings --max-source-positions 512 --max-target-positions 512 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 1e-4 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \
    --max-update 400000 --max-epoch 100 --weight-decay 0.0 --max-tokens 2048 --update-freq 16 --encode-layers 36 \
    --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --ddp-backend=no_c10d 2>&1 | tee -a $MODEL/train.log


#--same-lang-per-batch --enable-lang-ids
