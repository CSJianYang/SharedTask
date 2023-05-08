TEXT=/mnt/input/SharedTask/large-scale/small_task1/download/data-bin/
MODEL=/mnt/input/SharedTask/large-scale/small_task1/download/model/M2M100-lr3e-4/
PRETRAINED_MODEL=/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_615M/flores101_mm100_615M/model.pt
mkdir -p $MODEL
if [ ! -f $MODEL/checkpoint_last.pt ]; then
    echo "Copying $PRETRAINED_MODEL -> $MODEL/checkpoint_last.pt..."
    cp $PRETRAINED_MODEL $MODEL/checkpoint_last.pt
fi
echo "Start Training ..."
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT train.py $TEXT \
    --save-dir $MODEL --arch transformer_wmt_en_de_big --encoder-normalize-before --decoder-normalize-before --encoder-layers 12 --decoder-layers 12  \
    --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
    --encoder-langtok "src" --langtoks '{"main":("src","tgt")}' --langs "af,am,ar,as,ast,ay,az,ba,be,bg,bn,br,bs,ca,ceb,cjk,cs,cy,da,de,dyu,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kac,kam,kea,kg,kk,km,kmb,kmr,kn,ko,ku,ky,lb,lg,ln,lo,lt,luo,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,no,ns,ny,oc,om,or,pa,pl,ps,pt,qu,ro,ru,sd,shn,si,sk,sl,sn,so,sq,sr,ss,su,sv,sw,ta,te,tg,th,ti,tl,tn,tr,uk,umb,ur,uz,vi,wo,xh,yi,yo,zh,zu" \
    --lang-pairs "en-et,et-en,en-hr,hr-en,en-hu,hu-en,en-mk,mk-en,en-sr,sr-en,et-hr,hr-et,et-hu,hu-et,et-mk,mk-et,et-sr,sr-et,hr-hu,hu-hr,hr-mk,mk-hr,hr-sr,sr-hr,hu-mk,mk-hu,hu-sr,sr-hu,mk-sr,sr-mk" --truncate-source \
    --share-all-embeddings --max-source-positions 1024 --max-target-positions 1024 --criterion label_smoothed_cross_entropy_with_sparse --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 3e-4 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \
    --max-update 400000 --max-epoch 100 --weight-decay 0.0 --max-tokens 1536 --update-freq 10 \
    --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --virtual-epoch-size 100000000 \
    --reset-optimizer --reset-meters --reset-lr-scheduler --reset-dataloader --ddp-backend=no_c10d 2>&1 | tee -a $MODEL/train.log



#--same-lang-per-batch --enable-lang-ids