TEXT=/mnt/input/SharedTask/thunder/small_task1/download/data-bin/
MODEL=/mnt/input/SharedTask/thunder/small_task1/download/model/lr1e-4-deltalm-prenorm-tgt/
PRETRAINED_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-prenorm/model_state_update_300000.th
echo "Start Training ..."
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT train.py $TEXT \
    --save-dir $MODEL --arch xlmt_decoder_variant_large_from_deltalm_prenorm --pretrained-deltalm-checkpoint $PRETRAINED_MODEL --init-encoder-only --init-decoder-only \
    --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
    --encoder-langtok "tgt" --langtoks '{"main":("tgt",None)}' --langs "af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu" \
    --lang-pairs "en-et,et-en,en-hr,hr-en,en-hu,hu-en,en-mk,mk-en,en-sr,sr-en,et-hr,hr-et,et-hu,hu-et,et-mk,mk-et,et-sr,sr-et,hr-hu,hu-hr,hr-mk,mk-hr,hr-sr,sr-hr,hu-mk,mk-hu,hu-sr,sr-hu,mk-sr,sr-mk" --truncate-source \
    --share-all-embeddings --max-source-positions 512 --max-target-positions 512 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 1e-4 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \
    --max-update 400000 --max-epoch 100 --weight-decay 0.0 --max-tokens 2048 --update-freq 8 \
    --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --virtual-epoch-size 100000000 \
    --reset-optimizer --reset-meters --reset-lr-scheduler --reset-dataloader --ddp-backend=no_c10d 2>&1 | tee -a $MODEL/train.log


#--same-lang-per-batch --enable-lang-ids

#af,am,ar,as,ast,ay,az,ba,be,bg,bn,br,bs,ca,ceb,cjk,cs,cy,da,de,dyu,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kac,kam,kea,kg,kk,km,kmb,kmr,kn,ko,ku,ky,lb,lg,ln,lo,lt,luo,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,no,ns,ny,oc,om,or,pa,pl,ps,pt,qu,ro,ru,sd,shn,si,sk,sl,sn,so,sq,sr,ss,su,sv,sw,ta,te,tg,th,ti,tl,tn,tr,uk,umb,ur,uz,vi,wo,xh,yi,yo,zh,zu

#af,am,ar,as,ast,ay,az,ba,be,bg,bn,br,bs,ca,ceb,cjk,cs,cy,da,de,dyu,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kac,kam,kea,kg,kk,km,kmb,kmr,kn,ko,ku,ky,lb,lg,ln,lo,lt,luo,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,no,ns,ny,oc,om,or,pa,pl,ps,pt,qu,ro,ru,sd,shn,si,sk,sl,sn,so,sq,sr,ss,su,sv,sw,ta,te,tg,th,ti,tl,tn,tr,uk,umb,ur,uz,vi,wo,xh,yi,yo,zh,zu


#af am ar as ast az be bn bs bg ca ceb cs ku cy da de el en et fa fi fr ff ga gl gu ha he hi hr hu hy ig id is it jv ja kam kn ka kk kea km ky ko lo lv ln lt lb lg luo ml mr mk mt mn mi ms my nl no ne ns ny oc om or pa pl pt ps ro ru sk sl sn sd so es sr sv sw ta te tg tl th tr uk umb ur uz vi wo xh yo zh zt zu