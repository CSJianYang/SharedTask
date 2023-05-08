TEXT=/mnt/input/SharedTask/thunder/large_track/data/split-data-bin/
DATA_BIN=$TEXT/data-bin0:$TEXT/data-bin1:$TEXT/data-bin2:$TEXT/data-bin3:$TEXT/data-bin4:$TEXT/data-bin5:$TEXT/data-bin6:$TEXT/data-bin7:$TEXT/data-bin8:$TEXT/data-bin9:$TEXT/data-bin10:$TEXT/data-bin11:$TEXT/data-bin12:$TEXT/data-bin13:$TEXT/data-bin14:$TEXT/data-bin15:$TEXT/data-bin16:$TEXT/data-bin17:$TEXT/data-bin18:$TEXT/data-bin19
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu"
LANG_PAIRS="af-en,en-af,af-lb,lb-af,am-en,en-am,ar-en,en-ar,ar-kk,kk-ar,ar-lb,lb-ar,as-da,da-as,as-de,de-as,as-en,en-as,as-fr,fr-as,as-hi,hi-as,as-hu,hu-as,as-it,it-as,as-ja,ja-as,as-tr,tr-as,ast-de,de-ast,ast-en,en-ast,ast-es,es-ast,ast-fr,fr-ast,ast-ja,ja-ast,ast-nl,nl-ast,ast-pt,pt-ast,ast-ru,ru-ast,az-bg,bg-az,az-de,de-az,az-en,en-az,az-es,es-az,az-fr,fr-az,az-it,it-az,az-ja,ja-az,az-ko,ko-az,az-lt,lt-az,az-lv,lv-az,az-pt,pt-az,az-ru,ru-az,az-tr,tr-az,az-zh,zh-az,be-en,en-be,bg-en,en-bg,bn-en,en-bn,bs-en,en-bs,ca-en,en-ca,ceb-en,en-ceb,cs-en,en-cs,cs-lb,lb-cs,cy-en,en-cy,da-en,en-da,da-lb,lb-da,de-en,en-de,de-hy,hy-de,de-jv,jv-de,de-kk,kk-de,de-km,km-de,de-ky,ky-de,de-lb,lb-de,de-mn,mn-de,de-oc,oc-de,de-tg,tg-de,el-en,en-el,en-su,su-en,en-zh,zh-en,es-en,en-es,es-wo,wo-es,et-en,en-et,et-hr,hr-et,et-hu,hu-et,et-kk,kk-et,et-mk,mk-et,et-sr,sr-et,ff-en,en-ff,ff-es,es-ff,ff-it,it-ff,fi-en,en-fi,fi-km,km-fi,fi-lb,lb-fi,fi-oc,oc-fi,fr-en,en-fr,fr-ff,ff-fr,fr-hy,hy-fr,fr-kk,kk-fr,fr-km,km-fr,fr-lb,lb-fr,fr-ln,ln-fr,fr-lo,lo-fr,fr-mn,mn-fr,fr-oc,oc-fr,fr-sn,sn-fr,fr-so,so-fr,fr-tg,tg-fr,fr-wo,wo-fr,ga-en,en-ga,gl-en,en-gl,gu-en,en-gu,gu-es,es-gu,ha-en,en-ha,he-en,en-he,hi-en,en-hi,hr-en,en-hr,hr-hu,hu-hr,hr-mk,mk-hr,hr-sr,sr-hr,hu-en,en-hu,hu-lb,lb-hu,hu-mk,mk-hu,hu-sr,sr-hu,hy-en,en-hy,hy-es,es-hy,hy-ja,ja-hy,hy-zh,zh-hy,id-en,en-id,id-jv,jv-id,id-ms,ms-id,id-su,su-id,id-ta,ta-id,id-tl,tl-id,ig-en,en-ig,is-en,en-is,it-en,en-it,it-kk,kk-it,it-lb,lb-it,it-oc,oc-it,ja-en,en-ja,ja-km,km-ja,ja-ky,ky-ja,ja-lo,lo-ja,ja-mn,mn-ja,ja-oc,oc-ja,ja-tg,tg-ja,ja-zh,zh-ja,jv-en,en-jv,jv-es,es-jv,jv-ms,ms-jv,jv-su,su-jv,jv-ta,ta-jv,jv-tl,tl-jv,ka-en,en-ka,kam-en,en-kam,kk-en,en-kk,kk-es,es-kk,kk-lt,lt-kk,kk-lv,lv-kk,kk-ms,ms-kk,kk-pl,pl-kk,kk-ru,ru-kk,kk-tr,tr-kk,kk-uz,uz-kk,kk-zh,zh-kk,km-en,en-km,km-es,es-km,km-ms,ms-km,km-ru,ru-km,km-vi,vi-km,km-zh,zh-km,kn-en,en-kn,ko-en,en-ko,ko-mn,mn-ko,ko-zh,zh-ko,ku-en,en-ku,ky-en,en-ky,ky-lt,lt-ky,ky-lv,lv-ky,ky-ru,ru-ky,ky-tr,tr-ky,lb-en,en-lb,lb-es,es-lb,lb-nl,nl-lb,lb-no,no-lb,lb-pt,pt-lb,lb-ru,ru-lb,lb-sv,sv-lb,lb-zh,zh-lb,lg-en,en-lg,ln-en,en-ln,ln-es,es-ln,ln-zh,zh-ln,lo-en,en-lo,lo-zh,zh-lo,lt-en,en-lt,lv-en,en-lv,mi-en,en-mi,mk-en,en-mk,mk-sr,sr-mk,ml-en,en-ml,mn-en,en-mn,mn-zh,zh-mn,mr-en,en-mr,ms-en,en-ms,ms-su,su-ms,ms-ta,ta-ms,ms-tl,tl-ms,mt-en,en-mt,my-en,en-my,ne-en,en-ne,nl-en,en-nl,nl-oc,oc-nl,no-en,en-no,ns-en,en-ns,ny-en,en-ny,oc-en,en-oc,oc-es,es-oc,oc-pl,pl-oc,oc-ru,ru-oc,oc-tr,tr-oc,oc-zh,zh-oc,om-en,en-om,or-en,en-or,or-ru,ru-or,pa-en,en-pa,pl-en,en-pl,ps-en,en-ps,pt-en,en-pt,ro-en,en-ro,ru-en,en-ru,ru-zh,zh-ru,sd-en,en-sd,sk-en,en-sk,sl-en,en-sl,sn-en,en-sn,so-en,en-so,so-tr,tr-so,sr-en,en-sr,su-ta,ta-su,su-tl,tl-su,sv-en,en-sv,sw-en,en-sw,ta-en,en-ta,ta-tl,tl-ta,te-en,en-te,tg-en,en-tg,tg-zh,zh-tg,th-en,en-th,th-zh,zh-th,tl-en,en-tl,tr-en,en-tr,uk-en,en-uk,umb-en,en-umb,ur-en,en-ur,uz-en,en-uz,vi-en,en-vi,vi-zh,zh-vi,wo-en,en-wo,xh-en,en-xh,yo-en,en-yo,zt-zh,zh-zt,zu-en,en-zu"
MODEL=/mnt/input/SharedTask/thunder/large_track/data/model/deltalm/lr1e-4-deltalm-prenorm/
PRETRAINED_MODEL=/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-prenorm/model_state_update_300000.th
echo "Start Training ..."
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=4 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT train.py $DATA_BIN \
    --save-dir $MODEL --arch xlmt_decoder_variant_large_from_deltalm_prenorm --pretrained-deltalm-checkpoint $PRETRAINED_MODEL --init-encoder-only --init-decoder-only \
    --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
    --encoder-langtok "tgt" --langtoks '{"main":("tgt",None)}' --langs $LANGS --lang-pairs $LANG_PAIRS --truncate-source \
    --share-all-embeddings --max-source-positions 512 --max-target-positions 512 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 1e-4 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \
    --max-update 400000 --max-epoch 100 --weight-decay 0.0 --max-tokens 1024 --update-freq 16 \
    --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --ddp-backend=no_c10d 2>&1 | tee -a $MODEL/train.log


#--same-lang-per-batch --enable-lang-ids