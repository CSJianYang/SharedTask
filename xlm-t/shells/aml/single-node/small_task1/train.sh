TEXT=/mnt/input/SharedTask/large-scale/small_task1/download/data-bin/
MODEL=/mnt/input/SharedTask/large-scale/small_task1/download/model/M2M100/
PRETRAINED_ENCODER_MODEL=/mnt/input/Z-code/data/zcode20/premodels/xlmr.base/model.pt
mkdir -p $MODEL
echo "Start Training ..."
python train.py $TEXT \
    --save-dir $MODEL --arch transformer_vaswani_wmt_en_de_big --task translation_multi_simple_epoch \
    --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
    --encoder-langtok "src" --langtoks '{"main":("src","tgt")}' --langs "af,am,ar,as,ast,ay,az,ba,be,bg,bn,br,bs,ca,ceb,cjk,cs,cy,da,de,dyu,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kac,kam,kea,kg,kk,km,kmb,kmr,kn,ko,ku,ky,lb,lg,ln,lo,lt,luo,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,no,ns,ny,oc,om,or,pa,pl,ps,pt,qu,ro,ru,sd,shn,si,sk,sl,sn,so,sq,sr,ss,su,sv,sw,ta,te,tg,th,ti,tl,tn,tr,uk,umb,ur,uz,vi,wo,xh,yi,yo,zh,zu" \
    --lang-pairs "en-et,en-hr,en-hu,en-mk,en-sr,et-hr,et-hu,et-mk,et-sr,hr-hu,hr-mk,hr-sr,hu-mk,hu-sr,mk-sr" --truncate-source --ddp-backend=no_c10d \
    --share-all-embeddings --max-source-positions 512 --max-target-positions 512 --criterion label_smoothed_cross_entropy_with_sparse --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 1e-4 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 \
    --max-update 400000 --max-epoch 100 --dropout 0.1 --attention-dropout 0.0 --weight-decay 0.0 --max-tokens 4096 --update-freq 16 \
    --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --same-lang-per-batch --enable-lang-ids \
    --reset-optimizer --reset-meters --reset-lr-schedule --reset-dataloader 2>&1 | tee -a $MODEL/train.log
