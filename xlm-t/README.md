# Multilingual Machine Translation Systems from Microsoft for WMT21 Shared Task

## Abstract

This report describes Microsoft’s machine translation systems for the WMT21 shared task on large-scale multilingual machine translation. We participated in all three evaluation tracks including Large Track and two Small Tracks where the former one is unconstrained and the latter two are fully constrained. Our
model submissions to the shared task were initialized with DeltaLM, a generic pre-trained multilingual encoder-decoder model, and finetuned correspondingly with the vast collected parallel data and allowed data sources according to track settings, together with applying progressive learning and iterative backtranslation approaches to further improve the
performance. Our final submissions ranked
first on three tracks in terms of the automatic
evaluation metric.

## Models
* **Download Multilingual Pre-trained Models**
  * [Pre-norm Pre-trained Models](https://pan.baidu.com/s/19whE0DgyWFxRqZwQpmQCqg?pwd=qyhp)
  * [Post-norm Pre-trained Models](https://pan.baidu.com/s/1n8ag4aMqwiBEEGpov556iA?pwd=c23c)
* **Download Multilingual Translation Models**
  * [36 encoder layers - 12 decoder layers](https://pan.baidu.com/s/1fLs1F14Fc_Z-2V-H2kITxA?pwd=2u8c)
  * [24 encoder layers - 12 decoder layers](https://pan.baidu.com/s/1xRSm4ww_VvDJKfPBwh39Ig?pwd=f4kp)
  * [24 encoder layers - 6 decoder layers](https://pan.baidu.com/s/1ZDdtlbh-sEydNeqRzPWQag?pwd=y4f0)


## Environment

* Python: >= 3.7
* [PyTorch](http://pytorch.org/): >= 1.5.0
* NVIDIA GPUs and [NCCL](https://github.com/NVIDIA/nccl)
* [Fairseq](https://github.com/pytorch/fairseq): 1.0.0

```bash
cd xlm-t
pip install --editable ./
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```
## Supported Languages
Our multilingual translation model supports 100 languages: af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh (Simplified Chinese),zt (Traditional Chinese),zu

## Multilingual Fine-tuning from the Multilingual Pre-trained Model

* Training with the **Source** and **Target** Teacher:

```bash

#BT
BT_TEXT=/path/to/bt_data-bin/
BT_DATA_BIN=$BT_TEXT/data-bin0:$BT_TEXT/data-bin1

#PARALLEL BT
PARALLEL_BT_TEXT=/path/to/parallel_bt_data-bin/
PARALLEL_BT_DATA_BIN=$PARALLEL_BT_TEXT/data-bin0:$PARALLEL_BT_TEXT/data-bin1
#Bitext
TEXT=/path/to/bt_data-bin/data-bin/
DATA_BIN=$TEXT/data-bin0:$TEXT/data-bin1


#LANGS
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu"

#Pretrained Model
PRETRAINED_MODEL_DIR=/path/to/postnorm_pretrained_models/
PRETRAINED_MODEL=$PRETRAINED_MODEL_DIR/model_state_update_300000.th


#LANG PAIRS
LANG_PAIRS=$PRETRAINED_MODEL_DIR/lang_pairs.txt
BT_LANG_PAIRS=$PRETRAINED_MODEL_DIR/bt_lang_pairs.txt
PARALLEL_BT_LANG_PAIRS=$PRETRAINED_MODEL_DIR/parallel_bt_lang_pairs.txt

#Save Dir
MODEL=/path/to/model/
mkdir -p $MODEL
echo "Start Training ..."
echo "$TEXT"
echo "$BT_TEXT"
echo "$PARALLEL_BT_TEXT"
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=8 --node_rank=$OMPI_COMM_WORLD_RANK --master_addr="$MASTER_ADDR" --master_port=$MASTER_PORT train.py $DATA_BIN \
    --save-dir $MODEL --arch xlmt_decoder_variant_large_from_deltalm_prenorm --pretrained-deltalm-checkpoint $PRETRAINED_MODEL --init-encoder-only --init-decoder-only --variant addffn \
    --task translation_multi_simple_epoch --sampling-method "linear" --sampling-temperature 5.0 --min-sampling-temperature 1.0 --warmup-epoch 5 \
    --encoder-langtok "tgt" --langtoks '{"main":("tgt",None),"bt":("tgt",None),"parallel_bt":("tgt",None)}' --langs $LANGS --lang-pairs $LANG_PAIRS --truncate-source \
    --share-all-embeddings --max-source-positions 512 --max-target-positions 512 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt --lr 1e-4 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --warmup-updates 4000 --enable-reservsed-directions-shared-datasets --virtual-epoch-size 300000000 --data-param-list-sampling-ratios '{"main":0.6,"bt":0.2,"parallel_bt":0.2}' \
    --max-update 400000 --max-epoch 100 --weight-decay 0.0 --max-tokens 1280 --update-freq 64 --encoder-layers 36 \
    --seed 1 --log-format simple --skip-invalid-size-inputs-valid-test --fp16 --ddp-backend=no_c10d --dataset-impl mmap \
    --extra-data "{\"bt\":\"${BT_DATA_BIN}\",\"parallel_bt\":\"${PARALLEL_BT_DATA_BIN}\"}" --extra-lang-pairs "{\"bt\":\"${BT_LANG_PAIRS}\",\"parallel_bt\":\"${PARALLEL_BT_LANG_PAIRS}\"}" 2>&1 | tee -a $MODEL/train.log
```

---

* Inference:
```python
export PYTHONWARNINGS="ignore"
export CUDA_VISIBLE_DEVICES=0
DATA_BIN=/path/to/data-bin/
TEXT=/path/to/devtest/
SPM_MODEL=/path/to/spm.model
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu"
LANG_PAIRS="af-en,en-af,af-lb,lb-af,am-en,en-am,ar-en,en-ar,ar-kk,kk-ar,ar-lb,lb-ar,as-da,da-as,as-de,de-as,as-en,en-as,as-fr,fr-as,as-hi,hi-as,as-hu,hu-as,as-it,it-as,as-ja,ja-as,as-tr,tr-as,ast-de,de-ast,ast-en,en-ast,ast-es,es-ast,ast-fr,fr-ast,ast-ja,ja-ast,ast-nl,nl-ast,ast-pt,pt-ast,ast-ru,ru-ast,az-bg,bg-az,az-de,de-az,az-en,en-az,az-es,es-az,az-fr,fr-az,az-it,it-az,az-ja,ja-az,az-ko,ko-az,az-lt,lt-az,az-lv,lv-az,az-pt,pt-az,az-ru,ru-az,az-tr,tr-az,az-zh,zh-az,be-en,en-be,bg-en,en-bg,bn-en,en-bn,bs-en,en-bs,ca-en,en-ca,ceb-en,en-ceb,cs-en,en-cs,cs-lb,lb-cs,cy-en,en-cy,da-en,en-da,da-lb,lb-da,de-en,en-de,de-hy,hy-de,de-jv,jv-de,de-kk,kk-de,de-km,km-de,de-ky,ky-de,de-lb,lb-de,de-mn,mn-de,de-oc,oc-de,de-tg,tg-de,el-en,en-el,en-zh,zh-en,es-en,en-es,es-wo,wo-es,et-en,en-et,et-hr,hr-et,et-hu,hu-et,et-kk,kk-et,et-mk,mk-et,et-sr,sr-et,ff-en,en-ff,ff-es,es-ff,ff-it,it-ff,fi-en,en-fi,fi-km,km-fi,fi-lb,lb-fi,fi-oc,oc-fi,fr-en,en-fr,fr-ff,ff-fr,fr-hy,hy-fr,fr-kk,kk-fr,fr-km,km-fr,fr-lb,lb-fr,fr-ln,ln-fr,fr-lo,lo-fr,fr-mn,mn-fr,fr-oc,oc-fr,fr-sn,sn-fr,fr-so,so-fr,fr-tg,tg-fr,fr-wo,wo-fr,ga-en,en-ga,gl-en,en-gl,gu-en,en-gu,gu-es,es-gu,ha-en,en-ha,he-en,en-he,hi-en,en-hi,hr-en,en-hr,hr-hu,hu-hr,hr-mk,mk-hr,hr-sr,sr-hr,hu-en,en-hu,hu-lb,lb-hu,hu-mk,mk-hu,hu-sr,sr-hu,hy-en,en-hy,hy-es,es-hy,hy-ja,ja-hy,hy-zh,zh-hy,id-en,en-id,id-jv,jv-id,id-ms,ms-id,id-ta,ta-id,id-tl,tl-id,ig-en,en-ig,is-en,en-is,it-en,en-it,it-kk,kk-it,it-lb,lb-it,it-oc,oc-it,ja-en,en-ja,ja-km,km-ja,ja-ky,ky-ja,ja-lo,lo-ja,ja-mn,mn-ja,ja-oc,oc-ja,ja-tg,tg-ja,ja-zh,zh-ja,jv-en,en-jv,jv-es,es-jv,jv-ms,ms-jv,jv-ta,ta-jv,jv-tl,tl-jv,ka-en,en-ka,kam-en,en-kam,kk-en,en-kk,kk-es,es-kk,kk-lt,lt-kk,kk-lv,lv-kk,kk-ms,ms-kk,kk-pl,pl-kk,kk-ru,ru-kk,kk-tr,tr-kk,kk-uz,uz-kk,kk-zh,zh-kk,km-en,en-km,km-es,es-km,km-ms,ms-km,km-ru,ru-km,km-vi,vi-km,km-zh,zh-km,kn-en,en-kn,ko-en,en-ko,ko-mn,mn-ko,ko-zh,zh-ko,ku-en,en-ku,ky-en,en-ky,ky-lt,lt-ky,ky-lv,lv-ky,ky-ru,ru-ky,ky-tr,tr-ky,lb-en,en-lb,lb-es,es-lb,lb-nl,nl-lb,lb-no,no-lb,lb-pt,pt-lb,lb-ru,ru-lb,lb-sv,sv-lb,lb-zh,zh-lb,lg-en,en-lg,ln-en,en-ln,ln-es,es-ln,ln-zh,zh-ln,lo-en,en-lo,lo-zh,zh-lo,lt-en,en-lt,lv-en,en-lv,mi-en,en-mi,mk-en,en-mk,mk-sr,sr-mk,ml-en,en-ml,mn-en,en-mn,mn-zh,zh-mn,mr-en,en-mr,ms-en,en-ms,ms-ta,ta-ms,ms-tl,tl-ms,mt-en,en-mt,my-en,en-my,ne-en,en-ne,nl-en,en-nl,nl-oc,oc-nl,no-en,en-no,ns-en,en-ns,ny-en,en-ny,oc-en,en-oc,oc-es,es-oc,oc-pl,pl-oc,oc-ru,ru-oc,oc-tr,tr-oc,oc-zh,zh-oc,om-en,en-om,or-en,en-or,or-ru,ru-or,pa-en,en-pa,pl-en,en-pl,ps-en,en-ps,pt-en,en-pt,ro-en,en-ro,ru-en,en-ru,ru-zh,zh-ru,sd-en,en-sd,sk-en,en-sk,sl-en,en-sl,sn-en,en-sn,so-en,en-so,so-tr,tr-so,sr-en,en-sr,sv-en,en-sv,sw-en,en-sw,ta-en,en-ta,ta-tl,tl-ta,te-en,en-te,tg-en,en-tg,tg-zh,zh-tg,th-en,en-th,th-zh,zh-th,tl-en,en-tl,tr-en,en-tr,uk-en,en-uk,umb-en,en-umb,ur-en,en-ur,uz-en,en-uz,vi-en,en-vi,vi-zh,zh-vi,wo-en,en-wo,xh-en,en-xh,yo-en,en-yo,zt-zh,zh-zt,zu-en,en-zu"
src=$1
tgt=$2
batchsize=$3
beam=$4
MODEL=$5


lenpen=1.0
INPUT=$TEXT/valid.${src}
FTGT=$TEXT/valid.${tgt}


echo "$INPUT | Batchsize: $batchsize | Beam: $beam | $MODEL"
FOUT=$INPUT.2${tgt}
cat $INPUT | python ./fairseq_cli/interactive.py $DATA_BIN \
    --path $MODEL \
    --encoder-langtok "tgt" --langtoks '{"main":("tgt",None)}' \
    --task translation_multi_simple_epoch \
    --langs $LANGS \
    --lang-pairs $LANG_PAIRS \
    --source-lang $src --target-lang $tgt --truncate-source \
    --buffer-size 10000 --batch-size $batchsize --beam $beam --lenpen $lenpen \
    --bpe sentencepiece --sentencepiece-model $SPM_MODEL --no-progress-bar --fp16 > $FOUT
cat $FOUT | grep -P "^D" | cut -f 3- > $FOUT.out

BLEU_DIR=/mnt/input/SharedTask/thunder/flores101_dataset/BLEU
mkdir -p $BLEU_DIR
evaluation_tok="spm"
echo "Saving BLEU to $BLEU_DIR/${src}-${tgt}.BLEU..."
echo "$MODEL" | tee -a $BLEU_DIR/${src}-${tgt}.BLEU
if [ $evaluation_tok == "spm" ]; then
    cat $FOUT.out | sacrebleu -tok spm $FTGT | tee -a $BLEU_DIR/${src}-${tgt}.BLEU
else
    cat $FOUT.out | sacrebleu -l $src-$tgt $FTGT | tee -a $BLEU_DIR/${src}-${tgt}.BLEU
fi
```


## Inference & Evaluation

* **Beam Search**: (during the inference) beam size = 5; length penalty = 1.0.
* **Metrics**: the case-sensitive detokenized BLEU using sacreBLEU:
  * BLEU+case.mixed+lang.{src}-{tgt}+numrefs.1+smooth.exp+tok.13a+version.1.3.1



## Citation

* arXiv: https://arxiv.org/pdf/2111.02086.pdf
* WMT@EMNLP Anthology: https://aclanthology.org/2021.wmt-1.54.pdf

```bibtex
@inproceedings{microsoft_wmt2021,
  author       = {Jian Yang and
                  Shuming Ma and
                  Haoyang Huang and
                  Dongdong Zhang and
                  Li Dong and
                  Shaohan Huang and
                  Alexandre Muzio and
                  Saksham Singhal and
                  Hany Hassan and
                  Xia Song and
                  Furu Wei},
  editor       = {Lo{\"{\i}}c Barrault and
                  Ondrej Bojar and
                  Fethi Bougares and
                  Rajen Chatterjee and
                  Marta R. Costa{-}juss{\`{a}} and
                  Christian Federmann and
                  Mark Fishel and
                  Alexander Fraser and
                  Markus Freitag and
                  Yvette Graham and
                  Roman Grundkiewicz and
                  Paco Guzman and
                  Barry Haddow and
                  Matthias Huck and
                  Antonio Jimeno{-}Yepes and
                  Philipp Koehn and
                  Tom Kocmi and
                  Andr{\'{e}} Martins and
                  Makoto Morishita and
                  Christof Monz},
  title        = {Multilingual Machine Translation Systems from Microsoft for {WMT21}
                  Shared Task},
  booktitle    = {Proceedings of the Sixth Conference on Machine Translation, WMT@EMNLP
                  2021, Online Event, November 10-11, 2021},
  pages        = {446--455},
  publisher    = {Association for Computational Linguistics},
  year         = {2021},
  url          = {https://aclanthology.org/2021.wmt-1.54},
  timestamp    = {Wed, 19 Jan 2022 17:10:33 +0100},
  biburl       = {https://dblp.org/rec/conf/wmt/YangMH00HMSHSW21.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```


## License

Please refer to the [LICENSE](./LICENSE) file for more details.


