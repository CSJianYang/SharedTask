# Copyright (c) Facebook, Inc. and its affiliates.

"""
Instructions:
Please work through this file to construct your handler. Here are things
to watch out for:
- TODO blocks: you need to fill or modify these according to the instructions.
   The code in these blocks are for demo purpose only and they may not work.
- NOTE inline comments: remember to follow these instructions to pass the test.
For expected task I/O, please check dynalab/tasks/README.md
"""
from typing import NamedTuple
import json
import os
import sys
from omegaconf import DictConfig
import torch
from omegaconf import OmegaConf
from dynalab.handler.base_handler import BaseDynaHandler,ROOTPATH
#NOTE: use the following line to import modules from your repo
#sys.path.append(ROOTPATH)
sys.path.insert(0, ROOTPATH)
from dynalab.tasks.flores_small1 import TaskIO
from dynalab.tasks import flores_small1

####################################
import logging
import math
import os
import sys
import time
from argparse import Namespace
import torch
from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders
from fairseq_cli.generate import get_symbols_to_strip_from_output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
############################################################################
#LANG_CODE={'afr':'af', 'amh':'am', 'ara':'ar', 'asm':'as', 'ast':'ast', 'azj':'az', 'bel':'be', 'ben':'bn', 'bos':'bs', 'bul':'bg', 'cat':'ca', 'ceb':'ceb', 'ces':'cs', 'ckb':'ku', 'cym':'cy', 'dan':'da', 'deu':'de', 'ell':'el', 'eng':'en', 'est': 'et', 'fas':'fa', 'fin':'fi', 'fra':'fr', 'ful':'ff', 'gle':'ga', 'glg':'gl', 'guj':'gu', 'hau':'ha', 'heb':'he', 'hin':'hi', 'hrv':'hr', 'hun':'hu','hye':'hy','ibo':'ig','ind':'id','isl':'is','ita':'it','jav':'jv', 'jpn':'ja', 'kam':'kam', 'kan':'kn', 'kat':'ka', 'kaz':'kk', 'kea':'kea', 'khm':'km', 'kir':'ky', 'kor':'ko', 'lao':'lo', 'lav':'lv', 'lin':'ln', 'lit':'lt', 'ltz':'lb', 'lug':'lg', 'luo':'luo', 'mal':'ml', 'mar':'mr', 'mkd':'mk', 'mlt':'mt', 'mon':'mn', 'mri':'mi', 'msa':'ms','mya':'my','nld':'nl','nob':'no','npi':'ne','nso':'ns','nya':'ny','oci':'oc','orm':'om', 'ory':'or', 'pan':'pa', 'pol':'pl', 'por':'pt', 'pus':'ps', 'ron':'ro', 'rus':'ru', 'slk':'sk', 'slv':'sl', 'sna':'sn', 'snd':'sd', 'som':'so', 'spa':'es', 'srp':'sr', 'swe':'sv', 'swh':'sw', 'tam':'ta', 'tel':'te', 'tgk':'tg', 'tgl':'tl', 'tha':'th', 'tur':'tr', 'ukr':'uk', 'umb':'umb', 'urd':'ur', 'uzb':'uz', 'vie':'vi', 'wol':'wo', 'xho':'xh', 'yor':'yo', 'zho_simp':'zh', 'zho_trad':'zh', 'zul':'zu'}
LANG_CODE={'afr':'af', 'amh':'am', 'ara':'ar', 'asm':'as', 'ast':'ast', 'azj':'az', 'bel':'be', 'ben':'bn', 'bos':'bs', 'bul':'bg', 'cat':'ca', 'ceb':'ceb', 'ces':'cs', 'ckb':'ku', 'cym':'cy', 'dan':'da', 'deu':'de', 'ell':'el', 'eng':'en', 'est': 'et', 'fas':'fa', 'fin':'fi', 'fra':'fr', 'ful':'ff', 'gle':'ga', 'glg':'gl', 'guj':'gu', 'hau':'ha', 'heb':'he', 'hin':'hi', 'hrv':'hr', 'hun':'hu','hye':'hy','ibo':'ig','ind':'id','isl':'is','ita':'it','jav':'jv', 'jpn':'ja', 'kam':'kam', 'kan':'kn', 'kat':'ka', 'kaz':'kk', 'kea':'kea', 'khm':'km', 'kir':'ky', 'kor':'ko', 'lao':'lo', 'lav':'lv', 'lin':'ln', 'lit':'lt', 'ltz':'lb', 'lug':'lg', 'luo':'luo', 'mal':'ml', 'mar':'mr', 'mkd':'mk', 'mlt':'mt', 'mon':'mn', 'mri':'mi', 'msa':'ms','mya':'my','nld':'nl','nob':'no','npi':'ne','nso':'ns','nya':'ny','oci':'oc','orm':'om', 'ory':'or', 'pan':'pa', 'pol':'pl', 'por':'pt', 'pus':'ps', 'ron':'ro', 'rus':'ru', 'slk':'sk', 'slv':'sl', 'sna':'sn', 'snd':'sd', 'som':'so', 'spa':'es', 'srp':'sr', 'swe':'sv', 'swh':'sw', 'tam':'ta', 'tel':'te', 'tgk':'tg', 'tgl':'tl', 'tha':'th', 'tur':'tr', 'ukr':'uk', 'umb':'umb', 'urd':'ur', 'uzb':'uz', 'vie':'vi', 'wol':'wo', 'xho':'xh', 'yor':'yo', 'zho_simp':'zh', 'zho_trad':'zt', 'zul':'zu'}

def encode_fn(x, bpe, tokenizer):
    if tokenizer is not None:
        x = tokenizer.encode(x)
    if bpe is not None:
        x = bpe.encode(x)
    return x


def decode_fn(x, bpe, tokenizer):
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x


class Handler(BaseDynaHandler):
    def initialize(self, context):
        """
        load model and extra files
        """
        self.debug = True
        #context._system_properties['gpu_id'] = 0
        #print(context._system_properties)
        model_pt_path, model_file_dir, device_str = self._handler_initialize(context)
        self.taskIO = TaskIO()
        self.cfg = Namespace()
        self.cfg.common = OmegaConf.create({'_name': None, 'no_progress_bar': True, 'log_interval': 100, 'log_format': None, 'tensorboard_logdir': None, 'wandb_project': None, 'seed': 1, 'cpu': False, 'tpu': False, 'bf16': False, 'memory_efficient_bf16': False, 'fp16': False, 'memory_efficient_fp16': False, 'fp16_no_flatten_grads': False, 'fp16_init_scale': 128, 'fp16_scale_window': None, 'fp16_scale_tolerance': 0.0, 'min_loss_scale': 0.0001, 'threshold_loss_scale': None, 'user_dir': None, 'empty_cache_freq': 0, 'all_gather_list_size': 16384, 'model_parallel_size': 1, 'quantization_config_path': None, 'profile': False, 'reset_logging': True})
        self.cfg.task = Namespace(_name='translation_multi_simple_epoch', all_gather_list_size=16384, batch_size=1, batch_size_valid=1, beam=5, best_checkpoint_metric='loss', bf16=False, bpe=None, broadcast_buffers=False, bucket_cap_mb=25, buffer_size=1, checkpoint_shard_count=1, checkpoint_suffix='', constraints=None, cpu=False, criterion='cross_entropy', curriculum=0, data="dict.txt", data_buffer_size=10, dataset_impl=None, ddp_backend='c10d', debug=False, decoder_langtok=True, decoding_format=None, device_id=0, disable_validation=False, distributed_backend='nccl', distributed_init_method=None, distributed_no_spawn=False, distributed_port=-1, distributed_rank=0, distributed_world_size=1, distributed_wrapper='DDP', diverse_beam_groups=-1, diverse_beam_strength=0.5, diversity_rate=-1.0, empty_cache_freq=0, enable_lang_ids=False, enable_reservsed_directions_shared_datasets=False, encoder_langtok='src', eos=2, extra_data=None, extra_lang_pairs=None, fast_stat_sync=False, find_unused_parameters=True, finetune_from_model=None, fix_batches_to_gpus=False, fixed_dictionary="dict.txt", fixed_validation_seed=None, force_anneal=None, fp16=False, fp16_init_scale=128, fp16_no_flatten_grads=False, fp16_scale_tolerance=0.0, fp16_scale_window=None, gen_subset='test', input=None, iter_decode_eos_penalty=0.0, iter_decode_force_max_iter=False, iter_decode_max_iter=10, iter_decode_with_beam=1, iter_decode_with_external_reranker=False, keep_best_checkpoints=-1, keep_inference_langtok=False, keep_interval_updates=-1, keep_last_epochs=-1, lang_dict=None, lang_pairs='af-bs,bs-af,af-da,da-af,af-de,de-af,af-en,en-af,af-fy,fy-af,af-is,is-af,af-lb,lb-af,af-mn,mn-af,af-nl,nl-af,af-no,no-af,af-pa,pa-af,af-ps,ps-af,af-sv,sv-af,af-tr,tr-af,af-yi,yi-af,am-en,en-am,am-ff,ff-am,am-ha,ha-am,am-ig,ig-am,am-lg,lg-am,am-ln,ln-am,am-ns,ns-am,am-ss,ss-am,am-tn,tn-am,am-wo,wo-am,am-xh,xh-am,am-yo,yo-am,am-zu,zu-am,ar-az,az-ar,ar-ba,ba-ar,ar-bg,bg-ar,ar-bn,bn-ar,ar-bs,bs-ar,ar-cs,cs-ar,ar-da,da-ar,ar-de,de-ar,ar-el,el-ar,ar-en,en-ar,ar-es,es-ar,ar-fa,fa-ar,ar-fi,fi-ar,ar-fr,fr-ar,ar-gu,gu-ar,ar-he,he-ar,ar-hi,hi-ar,ar-hu,hu-ar,ar-id,id-ar,ar-it,it-ar,ar-ja,ja-ar,ar-kk,kk-ar,ar-ko,ko-ar,ar-lt,lt-ar,ar-ml,ml-ar,ar-mn,mn-ar,ar-mr,mr-ar,ar-ms,ms-ar,ar-ne,ne-ar,ar-nl,nl-ar,ar-no,no-ar,ar-pa,pa-ar,ar-pl,pl-ar,ar-ps,ps-ar,ar-pt,pt-ar,ar-ro,ro-ar,ar-ru,ru-ar,ar-si,si-ar,ar-sv,sv-ar,ar-sw,sw-ar,ar-ta,ta-ar,ar-tl,tl-ar,ar-tr,tr-ar,ar-uk,uk-ar,ar-ur,ur-ar,ar-uz,uz-ar,ar-vi,vi-ar,ar-zh,zh-ar,as-en,en-as,ast-en,en-ast,ast-es,es-ast,ay-en,en-ay,az-bs,bs-az,az-en,en-az,az-id,id-az,az-kk,kk-az,az-mn,mn-az,az-pa,pa-az,az-ps,ps-az,az-tr,tr-az,az-vi,vi-az,ba-fa,fa-ba,be-bg,bg-be,be-bs,bs-be,be-cs,cs-be,be-en,en-be,be-hr,hr-be,be-lv,lv-be,be-mn,mn-be,be-nl,nl-be,be-pa,pa-be,be-pl,pl-be,be-ps,ps-be,be-ru,ru-be,be-sk,sk-be,be-sl,sl-be,be-sr,sr-be,be-sv,sv-be,be-sw,sw-be,be-tr,tr-be,be-uk,uk-be,be-vi,vi-be,bg-bs,bs-bg,bg-ca,ca-bg,bg-cs,cs-bg,bg-da,da-bg,bg-de,de-bg,bg-el,el-bg,bg-en,en-bg,bg-es,es-bg,bg-fi,fi-bg,bg-fr,fr-bg,bg-hr,hr-bg,bg-hu,hu-bg,bg-id,id-bg,bg-it,it-bg,bg-ja,ja-bg,bg-ko,ko-bg,bg-lv,lv-bg,bg-mn,mn-bg,bg-nl,nl-bg,bg-no,no-bg,bg-pa,pa-bg,bg-pl,pl-bg,bg-ps,ps-bg,bg-pt,pt-bg,bg-ro,ro-bg,bg-ru,ru-bg,bg-sk,sk-bg,bg-sl,sl-bg,bg-sr,sr-bg,bg-sv,sv-bg,bg-sw,sw-bg,bg-tr,tr-bg,bg-uk,uk-bg,bg-vi,vi-bg,bg-zh,zh-bg,bn-bs,bs-bn,bn-cs,cs-bn,bn-el,el-bn,bn-en,en-bn,bn-es,es-bn,bn-fa,fa-bn,bn-fi,fi-bn,bn-fr,fr-bn,bn-he,he-bn,bn-hi,hi-bn,bn-hu,hu-bn,bn-id,id-bn,bn-ilo,ilo-bn,bn-it,it-bn,bn-ja,ja-bn,bn-ko,ko-bn,bn-lt,lt-bn,bn-mg,mg-bn,bn-mn,mn-bn,bn-mr,mr-bn,bn-ms,ms-bn,bn-ne,ne-bn,bn-nl,nl-bn,bn-or,or-bn,bn-pa,pa-bn,bn-pl,pl-bn,bn-ps,ps-bn,bn-pt,pt-bn,bn-ru,ru-bn,bn-si,si-bn,bn-sv,sv-bn,bn-sw,sw-bn,bn-ta,ta-bn,bn-tl,tl-bn,bn-tr,tr-bn,bn-ur,ur-bn,bn-vi,vi-bn,bn-zh,zh-bn,br-en,en-br,br-nl,nl-br,bs-ca,ca-bs,bs-cs,cs-bs,bs-da,da-bs,bs-de,de-bs,bs-el,el-bs,bs-en,en-bs,bs-es,es-bs,bs-et,et-bs,bs-fa,fa-bs,bs-fi,fi-bs,bs-fr,fr-bs,bs-gu,gu-bs,bs-he,he-bs,bs-hi,hi-bs,bs-hr,hr-bs,bs-hu,hu-bs,bs-id,id-bs,bs-is,is-bs,bs-it,it-bs,bs-ja,ja-bs,bs-kk,kk-bs,bs-ko,ko-bs,bs-lt,lt-bs,bs-lv,lv-bs,bs-mg,mg-bs,bs-mk,mk-bs,bs-ml,ml-bs,bs-mn,mn-bs,bs-mr,mr-bs,bs-ms,ms-bs,bs-ne,ne-bs,bs-nl,nl-bs,bs-no,no-bs,bs-pa,pa-bs,bs-pl,pl-bs,bs-ps,ps-bs,bs-pt,pt-bs,bs-ro,ro-bs,bs-ru,ru-bs,bs-si,si-bs,bs-sk,sk-bs,bs-sl,sl-bs,bs-sr,sr-bs,bs-su,su-bs,bs-sv,sv-bs,bs-sw,sw-bs,bs-ta,ta-bs,bs-tl,tl-bs,bs-tr,tr-bs,bs-uk,uk-bs,bs-ur,ur-bs,bs-vi,vi-bs,bs-zh,zh-bs,ca-da,da-ca,ca-en,en-ca,ca-es,es-ca,ca-fr,fr-ca,ca-gl,gl-ca,ca-it,it-ca,ca-mn,mn-ca,ca-pa,pa-ca,ca-ps,ps-ca,ca-pt,pt-ca,ca-ro,ro-ca,ca-sk,sk-ca,ca-sl,sl-ca,ca-sr,sr-ca,ca-uk,uk-ca,ceb-de,de-ceb,ceb-en,en-ceb,ceb-hi,hi-ceb,ceb-id,id-ceb,ceb-ilo,ilo-ceb,ceb-ja,ja-ceb,ceb-jv,jv-ceb,ceb-mg,mg-ceb,ceb-ml,ml-ceb,ceb-ms,ms-ceb,ceb-su,su-ceb,ceb-tl,tl-ceb,cs-da,da-cs,cs-de,de-cs,cs-el,el-cs,cs-en,en-cs,cs-es,es-cs,cs-fa,fa-cs,cs-fi,fi-cs,cs-fr,fr-cs,cs-he,he-cs,cs-hi,hi-cs,cs-hr,hr-cs,cs-hu,hu-cs,cs-id,id-cs,cs-it,it-cs,cs-ja,ja-cs,cs-ko,ko-cs,cs-lt,lt-cs,cs-mk,mk-cs,cs-mn,mn-cs,cs-nl,nl-cs,cs-no,no-cs,cs-pa,pa-cs,cs-pl,pl-cs,cs-ps,ps-cs,cs-pt,pt-cs,cs-ru,ru-cs,cs-sk,sk-cs,cs-sl,sl-cs,cs-sr,sr-cs,cs-sv,sv-cs,cs-sw,sw-cs,cs-ta,ta-cs,cs-tr,tr-cs,cs-uk,uk-cs,cs-vi,vi-cs,cs-zh,zh-cs,cy-en,en-cy,da-de,de-da,da-el,el-da,da-en,en-da,da-es,es-da,da-fi,fi-da,da-fr,fr-da,da-fy,fy-da,da-gd,gd-da,da-hu,hu-da,da-id,id-da,da-is,is-da,da-it,it-da,da-ja,ja-da,da-ko,ko-da,da-lb,lb-da,da-mn,mn-da,da-nl,nl-da,da-no,no-da,da-pa,pa-da,da-pl,pl-da,da-ps,ps-da,da-pt,pt-da,da-ro,ro-da,da-ru,ru-da,da-sv,sv-da,da-tr,tr-da,da-uk,uk-da,da-vi,vi-da,da-yi,yi-da,da-zh,zh-da,de-el,el-de,de-en,en-de,de-es,es-de,de-et,et-de,de-fa,fa-de,de-fi,fi-de,de-fr,fr-de,de-fy,fy-de,de-he,he-de,de-hu,hu-de,de-id,id-de,de-ilo,ilo-de,de-is,is-de,de-it,it-de,de-ja,ja-de,de-ko,ko-de,de-lb,lb-de,de-lt,lt-de,de-lv,lv-de,de-mg,mg-de,de-mn,mn-de,de-ms,ms-de,de-nl,nl-de,de-no,no-de,de-pa,pa-de,de-pl,pl-de,de-ps,ps-de,de-pt,pt-de,de-ru,ru-de,de-su,su-de,de-sv,sv-de,de-sw,sw-de,de-ta,ta-de,de-tl,tl-de,de-tr,tr-de,de-uk,uk-de,de-vi,vi-de,de-yi,yi-de,de-zh,zh-de,el-en,en-el,el-es,es-el,el-et,et-el,el-fa,fa-el,el-fi,fi-el,el-fr,fr-el,el-he,he-el,el-hi,hi-el,el-hu,hu-el,el-id,id-el,el-it,it-el,el-ja,ja-el,el-ko,ko-el,el-lt,lt-el,el-lv,lv-el,el-mn,mn-el,el-nl,nl-el,el-no,no-el,el-pa,pa-el,el-pl,pl-el,el-ps,ps-el,el-pt,pt-el,el-ro,ro-el,el-ru,ru-el,el-sq,sq-el,el-sv,sv-el,el-sw,sw-el,el-ta,ta-el,el-tr,tr-el,el-uk,uk-el,el-vi,vi-el,el-zh,zh-el,cjk-en,en-cjk,dyu-en,en-dyu,en-es,es-en,en-et,et-en,en-fa,fa-en,en-ff,ff-en,en-fi,fi-en,en-fr,fr-en,en-fy,fy-en,en-ga,ga-en,en-gd,gd-en,en-gl,gl-en,en-gu,gu-en,en-ha,ha-en,en-he,he-en,en-hi,hi-en,en-hr,hr-en,en-ht,ht-en,en-hu,hu-en,en-hy,hy-en,en-id,id-en,en-ig,ig-en,en-ilo,ilo-en,en-is,is-en,en-it,it-en,en-ja,ja-en,en-jv,jv-en,en-ka,ka-en,en-kac,kac-en,en-kam,kam-en,en-kea,kea-en,en-kg,kg-en,en-kk,kk-en,en-km,km-en,en-kmb,kmb-en,en-kmr,kmr-en,en-kn,kn-en,en-ko,ko-en,en-ku,ku-en,en-ky,ky-en,en-lb,lb-en,en-lg,lg-en,en-ln,ln-en,en-lo,lo-en,en-lt,lt-en,en-luo,luo-en,en-lv,lv-en,en-mg,mg-en,en-mi,mi-en,en-mk,mk-en,en-ml,ml-en,en-mn,mn-en,en-mr,mr-en,en-ms,ms-en,en-mt,mt-en,en-my,my-en,en-ne,ne-en,en-nl,nl-en,en-no,no-en,en-ns,ns-en,en-ny,ny-en,en-oc,oc-en,en-om,om-en,en-or,or-en,en-pa,pa-en,en-pl,pl-en,en-ps,ps-en,en-pt,pt-en,en-qu,qu-en,en-ro,ro-en,en-ru,ru-en,en-sd,sd-en,en-shn,shn-en,en-si,si-en,en-sk,sk-en,en-sl,sl-en,en-sn,sn-en,en-so,so-en,en-sq,sq-en,en-sr,sr-en,en-ss,ss-en,en-su,su-en,en-sv,sv-en,en-sw,sw-en,en-ta,ta-en,en-te,te-en,en-tg,tg-en,en-th,th-en,en-ti,ti-en,en-tl,tl-en,en-tn,tn-en,en-tr,tr-en,en-uk,uk-en,en-umb,umb-en,en-ur,ur-en,en-vi,vi-en,en-wo,wo-en,en-xh,xh-en,en-yi,yi-en,en-yo,yo-en,en-zh,zh-en,en-zu,zu-en,es-fa,fa-es,es-fi,fi-es,es-fr,fr-es,es-gl,gl-es,es-he,he-es,es-hi,hi-es,es-hu,hu-es,es-id,id-es,es-it,it-es,es-ja,ja-es,es-ko,ko-es,es-lt,lt-es,es-mn,mn-es,es-nl,nl-es,es-pa,pa-es,es-pl,pl-es,es-ps,ps-es,es-pt,pt-es,es-ro,ro-es,es-ru,ru-es,es-sv,sv-es,es-sw,sw-es,es-ta,ta-es,es-uk,uk-es,es-vi,vi-es,es-zh,zh-es,et-fi,fi-et,et-hi,hi-et,et-hr,hr-et,et-hu,hu-et,et-it,it-et,et-ko,ko-et,et-lt,lt-et,et-lv,lv-et,et-mk,mk-et,et-mn,mn-et,et-nl,nl-et,et-pa,pa-et,et-ps,ps-et,et-pt,pt-et,et-ro,ro-et,et-sv,sv-et,et-sw,sw-et,et-ta,ta-et,et-tr,tr-et,et-uk,uk-et,et-vi,vi-et,fa-fi,fi-fa,fa-fr,fr-fa,fa-he,he-fa,fa-hi,hi-fa,fa-hu,hu-fa,fa-id,id-fa,fa-ja,ja-fa,fa-ko,ko-fa,fa-lt,lt-fa,fa-ml,ml-fa,fa-mn,mn-fa,fa-mr,mr-fa,fa-ms,ms-fa,fa-ne,ne-fa,fa-nl,nl-fa,fa-pa,pa-fa,fa-pl,pl-fa,fa-ps,ps-fa,fa-pt,pt-fa,fa-ro,ro-fa,fa-ru,ru-fa,fa-si,si-fa,fa-sv,sv-fa,fa-sw,sw-fa,fa-ta,ta-fa,fa-tl,tl-fa,fa-tr,tr-fa,fa-ur,ur-fa,fa-uz,uz-fa,fa-vi,vi-fa,fa-zh,zh-fa,ff-ha,ha-ff,ff-ig,ig-ff,ff-lg,lg-ff,ff-ln,ln-ff,ff-ns,ns-ff,ff-so,so-ff,ff-ss,ss-ff,ff-tn,tn-ff,ff-wo,wo-ff,ff-xh,xh-ff,ff-yo,yo-ff,ff-zu,zu-ff,fi-fr,fr-fi,fi-he,he-fi,fi-hi,hi-fi,fi-hu,hu-fi,fi-id,id-fi,fi-it,it-fi,fi-ja,ja-fi,fi-ko,ko-fi,fi-lt,lt-fi,fi-lv,lv-fi,fi-mn,mn-fi,fi-nl,nl-fi,fi-no,no-fi,fi-pa,pa-fi,fi-pl,pl-fi,fi-ps,ps-fi,fi-pt,pt-fi,fi-ro,ro-fi,fi-ru,ru-fi,fi-sv,sv-fi,fi-sw,sw-fi,fi-ta,ta-fi,fi-tr,tr-fi,fi-uk,uk-fi,fi-vi,vi-fi,fi-zh,zh-fi,fr-gl,gl-fr,fr-he,he-fr,fr-hi,hi-fr,fr-hu,hu-fr,fr-id,id-fr,fr-it,it-fr,fr-ja,ja-fr,fr-ko,ko-fr,fr-lt,lt-fr,fr-mn,mn-fr,fr-nl,nl-fr,fr-no,no-fr,fr-pa,pa-fr,fr-pl,pl-fr,fr-ps,ps-fr,fr-pt,pt-fr,fr-ro,ro-fr,fr-ru,ru-fr,fr-sv,sv-fr,fr-sw,sw-fr,fr-ta,ta-fr,fr-tr,tr-fr,fr-uk,uk-fr,fr-vi,vi-fr,fr-zh,zh-fr,fy-is,is-fy,fy-lb,lb-fy,fy-nl,nl-fy,fy-no,no-fy,fy-yi,yi-fy,ga-nl,nl-ga,gd-no,no-gd,gl-it,it-gl,gl-pt,pt-gl,gl-ro,ro-gl,gu-mn,mn-gu,gu-pa,pa-gu,gu-ps,ps-gu,ha-ig,ig-ha,ha-lg,lg-ha,ha-ln,ln-ha,ha-ns,ns-ha,ha-so,so-ha,ha-ss,ss-ha,ha-tn,tn-ha,ha-wo,wo-ha,ha-xh,xh-ha,ha-yo,yo-ha,ha-zu,zu-ha,he-hi,hi-he,he-hu,hu-he,he-id,id-he,he-ja,ja-he,he-ko,ko-he,he-lt,lt-he,he-mg,mg-he,he-ml,ml-he,he-mn,mn-he,he-mr,mr-he,he-ms,ms-he,he-ne,ne-he,he-nl,nl-he,he-pa,pa-he,he-pl,pl-he,he-ps,ps-he,he-pt,pt-he,he-ru,ru-he,he-si,si-he,he-sv,sv-he,he-sw,sw-he,he-ta,ta-he,he-tl,tl-he,he-tr,tr-he,he-ur,ur-he,he-vi,vi-he,he-zh,zh-he,hi-hu,hu-hi,hi-id,id-hi,hi-ilo,ilo-hi,hi-ja,ja-hi,hi-ko,ko-hi,hi-lt,lt-hi,hi-lv,lv-hi,hi-mg,mg-hi,hi-mn,mn-hi,hi-mr,mr-hi,hi-ms,ms-hi,hi-ne,ne-hi,hi-nl,nl-hi,hi-or,or-hi,hi-pa,pa-hi,hi-pl,pl-hi,hi-ps,ps-hi,hi-pt,pt-hi,hi-ru,ru-hi,hi-si,si-hi,hi-sv,sv-hi,hi-sw,sw-hi,hi-ta,ta-hi,hi-tl,tl-hi,hi-tr,tr-hi,hi-ur,ur-hi,hi-vi,vi-hi,hi-zh,zh-hi,hr-hu,hu-hr,hr-lt,lt-hr,hr-lv,lv-hr,hr-mk,mk-hr,hr-mn,mn-hr,hr-pa,pa-hr,hr-ps,ps-hr,hr-ro,ro-hr,hr-ru,ru-hr,hr-sk,sk-hr,hr-sl,sl-hr,hr-sr,sr-hr,hr-uk,uk-hr,hu-id,id-hu,hu-it,it-hu,hu-ja,ja-hu,hu-ko,ko-hu,hu-lt,lt-hu,hu-lv,lv-hu,hu-mn,mn-hu,hu-nl,nl-hu,hu-no,no-hu,hu-pa,pa-hu,hu-pl,pl-hu,hu-ps,ps-hu,hu-pt,pt-hu,hu-ro,ro-hu,hu-ru,ru-hu,hu-sv,sv-hu,hu-sw,sw-hu,hu-ta,ta-hu,hu-tr,tr-hu,hu-uk,uk-hu,hu-vi,vi-hu,hu-zh,zh-hu,hy-ka,ka-hy,id-ilo,ilo-id,id-it,it-id,id-ja,ja-id,id-jv,jv-id,id-ko,ko-id,id-lt,lt-id,id-mg,mg-id,id-ml,ml-id,id-mn,mn-id,id-mr,mr-id,id-ms,ms-id,id-ne,ne-id,id-nl,nl-id,id-no,no-id,id-pa,pa-id,id-pl,pl-id,id-ps,ps-id,id-pt,pt-id,id-ro,ro-id,id-ru,ru-id,id-si,si-id,id-su,su-id,id-sv,sv-id,id-sw,sw-id,id-ta,ta-id,id-tl,tl-id,id-tr,tr-id,id-uk,uk-id,id-ur,ur-id,id-vi,vi-id,id-zh,zh-id,ig-lg,lg-ig,ig-ln,ln-ig,ig-ns,ns-ig,ig-so,so-ig,ig-ss,ss-ig,ig-tn,tn-ig,ig-wo,wo-ig,ig-xh,xh-ig,ig-yo,yo-ig,ig-zu,zu-ig,ilo-jv,jv-ilo,ilo-mg,mg-ilo,ilo-ml,ml-ilo,ilo-ms,ms-ilo,ilo-su,su-ilo,ilo-tl,tl-ilo,is-lb,lb-is,is-mn,mn-is,is-nl,nl-is,is-no,no-is,is-pa,pa-is,is-ps,ps-is,is-sv,sv-is,is-yi,yi-is,it-ja,ja-it,it-ko,ko-it,it-lt,lt-it,it-lv,lv-it,it-mn,mn-it,it-nl,nl-it,it-no,no-it,it-oc,oc-it,it-pa,pa-it,it-pl,pl-it,it-ps,ps-it,it-pt,pt-it,it-ro,ro-it,it-ru,ru-it,it-sv,sv-it,it-sw,sw-it,it-ta,ta-it,it-tr,tr-it,it-uk,uk-it,it-vi,vi-it,it-zh,zh-it,ja-ko,ko-ja,ja-lt,lt-ja,ja-mn,mn-ja,ja-ms,ms-ja,ja-nl,nl-ja,ja-no,no-ja,ja-pa,pa-ja,ja-pl,pl-ja,ja-ps,ps-ja,ja-pt,pt-ja,ja-ru,ru-ja,ja-sv,sv-ja,ja-sw,sw-ja,ja-ta,ta-ja,ja-tl,tl-ja,ja-tr,tr-ja,ja-uk,uk-ja,ja-vi,vi-ja,ja-zh,zh-ja,jv-mg,mg-jv,jv-ml,ml-jv,jv-ms,ms-jv,jv-su,su-jv,jv-tl,tl-jv,kk-mn,mn-kk,kk-pa,pa-kk,kk-ps,ps-kk,kk-tr,tr-kk,km-lo,lo-km,km-mn,mn-km,km-th,th-km,kn-pa,pa-kn,ko-lt,lt-ko,ko-lv,lv-ko,ko-mk,mk-ko,ko-mn,mn-ko,ko-nl,nl-ko,ko-no,no-ko,ko-pa,pa-ko,ko-pl,pl-ko,ko-ps,ps-ko,ko-pt,pt-ko,ko-ru,ru-ko,ko-sk,sk-ko,ko-sl,sl-ko,ko-sr,sr-ko,ko-sv,sv-ko,ko-sw,sw-ko,ko-ta,ta-ko,ko-tr,tr-ko,ko-uk,uk-ko,ko-vi,vi-ko,ko-zh,zh-ko,lb-nl,nl-lb,lb-no,no-lb,lb-zh,zh-lb,lg-ln,ln-lg,lg-ns,ns-lg,lg-so,so-lg,lg-ss,ss-lg,lg-tn,tn-lg,lg-wo,wo-lg,lg-xh,xh-lg,lg-yo,yo-lg,lg-zu,zu-lg,ln-ns,ns-ln,ln-so,so-ln,ln-ss,ss-ln,ln-tn,tn-ln,ln-wo,wo-ln,ln-xh,xh-ln,ln-yo,yo-ln,ln-zu,zu-ln,lo-mn,mn-lo,lo-my,my-lo,lo-th,th-lo,lt-lv,lv-lt,lt-mg,mg-lt,lt-mn,mn-lt,lt-nl,nl-lt,lt-pa,pa-lt,lt-pl,pl-lt,lt-ps,ps-lt,lt-pt,pt-lt,lt-ro,ro-lt,lt-ru,ru-lt,lt-sv,sv-lt,lt-sw,sw-lt,lt-ta,ta-lt,lt-tr,tr-lt,lt-vi,vi-lt,lt-zh,zh-lt,lv-mn,mn-lv,lv-nl,nl-lv,lv-pa,pa-lv,lv-ps,ps-lv,lv-ro,ro-lv,lv-ru,ru-lv,lv-sk,sk-lv,lv-sl,sl-lv,lv-sr,sr-lv,lv-sv,sv-lv,lv-sw,sw-lv,lv-ta,ta-lv,lv-tr,tr-lv,lv-uk,uk-lv,lv-vi,vi-lv,lv-zh,zh-lv,mg-ml,ml-mg,mg-mn,mn-mg,mg-ms,ms-mg,mg-pa,pa-mg,mg-ps,ps-mg,mg-su,su-mg,mg-tl,tl-mg,mk-mn,mn-mk,mk-nl,nl-mk,mk-pa,pa-mk,mk-ps,ps-mk,mk-sk,sk-mk,mk-sl,sl-mk,mk-sv,sv-mk,mk-vi,vi-mk,mk-zh,zh-mk,ml-mn,mn-ml,ml-ms,ms-ml,ml-pa,pa-ml,ml-ps,ps-ml,ml-su,su-ml,ml-tl,tl-ml,ml-tr,tr-ml,ml-zh,zh-ml,mn-mr,mr-mn,mn-ms,ms-mn,mn-my,my-mn,mn-ne,ne-mn,mn-nl,nl-mn,mn-no,no-mn,mn-pa,pa-mn,mn-pl,pl-mn,mn-ps,ps-mn,mn-pt,pt-mn,mn-ro,ro-mn,mn-ru,ru-mn,mn-si,si-mn,mn-sk,sk-mn,mn-sl,sl-mn,mn-sr,sr-mn,mn-su,su-mn,mn-sv,sv-mn,mn-sw,sw-mn,mn-ta,ta-mn,mn-th,th-mn,mn-tl,tl-mn,mn-tr,tr-mn,mn-uk,uk-mn,mn-ur,ur-mn,mn-vi,vi-mn,mn-zh,zh-mn,mr-ne,ne-mr,mr-pa,pa-mr,mr-ps,ps-mr,mr-si,si-mr,mr-ta,ta-mr,mr-tr,tr-mr,mr-ur,ur-mr,ms-pa,pa-ms,ms-ps,ps-ms,ms-su,su-ms,ms-tl,tl-ms,ms-tr,tr-ms,ms-vi,vi-ms,ms-zh,zh-ms,my-th,th-my,ne-pa,pa-ne,ne-ps,ps-ne,ne-si,si-ne,ne-ta,ta-ne,ne-tr,tr-ne,ne-ur,ur-ne,nl-no,no-nl,nl-pa,pa-nl,nl-pl,pl-nl,nl-ps,ps-nl,nl-pt,pt-nl,nl-ro,ro-nl,nl-ru,ru-nl,nl-sk,sk-nl,nl-sl,sl-nl,nl-sr,sr-nl,nl-sv,sv-nl,nl-sw,sw-nl,nl-ta,ta-nl,nl-tr,tr-nl,nl-uk,uk-nl,nl-vi,vi-nl,nl-yi,yi-nl,nl-zh,zh-nl,no-pa,pa-no,no-pl,pl-no,no-ps,ps-no,no-pt,pt-no,no-ro,ro-no,no-ru,ru-no,no-sv,sv-no,no-tr,tr-no,no-uk,uk-no,no-vi,vi-no,no-yi,yi-no,no-zh,zh-no,ns-so,so-ns,ns-ss,ss-ns,ns-tn,tn-ns,ns-wo,wo-ns,ns-xh,xh-ns,ns-yo,yo-ns,ns-zu,zu-ns,or-pa,pa-or,pa-pl,pl-pa,pa-ps,ps-pa,pa-pt,pt-pa,pa-ro,ro-pa,pa-ru,ru-pa,pa-si,si-pa,pa-sk,sk-pa,pa-sl,sl-pa,pa-sr,sr-pa,pa-su,su-pa,pa-sv,sv-pa,pa-sw,sw-pa,pa-ta,ta-pa,pa-tl,tl-pa,pa-tr,tr-pa,pa-uk,uk-pa,pa-ur,ur-pa,pa-vi,vi-pa,pa-zh,zh-pa,pl-ps,ps-pl,pl-pt,pt-pl,pl-ru,ru-pl,pl-sk,sk-pl,pl-sl,sl-pl,pl-sr,sr-pl,pl-sv,sv-pl,pl-ta,ta-pl,pl-tr,tr-pl,pl-uk,uk-pl,pl-vi,vi-pl,pl-zh,zh-pl,ps-pt,pt-ps,ps-ro,ro-ps,ps-ru,ru-ps,ps-si,si-ps,ps-sk,sk-ps,ps-sl,sl-ps,ps-sr,sr-ps,ps-su,su-ps,ps-sv,sv-ps,ps-sw,sw-ps,ps-ta,ta-ps,ps-tl,tl-ps,ps-tr,tr-ps,ps-uk,uk-ps,ps-ur,ur-ps,ps-vi,vi-ps,ps-zh,zh-ps,pt-ro,ro-pt,pt-ru,ru-pt,pt-sv,sv-pt,pt-sw,sw-pt,pt-ta,ta-pt,pt-tr,tr-pt,pt-uk,uk-pt,pt-vi,vi-pt,pt-zh,zh-pt,ro-sk,sk-ro,ro-sl,sl-ro,ro-sr,sr-ro,ro-ta,ta-ro,ro-tr,tr-ro,ro-uk,uk-ro,ru-sk,sk-ru,ru-sl,sl-ru,ru-sr,sr-ru,ru-sv,sv-ru,ru-sw,sw-ru,ru-ta,ta-ru,ru-tr,tr-ru,ru-uk,uk-ru,ru-ur,ur-ru,ru-vi,vi-ru,ru-zh,zh-ru,si-ta,ta-si,si-tr,tr-si,si-ur,ur-si,sk-sl,sl-sk,sk-sr,sr-sk,sk-sv,sv-sk,sk-sw,sw-sk,sk-uk,uk-sk,sk-vi,vi-sk,sk-zh,zh-sk,sl-sr,sr-sl,sl-sv,sv-sl,sl-sw,sw-sl,sl-uk,uk-sl,sl-vi,vi-sl,so-ss,ss-so,so-tn,tn-so,so-wo,wo-so,so-xh,xh-so,so-zu,zu-so,sr-sv,sv-sr,sr-sw,sw-sr,sr-uk,uk-sr,sr-vi,vi-sr,ss-tn,tn-ss,ss-wo,wo-ss,ss-xh,xh-ss,ss-yo,yo-ss,ss-zu,zu-ss,su-tl,tl-su,sv-sw,sw-sv,sv-ta,ta-sv,sv-tr,tr-sv,sv-uk,uk-sv,sv-vi,vi-sv,sv-zh,zh-sv,sw-ta,ta-sw,sw-tr,tr-sw,sw-uk,uk-sw,sw-vi,vi-sw,sw-zh,zh-sw,ta-tl,tl-ta,ta-tr,tr-ta,ta-ur,ur-ta,ta-vi,vi-ta,ta-zh,zh-ta,tl-tr,tr-tl,tl-vi,vi-tl,tl-zh,zh-tl,tn-wo,wo-tn,tn-xh,xh-tn,tn-yo,yo-tn,tn-zu,zu-tn,tr-uk,uk-tr,tr-ur,ur-tr,tr-vi,vi-tr,uk-vi,vi-uk,uk-zh,zh-uk,vi-zh,zh-vi,wo-xh,xh-wo,wo-yo,yo-wo,wo-zu,zu-wo,xh-yo,yo-xh,xh-zu,zu-xh,yi-zh,zh-yi,yo-zu,zu-yo', lang_tok_replacing_bos_eos=False, lang_tok_style='multilingual', langs=['af', 'am', 'ar', 'as', 'ast', 'ay', 'az', 'ba', 'be', 'bg', 'bn', 'br', 'bs', 'ca', 'ceb', 'cjk', 'cs', 'cy', 'da', 'de', 'dyu', 'el', 'en', 'es', 'et', 'fa', 'ff', 'fi', 'fr', 'fy', 'ga', 'gd', 'gl', 'gu', 'ha', 'he', 'hi', 'hr', 'ht', 'hu', 'hy', 'id', 'ig', 'ilo', 'is', 'it', 'ja', 'jv', 'ka', 'kac', 'kam', 'kea', 'kg', 'kk', 'km', 'kmb', 'kmr', 'kn', 'ko', 'ku', 'ky', 'lb', 'lg', 'ln', 'lo', 'lt', 'luo', 'lv', 'mg', 'mi', 'mk', 'ml', 'mn', 'mr', 'ms', 'mt', 'my', 'ne', 'nl', 'no', 'ns', 'ny', 'oc', 'om', 'or', 'pa', 'pl', 'ps', 'pt', 'qu', 'ro', 'ru', 'sd', 'shn', 'si', 'sk', 'sl', 'sn', 'so', 'sq', 'sr', 'ss', 'su', 'sv', 'sw', 'ta', 'te', 'tg', 'th', 'ti', 'tl', 'tn', 'tr', 'uk', 'umb', 'ur', 'uz', 'vi', 'wo', 'xh', 'yi', 'yo', 'zh', 'zu'], langtoks={'main': ('src', 'tgt')}, langtoks_specs=['main'], left_pad_source='False', left_pad_target='False', lenpen=1.0, lm_path=None, lm_weight=0.0, load_alignments=False, load_checkpoint_on_all_dp_ranks=False, localsgd_frequency=3, log_format=None, log_interval=100, lr_scheduler='fixed', lr_shrink=0.1, match_source_len=False, max_len_a=0, max_len_b=1024, max_source_positions=1024, max_target_positions=1024, max_tokens=None, max_tokens_valid=None, maximize_best_checkpoint_metric=False, memory_efficient_bf16=False, memory_efficient_fp16=False, min_len=1, min_loss_scale=0.0001, min_sampling_temperature=1.0, model_overrides='{}', model_parallel_size=1, nbest=1, no_beamable_mm=False, no_early_stop=False, no_epoch_checkpoints=False, no_last_checkpoints=False, no_progress_bar=True, no_repeat_ngram_size=0, no_save=False, no_save_optimizer_state=False, no_seed_provided=False, nprocs_per_node=4, num_shards=1, num_workers=1, optimizer=None, optimizer_overrides='{}', pad=1, path='/home/v-jiaya/SharedTask/PretrainedModel/mm100_615M/flores101_mm100_615M/model.pt', patience=-1, pipeline_balance=None, pipeline_checkpoint='never', pipeline_chunks=0, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_devices=None, pipeline_encoder_balance=None, pipeline_encoder_devices=None, pipeline_model_parallel=False, post_process=None, prefix_size=0, print_alignment=False, print_step=False, profile=False, quantization_config_path=None, quiet=False, replace_unk=None, required_batch_size_multiple=8, required_seq_len_multiple=1, reset_dataloader=False, reset_logging=True, reset_lr_scheduler=False, reset_meters=False, reset_optimizer=False, restore_file='checkpoint_last.pt', results_path=None, retain_dropout=False, retain_dropout_modules=None, retain_iter_history=False, sacrebleu=False, same_lang_per_batch=True, sampling=False, sampling_method='concat', sampling_temperature=1.5, sampling_topk=-1, sampling_topp=-1.0, sampling_weights=None, sampling_weights_from_file=None, save_dir='checkpoints', save_interval=1, save_interval_updates=0, score_reference=False, scoring='bleu', seed=1, shard_id=0, skip_invalid_size_inputs_valid_test=False, slowmo_algorithm='LocalSGD', slowmo_momentum=None, source_lang=None, target_lang=None, task='translation_multi_simple_epoch', temperature=1.0, tensorboard_logdir=None, threshold_loss_scale=None, tokenizer=None, tpu=False, train_subset='train', truncate_source=True, unk=3, unkpen=0, unnormalized=False, upsample_primary=1, user_dir=None, valid_subset='valid', validate_after_updates=0, validate_interval=1, validate_interval_updates=0, virtual_data_size=None, virtual_epoch_size=None, wandb_project=None, warmup_epoch=5, warmup_updates=0, zero_sharding='none')
        self.cfg.generation = OmegaConf.create({'_name': None, 'beam': 5, 'nbest': 1, 'max_len_a': 0.0, 'max_len_b': 1024, 'min_len': 1, 'match_source_len': False, 'unnormalized': False, 'no_early_stop': False, 'no_beamable_mm': False, 'lenpen': 1.0, 'unkpen': 0.0, 'replace_unk': None, 'sacrebleu': False, 'score_reference': False, 'prefix_size': 0, 'no_repeat_ngram_size': 0, 'sampling': False, 'sampling_topk': -1, 'sampling_topp': -1.0, 'constraints': None, 'temperature': 1.0, 'diverse_beam_groups': -1, 'diverse_beam_strength': 0.5, 'diversity_rate': -1.0, 'print_alignment': False, 'print_step': False, 'lm_path': None, 'lm_weight': 0.0, 'iter_decode_eos_penalty': 0.0, 'iter_decode_max_iter': 10, 'iter_decode_force_max_iter': False, 'iter_decode_with_beam': 1, 'iter_decode_with_external_reranker': False, 'retain_iter_history': False, 'retain_dropout': False, 'retain_dropout_modules': None, 'decoding_format': None, 'no_seed_provided': False})
        self.cfg.common_eval = OmegaConf.create({'_name': None, 'path': model_pt_path, 'post_process': None, 'quiet': False, 'model_overrides': '{}', 'results_path': None})
        logger.info(model_pt_path)
        logger.info(model_file_dir)
        logger.info(device_str)
        self.task = tasks.setup_task(self.cfg.task)
        # Handle tokenization and BPE
        self.cfg.tokenizer = None
        self.cfg.bpe = OmegaConf.create({'_name': 'sentencepiece', 'sentencepiece_model': 'sentencepiece.bpe.model'})
        self.tokenizer = encoders.build_tokenizer(self.cfg.tokenizer)
        self.bpe = encoders.build_bpe(self.cfg.bpe)
        self.src_langtok_spec, self.tgt_langtok_spec = self.cfg.task.langtoks["main"]
        print("SPM: {}".format(self.bpe))
        self.models, _model_args = checkpoint_utils.load_model_ensemble(model_pt_path.split(), task=self.task)
        print("Model: {}".format(self.models))
        # Set dictionaries
        self.src_dict = self.task.source_dictionary
        self.tgt_dict = self.task.target_dictionary

        self.use_cuda = torch.cuda.is_available()
        print("Use_Cuda: {}".format(self.use_cuda))
        # Optimize ensemble for generation
        for model in self.models:
            if model is None:
                continue
            if self.cfg.common.fp16:
                model.half()
            if self.use_cuda:
                model.cuda()
            model.prepare_for_inference_(self.cfg)

        # Initialize generator
        self.generator = self.task.build_generator(self.models, self.cfg.generation)
        self.max_positions = utils.resolve_max_positions(
            self.task.max_positions(), *[model.max_positions() for model in self.models]
        )
        self.initialized = True


    def preprocess(self, data):
        """
        preprocess data into a format that the model can do inference on
        """
        example = self._read_data(data)
        uid = example["uid"]
        src_lang = example["sourceLanguage"]
        tgt_lang = example["targetLanguage"]
        # self.task.source_langs = [ LANG_CODE[src_lang] ]
        # self.task.target_langs = [ LANG_CODE[tgt_lang] ]
        self.task.args.source_lang = LANG_CODE[src_lang]
        self.task.args.target_lang = LANG_CODE[tgt_lang]
        src_str = example["sourceText"]
        #print(encode_fn(src_str, self.bpe, self.tokenizer))
        src_tokens = self.task.source_dictionary.encode_line(encode_fn(src_str, self.bpe, self.tokenizer), add_if_not_exist=False).long()[:self.max_positions[0] - 1]
        src_lang_tok = torch.LongTensor([self.task.data_manager.get_encoder_langtok(LANG_CODE[src_lang], LANG_CODE[tgt_lang], spec=self.src_langtok_spec)])
        src_tokens = torch.cat([src_lang_tok, src_tokens]).unsqueeze(0)
        src_lengths = torch.LongTensor([src_tokens.numel()])
        input_data = {
            "net_input": {
                "src_tokens": src_tokens.cuda() if self.use_cuda else src_tokens,
                "src_lengths": src_lengths.cuda() if self.use_cuda else src_lengths,
            },
            "src_str": src_str,
            "src_lang": src_lang,
            "tgt_lang": tgt_lang
        }
        #print(input_data)
        return input_data

    def inference(self, input_data):
        """
        do inference on the processed example
        """

        # ############TODO 3: inference ###################
        """
        Run model prediction using the processed data
        """
        hypos = self.task.inference_step(
            self.generator, self.models, input_data,
        )
        #print(len(hypos[0]))
        #exit()
        # #################################################
        for hypo in hypos[: min(len(hypos), self.cfg.generation.nbest)]:
            #print(hypo[0]['tokens'])
            #print(self.cfg.common_eval.post_process)
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo[0]["tokens"].int().cpu(),
                src_str=input_data["src_str"],
                alignment=hypo[0]["alignment"],
                align_dict=None,
                tgt_dict=self.tgt_dict,
                remove_bpe=self.cfg.common_eval.post_process,
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(self.generator),
            )
            #if self.debug:
            #    print(hypo_str)
            inference_output = decode_fn(hypo_str, self.bpe, self.tokenizer)
            score = hypo[0]["score"] / math.log(2)  # convert to base 2
            #if self.debug:
            #    print(inference_output)
        return inference_output

    def postprocess(self, inference_output, data):
        """
        post process inference output into a response.
        response should be a single element list of a json
        the response format will need to pass the validation in
        ```
        dynalab.tasks.{your_task}.TaskIO().verify_response(response, data)
        ```
        """
        response = dict()
        example = self._read_data(data)
        # ############TODO 4: postprocess response ########
        """
        Add attributes to response
        """
        response["id"] = example["uid"]
        response["translatedText"] = inference_output
        if self.debug:
            print(inference_output)
        # #################################################
        response = self.taskIO.sign_response(response, example)
        return [response]


_service = Handler()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)
    if data is None:
        return None

    # ############TODO 5: assemble inference pipeline #####
    """
    Normally you don't need to change anything in this block.
    However, if you do need to change this part (e.g. function name, argument, etc.),
    remember to make corresponding changes in the Handler class definition.
    """
    input_data = _service.preprocess(data)
    output = _service.inference(input_data)
    response = _service.postprocess(output, data)
    # #####################################################

    return response


def local_test():
    from dynalab.tasks import flores_small1

    bin_data = b"\n".join(json.dumps(d).encode("utf-8") for d in flores_small1.data)
    torchserve_data = [{"body": bin_data}]

    manifest = {"model": {"serializedFile": "model.pt"}}
    system_properties = {"model_dir": ".", "gpu_id": 0}

    class Context(NamedTuple):
        system_properties: dict
        manifest: dict

    ctx = Context(system_properties, manifest)
    batch_responses = handle(torchserve_data, ctx)
    print(batch_responses)

    single_responses = [
        handle([{"body": json.dumps(d).encode("utf-8")}], ctx)[0]
        for d in flores_small1.data
    ]
    assert batch_responses == ["\n".join(single_responses)]


if __name__ == "__main__":
    local_test()