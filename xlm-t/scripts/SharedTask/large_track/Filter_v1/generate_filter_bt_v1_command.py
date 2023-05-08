import argparse
import os
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(',')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bt-lang-pairs', '-bt-lang-pairs', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/bt_split80/train0/', help='input stream')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    files = os.listdir(args.bt_lang_pairs)
    files.sort()
    bt_lang_pairs = []
    for file in files:
        #if os.path.getsize(os.path.join(args.bt_lang_pairs, file)) > 0:
        if file.split('.')[-2] not in bt_lang_pairs:
            src, tgt = file.split('.')[-2].split('-')
            if src not in LANGS or tgt not in LANGS:
                print("LANGS don not contain language {} or {}!".format(src, tgt))
                continue
            if os.path.getsize(os.path.join(args.bt_lang_pairs, file)) > 0:
                bt_lang_pairs.append("{}-{}".format(src, tgt))
            else:
                print("{} is empty !".format(file))
    cmds = ""
    input_dir = "/mnt/input/SharedTask/thunder/large_track/data/bt_spm/"
    output_dir = "/mnt/input/SharedTask/thunder/large_track/data/Filter_v1/bt_spm/"
    for lang_pair in bt_lang_pairs:
        src , tgt = lang_pair.split('-')
        input_src = os.path.join(input_dir, "{}{}".format(src, tgt), "train.{}-{}.{}".format(src, tgt, src))
        input_tgt = os.path.join(input_dir, "{}{}".format(src, tgt), "train.{}-{}.{}".format(src, tgt, tgt))
        output_src = os.path.join(output_dir, "{}{}".format(src, tgt), "train.{}-{}.{}".format(src, tgt, src))
        output_tgt = os.path.join(output_dir, "{}{}".format(src, tgt), "train.{}-{}.{}".format(src, tgt, tgt))
        cmds += "- name: bt_filter_v1_{}-{}\n  sku: G0\n  sku_count: 1\n  command: \n    - python ./scripts/SharedTask/large_track/Filter_v1/filter.py -src {} -tgt {} -new-src {} -new-tgt {} \n".format(src, tgt, input_src, input_tgt, output_src, output_tgt)
    print(cmds)