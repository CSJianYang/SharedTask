import argparse
import os
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(',')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang-pairs', '-lang-pairs', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/bt_split80/train0/', help='input stream')
    parser.add_argument('--split-num', '-split-num', type=int,
                        default=80, help='input stream')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    files = os.listdir(args.lang_pairs)
    files.sort()
    lang_pairs = []
    for file in files:
        #if os.path.getsize(os.path.join(args.lang_pairs, file)) > 0:
        if file.split('.')[-2] not in lang_pairs:
            src, tgt = file.split('.')[-2].split('-')
            if src not in LANGS or tgt not in LANGS:
                print("LANGS don not contain language {} or {}!".format(src, tgt))
                continue
            if os.path.getsize(os.path.join(args.lang_pairs, file)) > 0:
                lang_pairs.append("{}-{}".format(src, tgt))
            else:
                print("{} is empty !".format(file))

    split_num = args.split_num
    STEP = 10
    cmds = ""
    for lang_pair in lang_pairs:
        for i in range(0, split_num, STEP):
            src, tgt = lang_pair.split('-')
            cmds += "- name: binary_{}-{}_{}-{}\n  sku: G0\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/large_task/bt/bt_binary_lang_pair.sh {} {} {} {} {}\n".format(src, tgt, i, i + STEP, src, tgt, split_num, i, i + STEP)
    print(cmds)


