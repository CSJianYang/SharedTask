import argparse
import os
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(',')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang-pairs', '-lang-pairs', type=str,
                        default=r'/mnt/input/SharedTask/thunder/small_task2/Filter_v1/train/', help='input stream')
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
    cmds = ""
    split_num = 10
    input_dir = "/mnt/input/SharedTask/thunder/small_task2/Filter_v1/train/"
    output_dir = "/mnt/input/SharedTask/thunder/small_task2/Filter_v1/split{}/".format(split_num)
    for lang_pair in lang_pairs:
        src , tgt = lang_pair.split('-')
        input = os.path.join(input_dir, "{}{}".format(src, tgt))
        output = output_dir
        cmds += "- name: split_v1_{}-{}\n  sku: G0\n  sku_count: 1\n  command: \n    - python ./scripts/SharedTask/SplitTrainingData.py -input {} -output {} -split-num {}\n".format(src, tgt, input, output, split_num)
    print(cmds)