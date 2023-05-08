import argparse
import os
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(',')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parallel-bt-lang-pairs', '-parallel-bt-lang-pairs', type=str,
                        default=r'/mnt/input/SharedTask/thunder/BT/4M-all-spm/', help='input stream')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    files = os.listdir(args.parallel_bt_lang_pairs)
    files.sort()
    parallel_bt_lang_pairs = []
    for file in files:
        #if os.path.getsize(os.path.join(args.lang_pairs, file)) > 0:
        if file.split('.')[-1] not in parallel_bt_lang_pairs:
            lang = file.split('.')[-1]
            if lang not in LANGS:
                print("LANGS don not contain language {}!".format(lang))
                continue
            if os.path.getsize(os.path.join(args.parallel_bt_lang_pairs, file)) > 0:
                parallel_bt_lang_pairs.append(lang)
            else:
                print("{} is empty !".format(file))
    cmds = ""
    input_dir = "/mnt/input/SharedTask/thunder/large_track/data/Filter_v1/parallel_bt_spm/"
    output_dir = "/mnt/input/SharedTask/thunder/large_track/data/Filter_v1/parallel_bt_split80/"
    for lang in parallel_bt_lang_pairs:
        input = os.path.join(input_dir, "train.{}".format(lang))
        output = output_dir
        cmds += "- name: parallel_bt_split_v1_{}\n  sku: G0\n  sku_count: 1\n  command: \n    - python ./scripts/SharedTask/SplitTrainingData.py -input {} -output {} \n".format(lang, input, output)
    print(cmds)