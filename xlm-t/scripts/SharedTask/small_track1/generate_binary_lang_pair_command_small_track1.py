import argparse
import os
LANGS="en,et,hr,hu,sr,mk".split(',')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang-pairs', '-lang-pairs', type=str,
                        default=r'/mnt/input/SharedTask/thunder/small_task1/Filter_v1/train/', help='input stream')
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
    print("lang pairs: {}".format(len(lang_pairs)))
    for lang_pair in lang_pairs:
        src , tgt = lang_pair.split('-')
        cmds += "- name: binary_{}-{}\n  sku: G0\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/small_track1/thunder/binary_lang_pair.sh {} {} \n".format(src, tgt, src, tgt)
    print(cmds)