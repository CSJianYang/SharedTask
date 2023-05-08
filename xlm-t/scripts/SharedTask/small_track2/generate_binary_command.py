import argparse
import os
LANGS="en,id,jv,ms,ta,tl".split(',')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang-pairs', '-lang-pairs', type=str,
                        default=r'/mnt/input/SharedTask/thunder/small_task2/download/train/', help='input stream')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    used_pairs = [file.split('.')[-2] for file in os.listdir(args.lang_pairs)]
    lang_pairs = []
    for src in LANGS:
        for tgt in LANGS:
            if src != tgt and "{}-{}".format(src, tgt) not in lang_pairs and "{}-{}".format(src, tgt) in used_pairs:
                lang_pairs.append("{}-{}".format(src, tgt))
    cmds = ""
    for lang_pair in lang_pairs:
        src , tgt = lang_pair.split('-')
        cmds += "- name: binary_{}-{}\n  sku: G0\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/small_track2/binary_lang_pair.sh {} {}\n".format(src, tgt, src, tgt)
    print(cmds)