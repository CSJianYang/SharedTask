import argparse
import os
LANGS="en,et,hr,hu,sr,mk".split(',')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-num', '-split-num', type=int,
                        default=10, help='input stream')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    split_num = args.split_num
    STEP = 1
    cmds = ""
    for lang in LANGS:
        for i in range(0, split_num, STEP):
            cmds += "- name: binary_parallel_bt_{}_{}-{}\n  sku: G0\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/BT/small_track1/binary_parallel_bt.sh {} {} {} {}\n".format(lang, i, i + STEP, lang, split_num, i, i + STEP)
    print(cmds)