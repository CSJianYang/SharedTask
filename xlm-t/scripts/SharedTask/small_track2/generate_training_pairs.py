import argparse
import os
LANGS="en,id,jv,ms,ta,tl".split(',')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang-pairs', '-lang-pairs', type=str,
                        default=r'/mnt/input/SharedTask/thunder/small_task2/download/train/', help='input stream')
    parser.add_argument('--bt-lang-pairs', '-bt-lang-pairs', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/bt_split80/train0/', help='input stream')
    parser.add_argument('--parallel-bt-lang-pairs', '-parallel-bt-lang-pairs', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/parallel_bt_split80/train0/', help='input stream')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    files = os.listdir(args.lang_pairs)
    files.sort()
    lang_pairs = []
    for file in files:
        if file.split('.')[-2] not in lang_pairs:
            src, tgt = file.split('.')[-2].split('-')
            if src not in LANGS or tgt not in LANGS:
                print("LANGS don not contain language {} or {}!".format(src, tgt))
                continue
            if os.path.getsize(os.path.join(args.lang_pairs, file)) > 0:
                lang_pairs.append("{}-{}".format(src, tgt))
                lang_pairs.append("{}-{}".format(tgt, src))
            else:
                print("{} is empty !".format(file))
    #lang_pairs.sort()
    lang_pairs = ",".join(lang_pairs)
    print(lang_pairs)

