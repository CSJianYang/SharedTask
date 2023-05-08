import os
import argparse
import sentencepiece as spm
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/mnt/input/SharedTask/thunder/Bitext_v1/', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    lang_pairs = os.listdir(args.input)
    lang_pairs.sort()
    cmds = ""
    for lang_pair in lang_pairs:
        cmd = "- name: {}\n  sku: G0\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/large_task/spm_file.sh {}\n".format(lang_pair, lang_pair)
        cmds += cmd
    print(cmds)