import os
import argparse
import sentencepiece as spm
def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    N = 80
    cmds = ""
    for index in range(N):
        cmd = "- name: binary_{}\n  sku: G0\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/large_task/binary_file.sh {}\n".format(index, index)
        cmds += cmd
    print(cmds)