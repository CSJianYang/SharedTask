import os
import argparse
import sentencepiece as spm
def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )
SMALL_TRACK1_LANGS="en,et,hr,mk,sr,hu".split(",")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-input-dir', type=str,
                        default=r'/mnt/input/XTREME-Pattern/data_from_shuming/split_10W/', help='input stream')
    parser.add_argument('--output-dir', '-output-dir', type=str,
                        default=r'/mnt/input/XTREME-Pattern/data_from_shuming/split_10W_BT/', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cmds = ""
    files = os.listdir(args.input_dir)
    BEAM = 4
    MODEL = "/mnt/input/SharedTask/thunder/small_task1/download/model/deltalm/lr1e-4-deltalm-postnorm/checkpoint_last.pt"
    BATCH_SIZE = 32
    print("Command Start:")
    count = 0
    for file in files:
        src = file.split('.')[0]
        index = file.split('.')[-1][-4:]
        for tgt in SMALL_TRACK1_LANGS:
            if src == "en" and src != tgt:
                output = "{}{}.2{}".format(src, index, tgt)
                cmd = "- name: {}-{}\n  sku: G1\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/BT/small_track1/translate_our.sh {} {} {} {} {} {} {}\n".format(file, "{}2{}".format(src, tgt), src, tgt, BEAM, MODEL, file, output, BATCH_SIZE)
                count += 1
            elif src != "en" and tgt == "en":
                output = "{}{}.2{}".format(src, index, tgt)
                cmd = "- name: {}-{}\n  sku: G1\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/BT/small_track1/translate_our.sh {} {} {} {} {} {} {}\n".format(file, "{}2{}".format(src, tgt), src, tgt, BEAM, MODEL, file, output, BATCH_SIZE)
                count += 1
            else:
                continue
            cmds += cmd
    with open("/home/v-jiaya/SharedTask/xlm-t/small_track1_translate.txt", "w", encoding="utf-8") as w:
        w.write(cmds)
    print(cmds)
