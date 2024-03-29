import os
import argparse
import sentencepiece as spm
def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )
SMALL_TRACK2_LANGS="en,id,jv,ms,ta,tl".split(",")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-input-dir', type=str,
                        default=r'/mnt/input/SharedTask/thunder/MonolingualData/all_spm_split_10W/', help='input stream')
    parser.add_argument('--output-dir', '-output-dir', type=str,
                        default=r'/mnt/input/SharedTask/thunder/MonolingualData/all_spm_split_10W_bt/', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cmds = ""
    files = os.listdir(args.input_dir)
    BEAM = 4
    MODEL = "/mnt/input/SharedTask/thunder/small_task2/Filter_v1/model/24L-12L-step2/avg4_8.pt"
    BATCH_SIZE = 24
    print("Command Start:")
    count = 0
    for file in files:
        src = file.split('.')[0]
        index = file.split('.')[-1][-4:]
        if src not in SMALL_TRACK2_LANGS:
            continue
        for tgt in SMALL_TRACK2_LANGS:
            if src == "en" and src != tgt:
                output = "{}{}.2{}".format(src, index, tgt)
                cmd = "- name: {}-{}\n  sku: G1\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/BT/small_track2/translate_our.sh {} {} {} {} {} {} {}\n".format(file, "{}2{}".format(src, tgt), src, tgt, BEAM, MODEL, file, output, BATCH_SIZE)
                count += 1
            elif src != "en" and tgt == "en":
                output = "{}{}.2{}".format(src, index, tgt)
                cmd = "- name: {}-{}\n  sku: G1\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/BT/small_track2/translate_our.sh {} {} {} {} {} {} {}\n".format(file, "{}2{}".format(src, tgt), src, tgt, BEAM, MODEL, file, output, BATCH_SIZE)
                count += 1
            else:
                continue
            cmds += cmd
    with open("/home/v-jiaya/SharedTask/xlm-t/small_track2_translate.txt", "w", encoding="utf-8") as w:
        w.write(cmds)
    print(cmds)
