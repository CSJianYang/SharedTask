import os
import argparse
LANGS="en,be,ff,ku,lo,my,ns,om,or".split(",")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-input-dir', type=str,
                        default=r'/mnt/input/SharedTask/thunder/MonolingualData/cc100/all_spm_split_10W/', help='input stream')
    parser.add_argument('--output-dir', '-output-dir', type=str,
                        default=r'/mnt/input/SharedTask/thunder/MonolingualData/cc100/all_spm_split_10W_bt/', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cmds = ""
    files = os.listdir(args.input_dir)
    BEAM = 4
    MODEL = "/mnt/input/SharedTask/thunder/large_track/data/model/deltalm/bt/A100/lr1e-4-deltalm-postnorm/avg50_59.pt"
    BATCH_SIZE = 32
    print("Command Start:")
    count = 0
    for file in files:
        input = file
        src = input.split('.')[-1][:2]
        index = input.split('.')[-1][-4:]
        for tgt in LANGS:
            if src != "en" and tgt == "en":
                output = "{}{}.2{}".format(src, index, tgt)
                cmd = "- name: {}-{}\n  sku: G1\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/BT/translate_our.sh {} {} {} {} {} {} {}\n".format(file, "{}2{}".format(src, tgt), src, tgt, BEAM, MODEL, input, output, BATCH_SIZE)
                count += 1
            else:
                continue
            cmds += cmd
    with open("/home/v-jiaya/SharedTask/xlm-t/cc100_translate.txt", "w", encoding="utf-8") as w:
        w.write(cmds)
    print(cmds)
