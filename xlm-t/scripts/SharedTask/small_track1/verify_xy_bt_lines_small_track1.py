import os
import argparse
import linecache
def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )
SMALL_TRACK1_LANGS="en,et,hr,hu,sr,mk".split(",")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-input-dir', type=str,
                        default=r'/mnt/input/SharedTask/thunder/MonolingualData/all_spm_split_10W/', help='input stream')
    parser.add_argument('--output-dir', '-output-dir', type=str,
                        default=r'/mnt/input/SharedTask/thunder/MonolingualData/all_spm_split_10W_bt_iter2/', help='input stream')
    parser.add_argument('--max-index', '-max-index', type=int,
                        default=600, help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)
    cmds = ""
    input_files = os.listdir(args.input_dir)
    BEAM = 4
    MODEL = "/mnt/input/SharedTask/thunder/small_task2/Filter_v1/model/24L-12L-step2/avg4_8.pt"
    BATCH_SIZE = 48
    count = 0
    MAX_NUM = args.max_index
    for file in input_files:
        print(u"Complete processing {} examples".format(file), end="\r")
        src = file.split('.')[0]
        index = file.split('.')[-1][-4:]
        if int(index) >= MAX_NUM:
            continue
        if src not in SMALL_TRACK1_LANGS:
            continue
        for tgt in SMALL_TRACK1_LANGS:
            if src != "en" and tgt != "en" and src != tgt:
                output = "{}{}.2{}".format(src, index, tgt)
                #input_lines = linecache.getlines(os.path.join(args.input_dir, file))
                if  not os.path.exists(os.path.join(args.output_dir, output)):
                    print("{} don't exist!".format(os.path.join(args.output_dir, output)))
                else:
                    output_lines = linecache.getlines(os.path.join(args.output_dir, output))
                    if len(output_lines) != 100000:
                        input_lines = linecache.getlines(os.path.join(args.input_dir, file))
                        if len(input_lines) == len(output_lines):
                            continue
                        print("{} < 100000 lines | input: {} lines | output {} lines".format(os.path.join(args.output_dir, output), len(input_lines), len(output_lines)))
                    else:
                        continue
                cmd = "- name: {}-{}\n  sku: G1\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/BT/small_track1/translate_our.sh {} {} {} {} {} {} {}\n".format(file, "{}2{}".format(src, tgt), src, tgt, BEAM, MODEL, file, output, BATCH_SIZE)
                count += 1
                cmds += cmd
    with open("/mnt/input/SharedTask/thunder/MonolingualData/small_track2_translate.txt", "w", encoding="utf-8") as w:
        w.write(cmds)
    print(cmds)
