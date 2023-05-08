import os
import argparse
import linecache
import random
from multiprocessing import Pool
LANGS="en id ta tl jv ms".split()
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/mnt/input/SharedTask/thunder/MonolingualData/all_spm_x2e_bt/', help='input src')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/mnt/input/SharedTask/thunder/small_task2/Filter_v1/parallel_bt_train/', help='output tgt')
    parser.add_argument('--length-ratio', '-length-ratio', type=float, default=2.5)
    args = parser.parse_args()
    return args

def decode(x: str) -> str:
    return x.replace(" ", "").replace("\u2581", " ").strip()


def read_lines(file):
    lines = linecache.getlines(os.path.join(args.input, file))
    return lines


if __name__ == "__main__":
    args = parse_args()
    print(args)
    length_ratio = args.length_ratio
    max_length = 1024
    files = os.listdir(args.input)
    lines = []
    langs = []
    en_lines = None
    N = 60000000
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #files = files[:2] + ["train.en"]
    used_files = []
    for file in files:
        if file.split('.')[-1] not in LANGS:
            continue
        if file.split('.')[-1] == "en":
            en_lines = linecache.getlines(os.path.join(args.input, file))
            en_w = open(os.path.join(args.output, file), "w", encoding="utf-8")
            assert len(en_lines) == N
            print("Successfully loading from {} | {} lines".format(os.path.join(args.input, file), len(en_lines)))
        else:
            used_files.append(file)
            langs.append(file.split('.')[-1])
            lines.append(linecache.getlines(os.path.join(args.input, file)))
            assert len(lines[-1]) == N
            print("Successfully loading from {} | {} lines".format(os.path.join(args.input, file), len(lines[-1])))


    print("FILES: {}".format(used_files))
    w_list = [open(os.path.join(args.output, file), "w", encoding="utf-8") for file in used_files]

    remaining_count = 0
    count = 0
    rm_count = 0
    unk_rm_count = 0
    blank_rm_count = 0
    length_ratio_rm_count = 0
    max_length_rm_count = 0
    latin_rm_count = 0


    print("Start shuffling pairs...")
    indices = list(range(N))
    random.shuffle(indices)
    print("Complete shuffling pairs...")
    for i in indices:
        current_lines = [lines[j][i] for j in range(len(langs))]
        if count % 500000 == 0:
            print("Complete processing {} examples | removing {} examples (unk) | removing {} examples (length > {}) | removing {} examples (length ratio > {}) | removing {} examples (blank) | removing {} examples (latin)".format(count, unk_rm_count, max_length_rm_count, max_length, length_ratio_rm_count, length_ratio, blank_rm_count, latin_rm_count), end="\r")
        count += 1
        if_rm = False
        for tok_line in current_lines:
            if "< unk >" in tok_line:
                unk_rm_count += 1
                rm_count += 1
                if_rm = True
                break
        if if_rm:
            continue


        if len(en_lines[i].split()) >= max_length:
            max_length_rm_count += 1
            rm_count += 1
            continue


        if len(en_lines[i].split()) == 0:
            blank_rm_count += 1
            rm_count += 1
            continue


        if_rm = False
        for tok_line in current_lines:
            if len(tok_line.split()) == 0:
                max_length_rm_count += 1
                rm_count += 1
                if_rm = True
                break
        if if_rm:
            continue

        if_rm = False
        for tok_line in current_lines:
            if len(tok_line.split()) / len(en_lines[i].split()) > length_ratio or len(en_lines[i].split()) / len(tok_line.split()) > length_ratio:
                length_ratio_rm_count += 1
                rm_count += 1
                if_rm = True
                break
        if if_rm:
            continue

        for k in range(len(w_list)):
            w_list[k].write(current_lines[k])

        en_w.write(en_lines[i])
        remaining_count += 1

    print("\nResults: {} | removing {} examples (unk) | removing {} examples (length > {}) | removing {} examples (length ratio > {}) | removing {} examples (blank) | removing {} examples (latin)".format(remaining_count, unk_rm_count, max_length_rm_count, max_length, length_ratio_rm_count, length_ratio, blank_rm_count, latin_rm_count))
