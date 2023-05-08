import os
import argparse
import linecache
import random
from multiprocessing import Pool
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(',')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/home/v-jiaya/tmp/4M-all-spm/', help='input src')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/thunder/large_task/data/Filter_v1/parallel_bt_spm/', help='output tgt')
    args = parser.parse_args()
    return args

def decode(x: str) -> str:
    return x.replace(" ", "").replace("\u2581", " ").strip()


def read_lines(file):
    lines = linecache.getlines(os.path.join(args.input, file))
    return lines


if __name__ == "__main__":
    args = parse_args()
    length_ratio = 3.0
    max_length = 1024
    files = os.listdir(args.input)
    lines = []
    langs = []
    en_lines = None
    N = 4000000
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    #files = files[:2] + ["train.en"]
    for file in files:
        if file.split('.')[-1] == "en":
            en_lines = linecache.getlines(os.path.join(args.input, file))
            en_w = open(os.path.join(args.output, file), "w", encoding="utf-8")
            assert len(en_lines) == N
        else:
            langs.append(file.split('.')[-1])
            lines.append(linecache.getlines(os.path.join(args.input, file)))
            assert len(lines[-1]) == N
        print("Successfully loading from {} | {} lines".format(os.path.join(args.input, file), len(lines[-1])))

    files = list(filter(lambda x: "en" not in x, files))
    print("FILES: {}".format(files))
    w_list = [open(os.path.join(args.output, file), "w", encoding="utf-8") for file in files]
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
        if count % 100000 == 0:
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


        if len(en_lines[i].split()) > max_length:
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
