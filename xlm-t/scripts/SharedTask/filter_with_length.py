import os
import argparse
import chardet
import random
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(',')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', '-src', type=str,
                        default=r'/mnt/input/SharedTask/thunder/MonolingualData/all_spm_e2x_bt/train.en-sr.en', help='input src')
    parser.add_argument('--tgt', '-tgt', type=str,
                        default=r'/mnt/input/SharedTask/thunder/MonolingualData/all_spm_e2x_bt/train.en-sr.sr', help='input tgt')
    parser.add_argument('--new-src', '-new-src', type=str,
                        default=r'/home/v-jiaya/tmp/train.en-sr.en', help='output src')
    parser.add_argument('--new-tgt', '-new-tgt', type=str,
                        default=r'/home/v-jiaya/tmp/train.en-sr.sr', help='output tgt')
    parser.add_argument('--length-ratio', '-length-ratio', type=float,
                        default=2.0, help='output tgt')
    args = parser.parse_args()
    return args

def decode(x: str) -> str:
    return x.replace(" ", "").replace("\u2581", " ").strip()


def is_latin(input_str):
    cyrillic="АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгдежзийклмнопрстуфхцчшщьюя"
    latin="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    for c in input_str:
        if c in cyrillic:
            return False
        elif c in latin:
            return True
    return False

def is_latin_complex(input_str):
    cyrillic="АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯабвгдежзийклмнопрстуфхцчшщьюя"
    latin="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    cyrillic_num = 0
    latin_num = 0
    for c in input_str:
        if c in cyrillic:
            cyrillic_num += 1
        elif c in latin:
            latin_num += 1
    if latin_num >= cyrillic_num:
        return True
    else:
        return False


if __name__ == "__main__":
    args = parse_args()
    length_ratio = args.length_ratio
    max_length = 512
    with open(args.src, 'rb') as r:
        rawdata = r.readline()
        src_encoding = chardet.detect(rawdata)['encoding']
        if src_encoding != "UTF-16":
            src_encoding = "UTF-8"
        print("Detecting Source Encoding: {}".format(src_encoding))
    with open(args.tgt, 'rb') as r:
        rawdata = r.readline()
        tgt_encoding = chardet.detect(rawdata)['encoding']
        if tgt_encoding != "UTF-16":
            tgt_encoding = "UTF-8"
        print("Detecting Target Encoding: {}".format(tgt_encoding))
    with open(args.src, "r", encoding=src_encoding) as src_r:
        with open(args.tgt, "r", encoding=tgt_encoding) as tgt_r:
            if not os.path.exists(os.path.dirname(args.new_src)):
                os.makedirs(os.path.dirname(args.new_src))
            src_lang = args.src.split('.')[-1]
            tgt_lang = args.tgt.split('.')[-1]
            #assert src_lang in LANGS and tgt_lang in LANGS, "{} | {}".format(src_lang, tgt_lang)
            with open(args.new_src, "w", encoding="utf-8") as src_w:
                with open(args.new_tgt, "w", encoding="utf-8") as tgt_w:
                    remaining_count = 0
                    count = 0
                    rm_count = 0
                    unk_rm_count = 0
                    blank_rm_count = 0
                    length_ratio_rm_count = 0
                    max_length_rm_count = 0
                    latin_rm_count = 0

                    src_lines = src_r.readlines()
                    tgt_lines = tgt_r.readlines()
                    assert len(src_lines) == len(tgt_lines)
                    print("Start shuffling pairs...")
                    all_lines = list(zip(src_lines, tgt_lines))
                    random.shuffle(all_lines)
                    src_lines = [line[0] for line in all_lines]
                    tgt_lines = [line[1] for line in all_lines]
                    print("Complete shuffling pairs...")
                    print("Complete Reading from {}...".format(args.src))
                    print("Complete Reading from {}...".format(args.tgt))
                    for i, (src_line, tgt_line) in enumerate(zip(src_lines, tgt_lines)):
                        detok_src = src_line.strip()
                        detok_tgt = tgt_line.strip()
                        if count % 1000000 == 0:
                            print("Complete processing {} examples | removing {} examples (unk) | removing {} examples (length > {}) | removing {} examples (length ratio > {}) | removing {} examples (blank) | removing {} examples (latin)".format(count, unk_rm_count, max_length_rm_count, max_length, length_ratio_rm_count, length_ratio, blank_rm_count, latin_rm_count), end="\r")
                        count += 1
                        if "< unk >" in detok_src or "< unk >" in detok_tgt:
                            unk_rm_count += 1
                            rm_count += 1
                            continue
                        if len(detok_tgt.split()) == 0 or len(detok_src.split()) == 0:
                            blank_rm_count += 1
                            rm_count += 1
                            continue
                        if len(detok_src.split()) > max_length or len(detok_tgt.split()) > max_length:
                            max_length_rm_count += 1
                            rm_count += 1
                            continue
                        if len(detok_src.split()) / len(detok_tgt.split()) > length_ratio or len(detok_tgt.split()) / len(detok_src.split()) > length_ratio:
                            length_ratio_rm_count += 1
                            rm_count += 1
                            continue
                        if tgt_lang == "sr" and is_latin(detok_tgt):
                            latin_rm_count += 1
                            continue

                        src_w.write("{}".format(src_line))
                        tgt_w.write("{}".format(tgt_line))
                        src_w.flush()
                        tgt_w.flush()
                        remaining_count += 1
                    print("\nResults: {} | removing {} examples (unk) | removing {} examples (length > {}) | removing {} examples (length ratio > {}) | removing {} examples (blank) | removing {} examples (latin)".format(remaining_count, unk_rm_count, max_length_rm_count, max_length, length_ratio_rm_count, length_ratio, blank_rm_count, latin_rm_count))
