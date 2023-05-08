import os
import argparse
import chardet
import random
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(',')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', '-src', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/spm/afen/train.af-en.af', help='input src')
    parser.add_argument('--tgt', '-tgt', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/spm/afen/train.af-en.en', help='input tgt')
    parser.add_argument('--new-src', '-new-src', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/Filter_v1/spm/afen/train.af-en.af', help='output src')
    parser.add_argument('--new-tgt', '-new-tgt', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/Filter_v1/spm/afen/train.af-en.en', help='output tgt')
    args = parser.parse_args()
    return args

def decode(x: str) -> str:
    return x.replace(" ", "").replace("\u2581", " ").strip()


if __name__ == "__main__":
    args = parse_args()
    length_ratio = 2.0
    max_length = 1024
    with open(args.src, "r", encoding="utf-8") as src_r:
        with open(args.tgt, "r", encoding="utf-8") as tgt_r:
            if not os.path.exists(os.path.dirname(args.new_src)):
                os.makedirs(os.path.dirname(args.new_src))
            src_lang = args.src.split('.')[-1]
            tgt_lang = args.tgt.split('.')[-1]
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
                        if count % 100000 == 0:
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

                        src_w.write("{}".format(src_line))
                        tgt_w.write("{}".format(tgt_line))
                        remaining_count += 1
                    print("\nResults: {} | removing {} examples (unk) | removing {} examples (length > {}) | removing {} examples (length ratio > {}) | removing {} examples (blank) | removing {} examples (latin)".format(remaining_count, unk_rm_count, max_length_rm_count, max_length, length_ratio_rm_count, length_ratio, blank_rm_count, latin_rm_count))
