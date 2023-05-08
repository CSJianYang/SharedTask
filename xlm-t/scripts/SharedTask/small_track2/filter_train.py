# coding=utf-8
import langid
import os
import argparse
import sentencepiece as spm
LANGS="af, am, an, ar, as, az, be, bg, bn, br, bs, ca, cs, cy, da, de, dz, el, en, eo, es, et, eu, fa, fi, fo, fr, ga, gl, gu, he, hi, hr, ht, hu, hy, id, is, it, ja, jv, ka, kk, km, kn, ko, ku, ky, la, lb, lo, lt, lv, mg, mk, ml, mn, mr, ms, mt, nb, ne, nl, nn, no, oc, or, pa, pl, ps, pt, qu, ro, ru, rw, se, si, sk, sl, sq, sr, sv, sw, ta, te, th, tl, tr, ug, uk, ur, vi, vo, wa, xh, zh, zu".split(', ')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', '-src', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/thunder/small_task2/download/train/', help='input stream')
    parser.add_argument('--tgt', '-tgt', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/thunder/small_task2/download/train/', help='input stream')
    parser.add_argument('--new-src', '-new-src', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/thunder/small_task2/Filter_v1/train/', help='input stream')
    parser.add_argument('--new-tgt', '-new-tgt', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/thunder/small_task2/Filter_v1/train/', help='input stream')
    args = parser.parse_args()
    return args

def decode(x: str) -> str:
    return x.replace(" ", "").replace("\u2581", " ").strip()


if __name__ == "__main__":
    args = parse_args()
    length_ratio = 3.0
    if os.path.isdir(args.src) and args.src == args.tgt:
        files = os.listdir(args.src)
        pairs = set([file.split('.')[-2] for file in files])
        if not os.path.exists(args.new_src):
            os.makedirs(args.new_src)
        for pair in pairs:
            print(pair)
            src_lang, tgt_lang = pair.split('-')
            with open(os.path.join(args.src, "train.{}.{}".format(pair, src_lang)), "r", encoding="utf-8") as src_r:
                with open(os.path.join(args.tgt, "train.{}.{}".format(pair, tgt_lang)), "r", encoding="utf-8") as tgt_r:
                    assert src_lang in LANGS and tgt_lang in LANGS, "{} | {}".format(src_lang, tgt_lang)
                    with open(os.path.join(args.new_src, "train.{}.{}".format(pair, src_lang)), "w", encoding="utf-8") as src_w:
                        with open(os.path.join(args.new_tgt, "train.{}.{}".format(pair, tgt_lang)), "w", encoding="utf-8") as tgt_w:
                            count = 0
                            blank_rm_count = 0
                            length_ratio_rm_count = 0
                            latin_rm_count = 0
                            remaining_count = 0
                            for i, (src_line, tgt_line) in enumerate(zip(src_r, tgt_r)):
                                tok_src = src_line.strip()
                                tok_tgt = tgt_line.strip()
                                detok_src = decode(tok_src)
                                detok_tgt = decode(tok_tgt)
                                if count % 10000 == 0:
                                    print(u"Complete processing {} examples | Removing {} examples (blank) | Removing {} examples (length ratio={}) | Removing {} examples (latin) ".format(count, blank_rm_count, length_ratio_rm_count, length_ratio, latin_rm_count), end="\r")
                                count += 1
                                if len(detok_tgt.split()) == 0 or len(detok_src.split()) == 0:
                                    blank_rm_count += 1
                                    continue
                                if len(tok_src.split()) / len(tok_tgt.split()) > length_ratio or len(tok_tgt.split()) / len(tok_src.split()) > length_ratio:
                                    length_ratio_rm_count += 1
                                    continue
                                src_w.write("{}".format(src_line))
                                tgt_w.write("{}".format(tgt_line))
                                src_w.flush()
                                tgt_w.flush()
                                remaining_count += 1
                            print("\nRemoving {} examples (blank) | Removing {} examples (length ratio) | Removing {} examples (latin) || Results: {}".format(blank_rm_count, length_ratio_rm_count, latin_rm_count, remaining_count))
                            print("{} -> {}".format(os.path.join(args.src, "train.{}.{}".format(pair, src_lang)), os.path.join(args.new_src, "train.{}.{}".format(pair, src_lang))))
                            print("{} -> {}".format(os.path.join(args.tgt, "train.{}.{}".format(pair, tgt_lang)), os.path.join(args.new_tgt, "train.{}.{}".format(pair, tgt_lang))))

    else:
        with open(args.src, "r", encoding="utf-8") as src_r:
            with open(args.tgt, "r", encoding="utf-8") as tgt_r:
                if not os.path.exists(os.path.dirname(args.new_src)):
                    os.makedirs(os.path.dirname(args.new_src))
                src_lang = args.src.split('.')[-1]
                tgt_lang = args.tgt.split('.')[-1]
                assert src_lang in LANGS and tgt_lang in LANGS, "{} | {}".format(src_lang, tgt_lang)
                with open(args.new_src, "w", encoding="utf-8") as src_w:
                    with open(args.new_tgt, "w", encoding="utf-8") as tgt_w:
                        count = 0
                        blank_rm_count = 0
                        length_ratio_rm_count = 0
                        latin_rm_count = 0
                        remaining_count = 0
                        for i, (src_line, tgt_line) in enumerate(zip(src_r, tgt_r)):
                            tok_src = src_line.strip()
                            tok_tgt = tgt_line.strip()
                            detok_src = decode(tok_src)
                            detok_tgt = decode(tok_tgt)
                            if count % 1000 == 0:
                                print(u"Complete processing {} examples | Removing {} examples (blank) | Removing {} examples (length ratio) | Removing {} examples (latin) ".format(count, blank_rm_count, length_ratio_rm_count, latin_rm_count), end="\r")
                            count += 1
                            if len(detok_tgt.split()) == 0 or len(detok_src.split()) == 0:
                                blank_rm_count += 1
                                continue
                            if len(tok_src.split()) / len(tok_tgt.split()) > length_ratio or len(tok_tgt.split()) / len(tok_src.split()) > length_ratio:
                                length_ratio_rm_count += 1
                                continue
                            src_w.write("{}".format(src_line))
                            tgt_w.write("{}".format(tgt_line))
                            src_w.flush()
                            tgt_w.flush()
                            remaining_count += 1
                        print("Removing {} examples (blank) | Removing {} examples (length ratio) | Removing {} examples (latin) || Results: {}".format(blank_rm_count, length_ratio_rm_count, latin_rm_count, remaining_count))
                        print("{} -> {}".format(args.src, args.new_src))
                        print("{} -> {}".format(args.tgt, args.new_tgt))