import langid
import os
import argparse
from transliterate import detect_language
import sentencepiece as spm
LANGS="af, am, an, ar, as, az, be, bg, bn, br, bs, ca, cs, cy, da, de, dz, el, en, eo, es, et, eu, fa, fi, fo, fr, ga, gl, gu, he, hi, hr, ht, hu, hy, id, is, it, ja, jv, ka, kk, km, kn, ko, ku, ky, la, lb, lo, lt, lv, mg, mk, ml, mn, mr, ms, mt, nb, ne, nl, nn, no, oc, or, pa, pl, ps, pt, qu, ro, ru, rw, se, si, sk, sl, sq, sr, sv, sw, ta, te, th, tl, tr, ug, uk, ur, vi, vo, wa, xh, zh, zu".split(', ')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', '-src', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/thunder/small_task1/train/train.en-sr.en', help='input stream')
    parser.add_argument('--tgt', '-tgt', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/thunder/small_task1/train/train.en-sr.sr', help='input stream')
    parser.add_argument('--new-src', '-new-src', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/thunder/small_task1/train-filter/test.en', help='input stream')
    parser.add_argument('--new-tgt', '-new-tgt', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/thunder/small_task1/train-filter/test.sr', help='input stream')
    args = parser.parse_args()
    return args

def decode(x: str) -> str:
    return x.replace(" ", "").replace("\u2581", " ").strip()

if __name__ == "__main__":
    args = parse_args()
    length_ratio = 2.5
    with open(args.src, "r", encoding="utf-8") as src_r:
        with open(args.tgt, "r", encoding="utf-8") as tgt_r:
            if not os.path.exists(os.path.dirname(args.new_src)):
                os.makedirs(os.path.dirname(args.new_src))
            src_lang = args.src.split('.')[-1]
            tgt_lang = args.tgt.split('.')[-1]
            assert src_lang in LANGS and tgt_lang in LANGS, "{} | {}".format(src_lang, tgt_lang)
            with open(args.new_src, "w", encoding="utf-8") as src_w:
                with open(args.new_tgt, "w", encoding="utf-8") as tgt_w:
                    #src_lines = src_r.readlines()
                    #tgt_lines = tgt_r.readlines()
                    count = 0
                    rm_count = 0
                    for i, (src_line, tgt_line) in enumerate(zip(src_r, tgt_r)):
                        detok_src = decode(src_line.strip())
                        detok_tgt = decode(tgt_line.strip())
                        if count % 1000 == 0:
                            print(u"Complete processing {} examples | removing {} examples".format(count, rm_count), end="\r")
                        count += 1
                        if len(detok_tgt.split()) == 0 or len(detok_src.split()) == 0:
                            rm_count += 1
                            continue
                        if len(detok_src.split()) / len(detok_tgt.split()) > length_ratio or len(detok_tgt.split()) / len(detok_src.split()) > length_ratio:
                            rm_count += 1
                            continue
                        #infer_lang = detect_language(detok_tgt)
                        infer_lang = langid.classify(detok_tgt)[0]
                        if infer_lang != tgt_lang: #langid.classify(detok_src)[0] != src_lang or
                            rm_count += 1
                            continue
                        src_w.write("{}".format(src_line))
                        tgt_w.write("{}".format(tgt_line))
                        src_w.flush()
                        tgt_w.flush()
                    print("Removing {} examples || Total: {}".format(rm_count, count))