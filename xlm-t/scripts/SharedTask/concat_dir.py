import os
import argparse
import chardet
LANGS="af, am, an, ar, as, az, be, bg, bn, br, bs, ca, cs, cy, da, de, dz, el, en, eo, es, et, eu, fa, fi, fo, fr, ga, gl, gu, he, hi, hr, ht, hu, hy, id, is, it, ja, jv, ka, kk, km, kn, ko, ku, ky, la, lb, lo, lt, lv, mg, mk, ml, mn, mr, ms, mt, nb, ne, nl, nn, no, oc, or, pa, pl, ps, pt, qu, ro, ru, rw, se, si, sk, sl, sq, sr, sv, sw, ta, te, th, tl, tr, ug, uk, ur, vi, vo, wa, xh, zh, zu".split(', ')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/mnt/input/SharedTask/large-scale/ZC50-Updated/orig/ptb/', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/mnt/input/SharedTask/large-scale/ZC50-Updated/train/', help='input stream')
    parser.add_argument('--src', '-src', type=str,
                        default=r'ptb', help='input stream')
    parser.add_argument('--tgt', '-tgt', type=str,
                        default=r'enu', help='input stream')
    parser.add_argument('--new-src', '-new-src', type=str,
                        default=r'pt', help='input stream')
    parser.add_argument('--new-tgt', '-new-tgt', type=str,
                        default=r'en', help='input stream')
    args = parser.parse_args()
    return args

def decode(x: str) -> str:
    return x.replace(" ", "").replace("\u2581", " ").strip()

if __name__ == "__main__":
    args = parse_args()
    dirs = os.listdir(args.input)
    with open(os.path.join(args.output, "train.{}-{}.{}".format(args.new_src, args.new_tgt, args.new_src)), "w", encoding="utf-8") as src_w:
        for dir in dirs:
            files = os.listdir(os.path.join(args.input, dir))
            files.sort()
            files = list(filter(lambda x: "{}.snt".format(args.src) in x, files))
            for file in files:
                with open(os.path.join(args.input, dir, file), 'rb') as r:
                    rawdata = r.readline()
                    encoding = chardet.detect(rawdata)['encoding']
                    if encoding != "UTF-16":
                        encoding = "UTF-8"
                with open(os.path.join(args.input, dir, file), "r", encoding=encoding) as src_r:
                    src_w.write(src_r.read())
                    src_w.flush()
                print("Reading {} | {}".format(os.path.join(args.input, dir, file), encoding))


    with open(os.path.join(args.output, "train.{}-{}.{}".format(args.new_src, args.new_tgt, args.new_tgt)), "w", encoding="utf-8") as tgt_w:
        for dir in dirs:
            files = os.listdir(os.path.join(args.input, dir))
            files.sort()
            files = list(filter(lambda x: "{}.snt".format(args.tgt) in x, files))
            for file in files:
                with open(os.path.join(args.input, dir, file), 'rb') as r:
                    rawdata = r.readline()
                    encoding = chardet.detect(rawdata)['encoding']
                    if encoding != "UTF-16":
                        encoding = "UTF-8"
                with open(os.path.join(args.input, dir, file), "r", encoding=encoding) as tgt_r:
                    tgt_w.write(tgt_r.read())
                    tgt_w.flush()
                print("Reading {} | {}".format(os.path.join(args.input, dir, file), encoding))