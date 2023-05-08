import langid
import os
import argparse
import sentencepiece as spm
LANGS="af, am, an, ar, as, az, be, bg, bn, br, bs, ca, cs, cy, da, de, dz, el, en, eo, es, et, eu, fa, fi, fo, fr, ga, gl, gu, he, hi, hr, ht, hu, hy, id, is, it, ja, jv, ka, kk, km, kn, ko, ku, ky, la, lb, lo, lt, lv, mg, mk, ml, mn, mr, ms, mt, nb, ne, nl, nn, no, oc, or, pa, pl, ps, pt, qu, ro, ru, rw, se, si, sk, sl, sq, sr, sv, sw, ta, te, th, tl, tr, ug, uk, ur, vi, vo, wa, xh, zh, zu".split(', ')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'', help='input stream')
    parser.add_argument('--lang-pair', '-lang-pair', type=str,
                        default=r'', help='input stream')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    length_ratio = 2.0
    if os.path.isdir(args.input):
        files = os.listdir(args.input)
        files.sort()
        files = list(filter(lambda x: args.lang_pair in x, files))
        for file in files:
            with open(os.path.join(args.input, file), "r", encoding="utf-8") as src_r:
                lang = file.split('.')[-1]
                assert lang in LANGS
                count = 0
                rm_count = 0
                lines = src_r.readlines()
                lang_dict = {}
                for i, src_line in enumerate(lines):
                    detok_src = src_line.strip()
                    if count % 1000000 == 0:
                       print("Complete processing {} examples".format(count))
                    count += 1
                    infer_lang = langid.classify(detok_src)[0]
                    if infer_lang not in lang_dict.keys():
                        lang_dict[infer_lang] = 0
                    else:
                        lang_dict[infer_lang] += 1
                    if infer_lang != lang:
                        rm_count += 1
                        continue
                print(lang_dict)
                for key in lang_dict:
                    if lang_dict[key] == max(lang_dict.values()):
                        print("Infer {} dataset".format(key))
                        if key != lang:
                            print("WARNING: PLEASE ENSURE LANGUAGE !")
                print("{} || Removing {} examples || Total: {}".format(file, rm_count, count))
    else:
        with open(args.input, "r", encoding="utf-8") as src_r:
            lang = args.input.split('.')[-1]
            assert lang in LANGS, "lang"
            count = 0
            rm_count = 0
            lines = src_r.readlines()
            lang_dict = {}
            for i, src_line in enumerate(lines):
                detok_src = src_line.strip()
                if count % 1000000 == 0:
                   print("Complete processing {} examples".format(count))
                count += 1
                infer_lang = langid.classify(detok_src)[0]
                if infer_lang not in lang_dict.keys():
                    lang_dict[infer_lang] = 0
                else:
                    lang_dict[infer_lang] += 1
                if infer_lang != lang:
                    rm_count += 1
                    continue
            print(lang_dict)
            for key in lang_dict:
                if lang_dict[key] == max(lang_dict.values()):
                    print("Infer {} dataset".format(key))
                    if key != lang:
                        print("WARNING: PLEASE ENSURE LANGUAGE !")
            print("{} || Removing {} examples || Total: {}".format(args.input, rm_count, count))