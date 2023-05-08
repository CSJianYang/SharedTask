import argparse
import os
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(',')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang-pairs', '-lang-pairs', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/split80/train0/', help='input stream')
    parser.add_argument('--data-bin-lang-pairs', '-bt-lang-pairs', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/split-data-bin80/data-bin0/', help='input stream')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    files = os.listdir(args.lang_pairs)
    files.sort()
    lang_pairs = []
    for file in files:
        if file.split('.')[-2] not in lang_pairs:
            src, tgt = file.split('.')[-2].split('-')
            if src not in LANGS or tgt not in LANGS:
                print("LANGS don not contain language {} or {}!".format(src, tgt))
                os.remove(os.path.join(args.lang_pairs, file))
                print("Removing {}...".format(os.path.join(args.lang_pairs, file)))
                continue
            if os.path.getsize(os.path.join(args.lang_pairs, file)) > 0:
                lang_pairs.append("{}-{}".format(src, tgt))
                lang_pairs.append("{}-{}".format(tgt, src))
            else:
                print("{} is empty !".format(file))
                os.remove(os.path.join(args.lang_pairs, file))
                print("Removing {}...".format(os.path.join(args.lang_pairs, file)))
    #lang_pairs.sort()
    lang_pairs = ",".join(lang_pairs)
    print(lang_pairs)

    data_bin_files = os.listdir(args.data_bin_lang_pairs)
    data_bin_files.sort()
    data_bin_lang_pairs = []
    for data_bin_file in data_bin_files:
        if "train" in data_bin_file:
            if data_bin_file.split('.')[-2] not in data_bin_lang_pairs:
                src, tgt = data_bin_file.split('.')[-3].split('-')
                if src not in LANGS or tgt not in LANGS:
                    print("LANGS don not contain language {} or {}!".format(src, tgt))
                    continue
                if os.path.getsize(os.path.join(args.data_bin_lang_pairs, data_bin_file)) > 0:
                    data_bin_lang_pairs.append("{}-{}".format(src, tgt))
                    data_bin_lang_pairs.append("{}-{}".format(tgt, src))
                else:
                    print("{} is empty !".format(file))
    # lang_pairs.sort()
    data_bin_lang_pairs = ",".join(data_bin_lang_pairs)
    print(data_bin_lang_pairs)

    print(set(lang_pairs.split(',')) - set(data_bin_lang_pairs.split(',')))

