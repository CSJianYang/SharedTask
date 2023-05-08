import argparse
import os
BAD_SRC_LANGS="".split(',') #en
BAD_TGT_LANGS="be,ff,ku,lo,my,ns,om,or,kea,luo".split(',')
BAD_bt_lang_pairs="".split(',')


LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(',')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bt-lang-pairs', '-bt-lang-pairs', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/bt_split80/train0/', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/home/v-jiaya/SharedTask/xlm-t/shells/aml/multi-node/large_task/deltalm/bt_lang_pairs.txt', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    files = os.listdir(args.bt_lang_pairs)
    files.sort()
    bt_lang_pairs = []
    for file in files:
        #if os.path.getsize(os.path.join(args.bt_lang_pairs, file)) > 0:
        if file.split('.')[-2] not in bt_lang_pairs:
            src, tgt = file.split('.')[-2].split('-')
            if src not in LANGS or tgt not in LANGS:
                print("LANGS don not contain language {} or {}!".format(src, tgt))
                continue
            if os.path.getsize(os.path.join(args.bt_lang_pairs, file)) > 0:
                if src in BAD_bt_lang_pairs or tgt in BAD_bt_lang_pairs:
                    print("BAD PAIR: {}-{}".format(src, tgt))
                    continue

                if src in BAD_SRC_LANGS or tgt in BAD_TGT_LANGS:
                    print("BAD SRC: {}-{}".format(src, tgt))
                else:
                    bt_lang_pairs.append("{}-{}".format(src, tgt))

                if tgt in BAD_SRC_LANGS or src in BAD_TGT_LANGS:
                    print("BAD TGT: {}-{}".format(tgt, src))
                else:
                    bt_lang_pairs.append("{}-{}".format(tgt, src))
            else:
                print("{} is empty !".format(file))

    bt_lang_pairs = ",".join(bt_lang_pairs)
    print(bt_lang_pairs)
    N = 80
    DATA_BIN = ":".join(["$BT_TEXT/data-bin{}".format(i) for i in range(N)])
    print(DATA_BIN)

    with open(args.output, "w", encoding="utf-8") as w:
        w.write(bt_lang_pairs)
        print("Saving to {}".format(args.output))
