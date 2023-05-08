import argparse
import xlwt
import xlrd
import os
from collections import OrderedDict

TOTAL_DIRECTION = 102 * 102 - 102
def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(",")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bitext', '-bitext', type=str,
                        default=r'../results/small_track2_bitext_lines.txt', help='input stream')
    parser.add_argument('--bt', '-bt', type=str,
                        default=r'../results/small_track2_bt_lines.txt', help='input stream')
    parser.add_argument('--parallel-bt', '-parallel-bt', type=str,
                        default=r'../results/small_track2_parallel_bt_lines.txt', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'../results/small_track2.txt', help='input stream')
    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    print(args)
    bitext = {}
    bt = {}
    parallel_bt = {}
    with open(args.bitext, "r", encoding="utf-8") as r:
        lines = r.readlines()
        for line in lines:
            if "train" in line:
                line = line.strip()
                lines_num = line.split()[-2]
                pair = line.split()[-1].split('.')[-2]
                bitext[pair] = int(lines_num)

    with open(args.bt, "r", encoding="utf-8") as r:
        lines = r.readlines()
        for line in lines:
            if "train" in line:
                line = line.strip()
                lines_num = line.split()[-2]
                pair = line.split()[-1].split('.')[-2]
                bt[pair] = int(lines_num)

    with open(args.parallel_bt, "r", encoding="utf-8") as r:
        lines = r.readlines()
        for line in lines:
            if "train" in line:
                line = line.strip()
                lines_num = line.split()[-2]
                pair = line.split()[-1].split('.')[-1]
                parallel_bt[pair] = int(lines_num)

    print(bitext)
    print(bt)
    print(parallel_bt)
    print("BITEXT: {} pairs | {} pairs".format(sum(bitext.values()), len(bitext)))
    print("BT: {} pairs | {} pairs".format(sum(bt.values()), len(bt)))
    print("PARALLEL BT: {} pairs | {} langs".format(list(parallel_bt.values())[0], len(parallel_bt)))
    print("BITEXT: {} sents | {} pairs".format(sum(bitext.values()) * 2, len(bitext)))
    print("BT: {} sents | {} pairs".format(sum(bt.values()) * 2, len(bt)))
    print("PARALLEL BT: {}  sents | {} langs".format(list(parallel_bt.values())[0], len(parallel_bt)))
    with open(args.output, "w", encoding="utf-8") as w:
        w.write("{}\n".format(bitext))
        w.write("{}\n".format(bt))
        w.write("{}\n".format(parallel_bt))







