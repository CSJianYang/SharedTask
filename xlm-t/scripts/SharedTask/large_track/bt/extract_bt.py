import argparse
import os
import gzip

def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )
DD2OUR = mapping(
"""
afk:af,amh:am,ara:ar,bel:be,bnb:bn,bsb:bs,bgr:bg,
cat:ca,ceb:ceb,chs:zh,cht:zt,hrv:hr,dan:da,nld:nl,enu:en,eti:et,fpo:tl,fin:fi,fra:fr,glc:gl,
kat:ka,deu:de,ell:el,heb:he,hib:hi,hun:hu,isl:is,ind:id,ire:ga,ita:it,jpn:ja,kkz:kk,kor:ko,lvi:lv,lth:lt,
mki:mk,msl:ms,mym:ml,mlt:mt,mri:mi,mar:mr,mnn:mn,nep:ne,nor:no,pas:ps,far:fa,plk:pl,ptb:pt,rom:ro,rus:ru,srb:sr,
sky:sk,slv:sl,kur:ku,esn:es,swh:sw,sve:sv,tam:ta,tel:te,tha:th,trk:tr,ukr:uk,urd:ur,uzb:uz,vit:vi,cym:cy,yor:yo,zul:zu,
cht:zu
"""
)
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(',')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-input-dir', type=str,
                        default=r'/mnt/input/SharedTask/large-scale/BT/for_X2E_WikiData/', help='input stream')
    parser.add_argument('--output-dir', '-output-dir', type=str,
                        default=r'/mnt/input/SharedTask/thunder/BT/for_X2E_WikiData/', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    dirs = os.listdir(args.input_dir)
    for dir in dirs:
        translation = list(filter(lambda x: "enu" not in x and ".gz" in x, os.listdir(os.path.join(args.input_dir, dir, "CreateParallelCorpus"))))[0]
        file = os.path.join(args.input_dir, dir, "CreateParallelCorpus/{}".format(translation))
        f = gzip.open(file, 'rb')
        lang = DD2OUR[dir]
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        with open(os.path.join(args.output_dir, "train.{}".format(lang)), "wb") as w:
            w.write(f.read().strip())
        print("{} -> {}".format(os.path.join(args.input_dir, dir, "CreateParallelCorpus/en.tsv.filt.1.{}.snt.gz".format(dir)), os.path.join(args.output_dir, "train.{}".format(lang))))
