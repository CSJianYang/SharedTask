import os
import argparse
import sentencepiece as spm
def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )
DD2OUR = mapping(
"""
afk:af,amh:am,ara:ar,bel:be,bnb:bn,bsb:bs,bgr:bg,
cat:ca,ceb:ceb,chs:zh,cht:zt,hrv:hr,csy:cs,dan:da,nld:nl,enu:en,eti:et,fpo:tl,fin:fi,fra:fr,glc:gl,
kat:ka,deu:de,ell:el,heb:he,hin:hi,hun:hu,isl:is,ind:id,ire:ga,ita:it,jpn:ja,kkz:kk,kor:ko,lvi:lv,lth:lt,
mki:mk,msl:ms,mym:ml,mlt:mt,mri:mi,mar:mr,mnn:mn,nep:ne,nor:no,pas:ps,far:fa,plk:pl,ptb:pt,rom:ro,rus:ru,srb:sr,
sky:sk,slv:sl,kur:ku,esn:es,swh:sw,sve:sv,tam:ta,tel:te,tha:th,trk:tr,ukr:uk,urd:ur,uzb:uz,vit:vi,cym:cy,yor:yo,zul:zu,
cht:zu
"""
)
LANGS="af,am,ar,as,ast,ay,az,ba,be,bg,bn,br,bs,ca,ceb,cjk,cs,cy,da,de,dyu,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kac,kam,kea,kg,kk,km,kmb,kmr,kn,ko,ku,ky,lb,lg,ln,lo,lt,luo,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,no,ns,ny,oc,om,or,pa,pl,ps,pt,qu,ro,ru,sd,shn,si,sk,sl,sn,so,sq,sr,ss,su,sv,sw,ta,te,tg,th,ti,tl,tn,tr,uk,umb,ur,uz,vi,wo,xh,yi,yo,zh,zu".split(",")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dd-input-dir', '-dd-input-dir', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/thunder/BT/for_XY/', help='input stream')
    parser.add_argument('--yj-input-dir', '-yj-input-dir', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/thunder/BT/for_XY/', help='input stream')
    parser.add_argument('--output-dir', '-output-dir', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/thunder/BT/4M/', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cmds = ""
    files = os.listdir(args.dd_input_dir)
    files.sort()
    for file in files:
        with open(os.path.join(args.dd_input_dir, file), "r", encoding="utf-8") as r:
            lang = DD2OUR[file.split('.')[-2]]
            with open(os.path.join(args.output_dir, "train.{}".format(lang)), "w", encoding="utf-8") as w:
                w.write(r.read())
            print("{} -> {}".format(os.path.join(args.dd_input_dir, file), os.path.join(args.output_dir, "train.{}".format(lang))))

