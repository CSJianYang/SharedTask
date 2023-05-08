import os
import argparse
import sentencepiece as spm
def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )
VALIDBLEU_ISO2M100 = mapping(
"""
afr:af,ara:ar,ast:ast,ben:bn,bos:bs,bul:bg,
cat:ca,ces:cs,ckb:ku,cym:cy,dan:da,deu:de,ell:el,eng:en,est:et,
fas:fa,fin:fi,fra:fr,glg:gl,heb:he,hin:hi,
hrv:hr,hun:hu,hye:hy,ind:id,isl:is,ita:it,jav:jv,jpn:ja,kea:kea,khm:km,kir:ky,kor:ko,lav:lv,
lit:lt,ltz:lb,luo:luo,mal:ml,mar:mr,mkd:mk,mlt:mt,mon:mn,mri:mi,
msa:ms,mya:my,nld:nl,nob:no,nya:ny,oci:oc,
pan:pa,pol:pl,por:pt,pus:ps,ron:ro,rus:ru,slk:sk,slv:sl,sna:sn,snd:sd,
som:so,spa:es,srp:sr,swe:sv,swh:sw,tam:ta,tgl:tl,tha:th,
tur:tr,ukr:uk,umb:umb,urd:ur,vie:vi,zho_simp:zh
"""
)
LANGS="af,am,ar,as,ast,ay,az,ba,be,bg,bn,br,bs,ca,ceb,cjk,cs,cy,da,de,dyu,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kac,kam,kea,kg,kk,km,kmb,kmr,kn,ko,ku,ky,lb,lg,ln,lo,lt,luo,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,no,ns,ny,oc,om,or,pa,pl,ps,pt,qu,ro,ru,sd,shn,si,sk,sl,sn,so,sq,sr,ss,su,sv,sw,ta,te,tg,th,ti,tl,tn,tr,uk,umb,ur,uz,vi,wo,xh,yi,yo,zh,zu".split(",")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yj-input-dir', '-yj-input-dir', type=str,
                        default=r'/mnt/input/SharedTask/large-scale/MonolingualData/all_spm_split_bt/', help='input stream')
    parser.add_argument('--output-dir', '-output-dir', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/thunder/BT/4M/', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cmds = ""
    files = os.listdir(args.yj_input_dir)
    files.sort()
    files = list(filter(lambda x: ".log" not in x, files))
    for file in files:
        with open(os.path.join(args.yj_input_dir, file), "r", encoding="utf-8") as r:
            lang = file.split('.')[-2]
            if lang in VALIDBLEU_ISO2M100.values():
                with open(os.path.join(args.output_dir, "train.{}".format(lang)), "w", encoding="utf-8") as w:
                    w.write(r.read())
                print("{} -> {}".format(os.path.join(args.yj_input_dir, file), os.path.join(args.output_dir, "train.{}".format(lang))))

