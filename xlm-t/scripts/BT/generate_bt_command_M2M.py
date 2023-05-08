import os
import argparse
import sentencepiece as spm
def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )
ISO2M100 = mapping(
    """
afr:af,ara:ar,ast:ast,ben:bn,bos:bs,bul:bg,
cat:ca,ces:cs,ckb:ku,cym:cy,dan:da,deu:de,ell:el,eng:en,est:et,
fas:fa,fin:fi,fra:fr,glg:gl,heb:he,hin:hi,
hrv:hr,hun:hu,hye:hy,ind:id,isl:is,ita:it,jav:jv,jpn:ja,kea:kea,khm:km,kir:ky,kor:ko,lav:lv,
lit:lt,ltz:lb,luo:luo,mal:ml,mar:mr,mkd:mk,mlt:mt,mon:mn,mri:mi,
msa:ms,mya:my,nld:nl,nob:no,nya:ny,oci:oc,
pan:pa,pol:pl,por:pt,pus:ps,ron:ro,rus:ru,slk:sk,slv:sl,sna:sn,snd:sd,
som:so,spa:es,srp:sr,swe:sv,swh:sw,tam:ta,tgl:tl,tha:th,
tur:tr,ukr:uk,umb:umb,urd:ur,vie:vi,zho_simp:zh,
"""
)
LANGS="af,am,ar,as,ast,ay,az,ba,be,bg,bn,br,bs,ca,ceb,cjk,cs,cy,da,de,dyu,el,en,es,et,fa,ff,fi,fr,fy,ga,gd,gl,gu,ha,he,hi,hr,ht,hu,hy,id,ig,ilo,is,it,ja,jv,ka,kac,kam,kea,kg,kk,km,kmb,kmr,kn,ko,ku,ky,lb,lg,ln,lo,lt,luo,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,no,ns,ny,oc,om,or,pa,pl,ps,pt,qu,ro,ru,sd,shn,si,sk,sl,sn,so,sq,sr,ss,su,sv,sw,ta,te,tg,th,ti,tl,tn,tr,uk,umb,ur,uz,vi,wo,xh,yi,yo,zh,zu".split(",")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', '-input-dir', type=str,
                        default=r'/mnt/input/SharedTask/large-scale/MonolingualData/all_spm_split/', help='input stream')
    parser.add_argument('--output-dir', '-output-dir', type=str,
                        default=r'/mnt/input/SharedTask/large-scale/MonolingualData/all_spm_split_bt/', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cmds = ""
    files = os.listdir(args.input_dir)
    BEAM = 2
    MODEL = "/mnt/input/SharedTask/large-scale/PretrainedModel/mm100_615M/flores101_mm100_615M/model.pt"
    BATCH_SIZE = 64
    print("Command Start:")
    count = 0
    for file in files:
        input = file
        src = input.split('.')[0]
        index = input.split('.')[-1][-3:]
        for tgt in LANGS:
            if "{}-{}".format(file, "{}2{}".format(src, tgt)) == "en.tsv.filt001-en2el":
                print(file)
            if src == "en" and src != tgt:
                output = "{}{}.2{}".format(src, index, tgt)
                cmd = "- name: {}-{}\n  sku: G1\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/BT/translate.sh {} {} {} {} {} {} {}\n".format(file, "{}2{}".format(src, tgt), src, tgt, BEAM, MODEL, input, output, BATCH_SIZE)
                count += 1
            elif src != "en" and tgt == "en":
                output = "{}{}.2{}".format(src, index, tgt)
                cmd = "- name: {}-{}\n  sku: G1\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/BT/translate.sh {} {} {} {} {} {} {}\n".format(file, "{}2{}".format(src, tgt), src, tgt, BEAM, MODEL, input, output, BATCH_SIZE)
                count += 1
            else:
                continue
            cmds += cmd
    with open("/home/v-jiaya/SharedTask/xlm-t/translate.txt", "w", encoding="utf-8") as w:
        w.write(cmds)
    print(cmds)
