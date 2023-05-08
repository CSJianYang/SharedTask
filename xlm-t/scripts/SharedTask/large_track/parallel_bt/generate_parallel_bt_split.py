import os
import argparse
def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )
VALIDBLEU_ISO2M100 = mapping(
"""
afr:af,ara:ar,ast:ast,ben:bn,bos:bs,bul:bg,
cat:ca,ces:cs,ckb:ku,cym:cy,dan:da,deu:de,ell:el,est:et,eng:en,
fas:fa,fin:fi,fra:fr,gle:ga,glg:gl,heb:he,hin:hi,
hrv:hr,hun:hu,hye:hy,ind:id,isl:is,ita:it,jav:jv,jpn:ja,kat:ka,khm:km,kir:ky,kor:ko,lav:lv,
lit:lt,ltz:lb,mal:ml,mar:mr,mkd:mk,mlt:mt,mon:mn,mri:mi,
msa:ms,nld:nl,nob:no,nya:ny,oci:oc,pol:pl,por:pt,pus:ps,ron:ro,rus:ru,slk:sk,slv:sl,sna:sn,
som:so,spa:es,srp:sr,swe:sv,swh:sw,tam:ta,tgl:tl,tha:th,
tur:tr,ukr:uk,umb:umb,urd:ur,vie:vi,zho_simp:zh
"""
)
def parse_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    split_num = 80
    cmds = ""
    for lang in VALIDBLEU_ISO2M100.values():
        cmd = "- name: parallel_bt_split_{}\n  sku: G0\n  sku_count: 1\n  command: \n    - bash ./shells/preprocess/large_task/parallel_bt/parallel_bt_split.sh {} {}\n".format(lang, lang, split_num)
        cmds += cmd
    print(cmds)