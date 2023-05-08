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
msa:ms,nld:nl,nob:no,nya:ny,oci:oc,pol:pl,por:pt,pus:ps,ron:ro,rus:ru,slk:sk,slv:sl,
spa:es,srp:sr,swe:sv,swh:sw,tam:ta,tgl:tl,tha:th,
tur:tr,ukr:uk,urd:ur,vie:vi,zho_simp:zh
"""
)
BAD_SRC_LANGS="en".split(',') #en
BAD_TGT_LANGS="af,ar,ast,bg,bs,ca,cs,da,hy,ky,sl,sk,sn,fr,de,ta,hr,ml,mr,ms,vi,mt,mn,mk".split(',')
BAD_LANG_PAIRS="".split(',')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-output', type=str,
                        default=r'/mnt/input/SharedTask/thunder/PretrainedModel/deltalm/large-postnorm/parallel_bt_lang_pairs.txt', help='input stream')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    lang_pairs = []
    for src in VALIDBLEU_ISO2M100.values():
        for tgt in VALIDBLEU_ISO2M100.values():
            if src != tgt:
                if src in BAD_SRC_LANGS:
                    print("BAD SRC: {}-{}".format(src, tgt))
                    continue
                if tgt in BAD_TGT_LANGS:
                    print("BAD TGT: {}-{}".format(src, tgt))
                    continue
                if src in BAD_LANG_PAIRS or tgt in BAD_LANG_PAIRS:
                    print("BAD PAIR: {}-{}".format(src, tgt))
                    continue
                lang_pair = "{}-{}".format(src, tgt)
                lang_pairs.append(lang_pair)
    print("{} lang pairs".format(len(lang_pairs)))
    print(",".join(lang_pairs))
    with open(args.output, "w", encoding="utf-8") as w:
        w.write(",".join(lang_pairs))
        print("Saving to {}".format(args.output))


