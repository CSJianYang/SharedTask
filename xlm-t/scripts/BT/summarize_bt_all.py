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
    parser.add_argument('--dd-input-dir', '-dd-input-dir', type=str,
                        default=r'/mnt/input/SharedTask/thunder/BT/4M/', help='input stream')
    parser.add_argument('--yj-input-dir', '-yj-input-dir', type=str,
                        default=r'/mnt/input/SharedTask/large-scale/MonolingualData/4M/', help='input stream')
    parser.add_argument('--output-dir', '-output-dir', type=str,
                        default=r'/mnt/input/SharedTask/thunder/BT/4M-all/', help='input stream')
    args = parser.parse_args()
    return args


def filter(data):
    filter_data = {}
    MAX_LENGTH = 512
    remove_index = []
    for i in range(len(data['en'])):
        if len(data['en'][i].split()) > MAX_LENGTH:
            remove_index.append(i)

    for lang in VALIDBLEU_ISO2M100.values():
        filter_data[lang] = [data[lang][i] for i in remove_index]

    print("Remaining {} lines".format(len(filter_data['en'])))
    return filter_data


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    data = {}

    files = os.listdir(args.dd_input_dir)
    files.sort()
    for file in files:
        with open(os.path.join(args.dd_input_dir, file), "r", encoding="utf-8") as r:
            lang = file.split('.')[-1]
            if lang in VALIDBLEU_ISO2M100.values():
                data[lang] = r.readlines()
                assert len(data[lang]) == 4000000, "{} | {}".format(lang, len(data[lang]))
                print("Successfully Load {} from {} | {} lines".format(file, args.dd_input_dir, len(data[lang])))
                print(data[lang][0].strip())
                print(data[lang][-2].strip())
                print(data[lang][-1].strip())


    files = os.listdir(args.yj_input_dir)
    files.sort()
    for file in files:
        with open(os.path.join(args.yj_input_dir, file), "r", encoding="utf-8") as r:
            lang = file.split('.')[-1]
            if lang in VALIDBLEU_ISO2M100.values() and lang not in data.keys():
                data[lang] = r.readlines()
                assert len(data[lang]) == 4000000, "{} | {}".format(lang, len(data[lang]))
                print("Successfully Load {} from {} | {} lines".format(file, args.yj_input_dir, len(data[lang])))
                print(data[lang][0].strip())
                print(data[lang][-2].strip())
                print(data[lang][-1].strip())


    print("{} Parallel Languages".format(len(data)))
    for lang in data.keys():
        with open(os.path.join(args.output_dir, "train.{}".format(lang)), "w", encoding="utf-8") as w:
            w.write("".join(data[lang]))
        print("Saving to {}...".format(os.path.join(args.output_dir, "train.{}".format(lang))))

