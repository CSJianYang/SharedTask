import argparse
import os
LANG_CODE={'afr':'af', 'amh':'am', 'ara':'ar', 'asm':'as', 'ast':'ast', 'azj':'az', 'bel':'be', 'ben':'bn', 'bos':'bs', 'bul':'bg', 'cat':'ca', 'ceb':'ceb', 'ces':'cs', 'ckb':'ku', 'cym':'cy', 'dan':'da', 'deu':'de', 'ell':'el', 'eng':'en', 'est': 'et', 'fas':'fa', 'fin':'fi', 'fra':'fr', 'ful':'ff', 'gle':'ga', 'glg':'gl', 'guj':'gu', 'hau':'ha', 'heb':'he', 'hin':'hi', 'hrv':'hr', 'hun':'hu','hye':'hy','ibo':'ig','ind':'id','isl':'is','ita':'it','jav':'jv', 'jpn':'ja', 'kam':'kam', 'kan':'kn', 'kat':'ka', 'kaz':'kk', 'kea':'kea', 'khm':'km', 'kir':'ky', 'kor':'ko', 'lao':'lo', 'lav':'lv', 'lin':'ln', 'lit':'lt', 'ltz':'lb', 'lug':'lg', 'luo':'luo', 'mal':'ml', 'mar':'mr', 'mkd':'mk', 'mlt':'mt', 'mon':'mn', 'mri':'mi', 'msa':'ms','mya':'my','nld':'nl','nob':'no','npi':'ne','nso':'ns','nya':'ny','oci':'oc','orm':'om', 'ory':'or', 'pan':'pa', 'pol':'pl', 'por':'pt', 'pus':'ps', 'ron':'ro', 'rus':'ru', 'slk':'sk', 'slv':'sl', 'sna':'sn', 'snd':'sd', 'som':'so', 'spa':'es', 'srp':'sr', 'swe':'sv', 'swh':'sw', 'tam':'ta', 'tel':'te', 'tgk':'tg', 'tgl':'tl', 'tha':'th', 'tur':'tr', 'ukr':'uk', 'umb':'umb', 'urd':'ur', 'uzb':'uz', 'vie':'vi', 'wol':'wo', 'xho':'xh', 'yor':'yo', 'zho_simpl':'zh', 'zho_trad':'zt', 'zul':'zu'}
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/large-scale/flores101_dataset/devtest/', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/home/v-jiaya/SharedTask/data/large-scale/flores101_dataset/devtest-code/', help='input stream')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    files = os.listdir(args.input)
    for file in files:
        with open(os.path.join(args.input, file), "r", encoding="utf-8") as r:
            if not os.path.exists(args.output):
                os.makedirs(args.output)
            with open(os.path.join(args.output, "valid.{}".format(LANG_CODE[file.replace(".devtest", "").replace(".dev", "")])), "w", encoding="utf-8") as w:
                w.write(r.read())
