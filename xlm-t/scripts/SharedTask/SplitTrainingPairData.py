import argparse
import os
import math
import random
LANG_CODE={'afr':'af', 'amh':'am', 'ara':'ar', 'asm':'as', 'ast':'ast', 'azj':'az', 'bel':'be', 'ben':'bn', 'bos':'bs', 'bul':'bg', 'cat':'ca', 'ceb':'ceb', 'ces':'cs', 'ckb':'ku', 'cym':'cy', 'dan':'da', 'deu':'de', 'ell':'el', 'eng':'en', 'est': 'et', 'fas':'fa', 'fin':'fi', 'fra':'fr', 'ful':'ff', 'gle':'ga', 'glg':'gl', 'guj':'gu', 'hau':'ha', 'heb':'he', 'hin':'hi', 'hrv':'hr', 'hun':'hu','hye':'hy','ibo':'ig','ind':'id','isl':'is','ita':'it','jav':'jv', 'jpn':'ja', 'kam':'kam', 'kan':'kn', 'kat':'ka', 'kaz':'kk', 'kea':'kea', 'khm':'km', 'kir':'ky', 'kor':'ko', 'lao':'lo', 'lav':'lv', 'lin':'ln', 'lit':'lt', 'ltz':'lb', 'lug':'lg', 'luo':'luo', 'mal':'ml', 'mar':'mr', 'mkd':'mk', 'mlt':'mt', 'mon':'mn', 'mri':'mi', 'msa':'ms','mya':'my','nld':'nl','nob':'no','npi':'ne','nso':'ns','nya':'ny','oci':'oc','orm':'om', 'ory':'or', 'pan':'pa', 'pol':'pl', 'por':'pt', 'pus':'ps', 'ron':'ro', 'rus':'ru', 'slk':'sk', 'slv':'sl', 'sna':'sn', 'snd':'sd', 'som':'so', 'spa':'es', 'srp':'sr', 'swe':'sv', 'swh':'sw', 'tam':'ta', 'tel':'te', 'tgk':'tg', 'tgl':'tl', 'tha':'th', 'tur':'tr', 'ukr':'uk', 'umb':'umb', 'urd':'ur', 'uzb':'uz', 'vie':'vi', 'wol':'wo', 'xho':'xh', 'yor':'yo', 'zho_simpl':'zh', 'zho_trad':'zt', 'zul':'zu'}
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-input', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/bt_spm/lven/', help='input stream')
    parser.add_argument('--output', '-output', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/bt_split80/', help='input stream')
    parser.add_argument('--split-num', '-split-num', type=int,
                        default=80, help='input stream')
    parser.add_argument('--shuffle', '-shuffle', action="store_true", help='input stream')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    if os.path.isdir(args.input):
        length_ratio = 2.5
        files = os.listdir(args.input)
        assert len(files) == 2
        with open(os.path.join(args.input, files[0]), "r", encoding="utf-8") as src_r:
            with open(os.path.join(args.input, files[1]), "r", encoding="utf-8") as tgt_r:
                src_lines = src_r.readlines()
                tgt_lines = tgt_r.readlines()
                pair = list(zip(src_lines, tgt_lines))
                pair = filter(lambda x: x[0]!="" and x[1]!="" and len(x[0].split())/len(x[1].split(0)) < length_ratio and len(x[1].split())/len(x[0].split(0)) < length_ratio, pair)
                random.shuffle(pair)
                src_lines = [item[0] for item in pair]
                tgt_lines = [item[1] for item in pair]
                assert len(src_lines) == len(tgt_lines)
                for i in range(args.split_num):
                    chunk_num = math.ceil(len(src_lines)/args.split_num)
                    print("Chunk Num: {}".format(chunk_num))
                    output_dir = os.path.join(args.output, "train{}".format(i))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    with open(os.path.join(output_dir, files[0]), "w", encoding="utf-8") as src_w:
                        with open(os.path.join(output_dir, files[1]), "w", encoding="utf-8") as tgt_w:
                            start_index = i * chunk_num
                            end_index = (i + 1) *chunk_num
                            print("Successfully saving to {}: {} lines".format(os.path.join(output_dir, files[0]), len(src_lines[start_index: end_index])))
                            print("Successfully saving to {}: {} lines".format(os.path.join(output_dir, files[1]), len(src_lines[start_index: end_index])))
                            chunk_num_limit = 100
                            if chunk_num <= chunk_num_limit:
                                print("#Chunk Num < {} ! Will Save all {} lines...".format(chunk_num_limit, len(src_lines)))
                                src_w.write("".join(src_lines))
                                tgt_w.write("".join(tgt_lines))
                            else:
                                src_w.write("".join(src_lines[start_index: end_index]))
                                tgt_w.write("".join(tgt_lines[start_index: end_index]))
                del src_lines
                del tgt_lines
    else:
        with open(os.path.join(args.input), "r", encoding="utf-8") as r:
            file = os.path.basename(args.input)
            lines = r.readlines()
            for i in range(args.split_num):
                chunk_num = math.ceil(len(lines) / args.split_num)
                print("Chunk Num: {}".format(chunk_num))
                output_dir = os.path.join(args.output, "train{}".format(i))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                with open(os.path.join(output_dir, file), "w", encoding="utf-8") as w:
                    start_index = i * chunk_num
                    end_index = (i + 1) * chunk_num
                    w.write("".join(lines[start_index: end_index]))
                    print("Successfully saving to {}: {} lines".format(os.path.join(output_dir, file), len(lines[start_index: end_index])))
