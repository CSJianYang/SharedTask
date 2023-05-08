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
    parser.add_argument('--log', '-log', type=str,
                        default=r'/mnt/input/SharedTask/thunder/flores101_dataset/BLEU/', help='input stream')
    parser.add_argument('--pivot-log', '-pivot-log', type=str,
                        default=r'/mnt/input/SharedTask/thunder/flores101_dataset/PIVOT-BLEU/', help='input stream')
    parser.add_argument('--checkpoint-name', '-checkpoint-name', type=str,
                        default=r'/mnt/input/SharedTask/thunder/large_track/data/model/deltalm/FULL-v1/lr1e-4-deltalm-postnorm-24L-6L/avg30_39.pt', help='input stream')
    parser.add_argument('--threshold', '-threshold', type=float,
                        default=0, help='input stream')
    parser.add_argument('--pivot-pairs', '-pivot-pairs', type=str,
                        default=r'/mnt/input/SharedTask/thunder/flores101_dataset/pivot_pairs.txt', help='input stream')
    args = parser.parse_args()
    return args


def create_excel(results, name):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet("LargeTrack", cell_overwrite_ok=True)
    worksheet.write(1, 0, label="DeltaLM-Postnorm (Large)")
    for i in range(len(LANGS)):
        worksheet.write(0, i + 1, label=LANGS[i])
        worksheet.write(i + 1, 0, label=LANGS[i])
    for i in range(len(LANGS)):
        for j in range(len(LANGS)):
            worksheet.write(i + 1, j + 1, label=results[i][j])
    save_path = '/mnt/input/SharedTask/{}.xls'.format(name)
    workbook.save(save_path)
    print("Saving to {}".format(save_path))
    return workbook


def _lang_pair(src, tgt):
    return "{}->{}".format(src, tgt)


def read_excel(filename):
    #m2m_results = []
    m2m_x2x = {}
    workbook = xlrd.open_workbook(filename)
    worksheet = workbook.sheets()[0]
    ncols = worksheet.ncols
    nrows = worksheet.nrows
    M2M_LANGS = []
    for i in range(1, ncols):
        M2M_LANGS.append(worksheet[0][i].value)
    for i in range(1, nrows):
        for j in range(1, ncols):
            if i != j:
                m2m_x2x[_lang_pair(M2M_LANGS[i-1], M2M_LANGS[j-1])] = float(worksheet[i][j].value)
    return m2m_x2x


def calculate_avg_score(x2x, src=None, tgt=None, model_name="m2m"):
    results = []
    if src == "x" and tgt == "y":
        for key in x2x.keys():
            if "{}".format("en") not in key:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: x->y: {:.2f}".format(model_name, avg))
    elif src is not None:
        for key in x2x.keys():
            if "{}->".format(src) in key:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        assert len(results) == 101, "{}".format(len(results))
        print("{}: {}->x: {:.2f}".format(model_name, src, avg))
    elif tgt is not None:
        for key in x2x.keys():
            if "->{}".format(tgt) in key:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        assert len(results) == 101, "{}".format(len(results))
        print("{}: x->{}: {:.2f}".format(model_name, tgt, avg))
    else:
        print("{}: all: {:.2f}".format(model_name, sum(x2x.values())/len(x2x)))


def get_results(method):
    x2x = {}
    results = []
    checkpoint_name = args.checkpoint_name
    print(checkpoint_name)
    for i, src in enumerate(LANGS):
        results.append([])
        for j, tgt in enumerate(LANGS):
            if src != tgt:
                try:
                    if method == "direct":
                        with open(os.path.join(args.log, "{}-{}.BLEU".format(src, tgt)), "r", encoding="utf-8") as r:
                            result_lines = r.readlines()
                            fill = False
                            for i in range(len(result_lines) - 1, -1, -1):
                                if checkpoint_name.replace("//", "/") == result_lines[i].strip().replace("//", "/").replace("MODEL: ", ""):
                                    last_line = result_lines[i + 1]  # read the latest results
                                    if 'BLEU+case.mixed' in last_line:
                                        score = float(last_line.split()[2])
                                        x2x["{}->{}".format(src, tgt)] = score
                                        results[-1].append(score)
                                        fill = True
                                        break
                            if not fill:
                                results[-1].append(0)
                                x2x["{}->{}".format(src, tgt)] = 0
                                print(os.path.join(args.log, "{}-{}.BLEU".format(src, tgt)))
                    else:
                        path = os.path.join(args.pivot_log, "{}-{}.BLEU".format(src, tgt)) if src != "en" and tgt != "en" else os.path.join(args.log, "{}-{}.BLEU".format(src, tgt))
                        with open(path, "r", encoding="utf-8") as r:
                            result_lines = r.readlines()
                            fill = False
                            for i in range(len(result_lines) - 1, -1, -1):
                                if checkpoint_name.replace("//", "/") == result_lines[i].strip().replace("//", "/").replace("MODEL: ", ""):
                                    last_line = result_lines[i + 1]  # read the latest results
                                    if 'BLEU+case.mixed' in last_line:
                                        score = float(last_line.split()[2])
                                        x2x["{}->{}".format(src, tgt)] = score
                                        results[-1].append(score)
                                        fill = True
                                        break
                            if not fill:
                                results[-1].append(0)
                                x2x["{}->{}".format(src, tgt)] = 0
                                print(os.path.join(args.log, "{}-{}.BLEU".format(src, tgt)))
                except:
                    print(os.path.join(args.log, "{}-{}.BLEU".format(src, tgt)))
                    x2x["{}->{}".format(src, tgt)] = 0
                    results[-1].append(0)
            else:
                results[-1].append(0)
            assert len(results[-1]) == j + 1, "{}-{} | {}".format(src, tgt, len(results[-1]))
        assert len(results[-1]) == 102, "{}-{} | {}".format(src, tgt, len(results[-1]))

    calculate_avg_score(x2x, src="en", model_name=method)
    calculate_avg_score(x2x, tgt="en", model_name=method)
    calculate_avg_score(x2x, src="x", tgt="y", model_name=method)
    calculate_avg_score(x2x, model_name=method)
    return x2x


if __name__ == "__main__":
    args = parse_args()

    # m2m_x2x = read_excel("/home/v-jiaya/SharedTask/m2m.xls")
    m2m_x2x = read_excel("/mnt/input/SharedTask/m2m.xls")
    calculate_avg_score(m2m_x2x, src="en")
    calculate_avg_score(m2m_x2x, tgt="en")
    calculate_avg_score(m2m_x2x, src="x", tgt="y")
    calculate_avg_score(m2m_x2x)
    #
    direct_x2x = get_results(method="direct")
    pivot_x2x = get_results(method="pivot")
    threshold = args.threshold
    pivot_dirs = []
    for dir in direct_x2x.keys():
        if "en" not in dir and (pivot_x2x[dir] - direct_x2x[dir]) > threshold:
            pivot_dirs.append(dir)
    print(",".join(pivot_dirs).replace("->","-"))
    # Best
    best_x2x = {}
    for dir in direct_x2x.keys():
        if dir in pivot_dirs:
            best_x2x[dir] = pivot_x2x[dir]
        else:
            best_x2x[dir] = direct_x2x[dir]
    calculate_avg_score(best_x2x, src="en", model_name="best")
    calculate_avg_score(best_x2x, tgt="en", model_name="best")
    calculate_avg_score(best_x2x, src="x", tgt="y", model_name="best")
    calculate_avg_score(best_x2x, model_name="best")

    with open(args.pivot_pairs, "w", encoding="utf-8") as w:
        w.write("{}\n".format(",".join(pivot_dirs).replace("->","-")))
        print("Successfully Saving Pivot Pairs to {}".format(args.pivot_pairs))







