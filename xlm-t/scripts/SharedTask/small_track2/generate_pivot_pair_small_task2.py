import argparse
import xlwt
import xlrd
import os


TOTAL_DIRECTION = 30
def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )
LANGS="en,id,jv,ms,ta,tl".split(",")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', '-log', type=str,
                        default=r'/mnt/input/SharedTask/thunder/flores101_dataset/SmallTask2/BLEU/', help='input stream')
    parser.add_argument('--pivot-log', '-pivot-log', type=str,
                        default=r'/mnt/input/SharedTask/thunder/flores101_dataset/SmallTask2/PIVOT-BLEU/', help='input stream')
    parser.add_argument('--checkpoint-name', '-checkpoint-name', type=str,
                        default=r'/mnt/input/SharedTask/thunder/small_task2/Filter_v1/model/24L-12L-A100/avg3_18.pt', help='input stream')
    parser.add_argument('--threshold', '-threshold', type=float,
                        default=0, help='input stream')
    parser.add_argument('--pivot-pairs', '-pivot-pairs', type=str,
                        default=r'/home/v-jiaya/SharedTask/xlm-t/test_small_track2_pivot_pairs.txt', help='input stream')
    args = parser.parse_args()
    return args


def create_excel(results, name, save_dir = '/home/v-jiaya/SharedTask/SmallTask2_ExcelResults/'):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet(name, cell_overwrite_ok=True)
    worksheet.write(1, 0, label="DeltaLM-Postnorm (Large)")
    for i in range(len(LANGS)):
        worksheet.write(0, i + 1, label=LANGS[i])
        worksheet.write(i + 1, 0, label=LANGS[i])
    for i in range(len(LANGS)):
        for j in range(len(LANGS)):
            worksheet.write(i + 1, j + 1, label=results[i][j])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    workbook.save('{}/{}.xls'.format(save_dir, name))
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
                #m2m_results.append(float(worksheet[i][j].value))
                m2m_x2x[_lang_pair(M2M_LANGS[i-1], M2M_LANGS[j-1])] = float(worksheet[i][j].value)
    #add ku lang
    omitted_lang = "ku"
    if omitted_lang not in M2M_LANGS:
        for lang in M2M_LANGS:
            m2m_x2x[_lang_pair(lang, omitted_lang)] = 0
            m2m_x2x[_lang_pair(omitted_lang, lang)] = 0
    ##############
    return m2m_x2x


def calculate_avg_score(x2x, src=None, tgt=None, model_name="m2m"):
    results = []
    if src == "x" and tgt == "y":
        for key in x2x.keys():
            if "{}".format("en") not in key and key.split('->')[0] in LANGS and key.split('->')[1] in LANGS:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        assert len(results) == 20, "{}".format(len(results))
        print("{}: x->y: {:.2f}".format(model_name, avg))
    elif src is not None:
        for key in x2x.keys():
            if "{}->".format(src) in key and key.split('->')[0] in LANGS and key.split('->')[1] in LANGS:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        assert len(results) == 5, "{}".format(len(results))
        print("{}: {}->x: {:.2f}".format(model_name, src, avg))
    elif tgt is not None:
        for key in x2x.keys():
            if "->{}".format(tgt) in key and key.split('->')[0] in LANGS and key.split('->')[1] in LANGS:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        assert len(results) == 5, "{}".format(len(results))
        print("{}: x->{}: {:.2f}".format(model_name, tgt, avg))
    else:
        for key in x2x.keys():
            if key.split('->')[0] in LANGS and key.split('->')[1] in LANGS:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        assert len(results) == 30, "{}".format(len(results))
        print("{}: all: {:.2f}".format(model_name, avg))
    return avg


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
                            for i in range(len(result_lines) - 1, -1, -1):
                                if checkpoint_name.replace("//", "/") == result_lines[i].strip().replace("//", "/").replace("MODEL: ", ""):
                                    last_line = result_lines[i + 1]  # read the latest results
                                    if 'BLEU+case.mixed' in last_line:
                                        score = float(last_line.split()[2])
                                        x2x["{}->{}".format(src, tgt)] = score
                                        results[-1].append(score)
                                        break
                                    else:
                                        print(os.path.join(args.log, "{}-{}.BLEU".format(src, tgt)))
                    else:
                        path = os.path.join(args.pivot_log, "{}-{}.BLEU".format(src, tgt)) if src != "en" and tgt != "en" else os.path.join(args.log, "{}-{}.BLEU".format(src, tgt))
                        with open(path, "r", encoding="utf-8") as r:
                            result_lines = r.readlines()
                            for i in range(len(result_lines) - 1, -1, -1):
                                if checkpoint_name.replace("//", "/") == result_lines[i].strip().replace("//", "/").replace("MODEL: ", ""):
                                    last_line = result_lines[i + 1]  # read the latest results
                                    if 'BLEU+case.mixed' in last_line:
                                        score = float(last_line.split()[2])
                                        x2x["{}->{}".format(src, tgt)] = score
                                        results[-1].append(score)
                                        break
                                    else:
                                        print(os.path.join(args.log, "{}-{}.BLEU".format(src, tgt)))
                            if i == 0:
                                print(result_lines)
                except:
                    print(os.path.join(args.log, "{}-{}.BLEU".format(src, tgt)))
                    x2x["{}->{}".format(src, tgt)] = 0
                    results[-1].append(0)
            else:
                results[-1].append(0)
            assert len(results[-1]) == j + 1, "{}-{} | {}".format(src, tgt, len(results[-1]))
        assert len(results[-1]) == 6, "{}-{} | {}".format(src, tgt, len(results[-1]))

    x2e_avg = calculate_avg_score(x2x, tgt="en", model_name=method)
    e2x_avg = calculate_avg_score(x2x, src="en", model_name=method)
    x2y_avg = calculate_avg_score(x2x, src="x", tgt="y", model_name=method)
    all_avg = calculate_avg_score(x2x, model_name=method)
    print(" & {} & {} & {} & {}\\\\".format(round(x2e_avg, 2), round(e2x_avg, 2), round(x2y_avg, 2), round(all_avg, 2)))
    return x2x

def calculate_all_avg(_x2x):
    x2e_avg = calculate_avg_score(_x2x, tgt="en")
    e2x_avg = calculate_avg_score(_x2x, src="en")
    x2y_avg = calculate_avg_score(_x2x, src="x", tgt="y")
    all_avg = calculate_avg_score(_x2x)
    print(" & {} & {} & {} & {}\\\\".format(round(x2e_avg, 2), round(e2x_avg, 2), round(x2y_avg, 2), round(all_avg, 2)))

if __name__ == "__main__":
    args = parse_args()
    #
    m2m_x2x = read_excel("/home/v-jiaya/SharedTask/m2m_175M.xls")
    print("m2m_175M")
    calculate_all_avg(m2m_x2x)
    #
    print("m2m_615M")
    m2m_x2x = read_excel("/home/v-jiaya/SharedTask/m2m_615M.xls")
    print("m2m_615M")
    calculate_all_avg(m2m_x2x)
    #
    direct_x2x = get_results(method="direct")
    pivot_x2x = get_results(method="pivot")
    threshold = args.threshold
    pivot_dirs = []
    direct_dirs = []
    for dir in direct_x2x.keys():
        if "en" not in dir and (pivot_x2x[dir] - direct_x2x[dir]) > threshold:
            pivot_dirs.append(dir)
        else:
            if "en" not in dir:
                direct_dirs.append(dir)
    print("Pivot Pair")
    print(",".join(pivot_dirs).replace("->", "-"))
    print("Direct Pair")
    print(",".join(direct_dirs).replace("->", "-"))
    #Best
    best_x2x = {}
    for dir in direct_x2x.keys():
        if dir in pivot_dirs:
            best_x2x[dir] = pivot_x2x[dir]
        else:
            best_x2x[dir] = direct_x2x[dir]

    x2e_avg = calculate_avg_score(best_x2x, tgt="en", model_name="best")
    e2x_avg = calculate_avg_score(best_x2x, src="en", model_name="best")
    x2y_avg = calculate_avg_score(best_x2x, src="x", tgt="y", model_name="best")
    all_avg = calculate_avg_score(best_x2x, model_name="best")
    print(" & {} & {} & {} & {}\\\\".format(round(x2e_avg, 2), round(e2x_avg, 2), round(x2y_avg, 2), round(all_avg, 2)))

    with open(args.pivot_pairs, "w", encoding="utf-8") as w:
        w.write("{}\n".format(",".join(pivot_dirs).replace("->","-")))
        print("Successfully Saving Pivot Pairs to {}".format(args.pivot_pairs))












