import argparse
import xlwt
import xlrd
import os
import logging
################Our##############################################################
logger = logging.getLogger(__name__)
logging.basicConfig( #must set the level
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
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
                        default=r'/mnt/input/SharedTask/devtest_Product_Translation/Evaluation/BLEU/', help='input stream')
    parser.add_argument('--checkpoint-name', '-checkpoint-name', type=str,
                        default=r'prod', help='input stream')
    parser.add_argument('--result', '-result', type=str,
                        default=r'/mnt/input/SharedTask/ExcelResults/prod.xls', help='input stream')
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
    save_path = '/mnt/input/SharedTask/ExcelResults/{}.xls'.format(name)
    workbook.save(save_path)
    print("Saving to {}".format(save_path))
    return workbook


def _lang_pair(src, tgt):
    return "{}->{}".format(src, tgt)


def read_excel(filename):
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
        print("{}: {}->x: {:.2f}".format(model_name, src, avg))
    elif tgt is not None:
        for key in x2x.keys():
            if "->{}".format(tgt) in key:
                results.append(x2x[key])
        avg = sum(results) / len(results)
        print("{}: x->{}: {:.2f}".format(model_name, tgt, avg))
    else:
        avg = sum(x2x.values())/len(x2x)
        print("{}: all: {:.2f}".format(model_name, avg))
    return avg


if __name__ == "__main__":
    args = parse_args()
    x2x = {}
    results = []
    for i, src in enumerate(LANGS):
        results.append([])
        for j, tgt in enumerate(LANGS):
            bleu_file = os.path.join(args.log, "{}-{}.BLEU".format(src, tgt))
            if src != tgt and os.path.exists(bleu_file):
                with open(os.path.join(args.log, "{}-{}.BLEU".format(src, tgt)), "r", encoding="utf-8") as r:
                    result_lines = r.readlines()
                    last_line = result_lines[-1] # read the latest results
                    score = float(last_line.split()[2])
                    x2x["{}->{}".format(src, tgt)] = score
                    results[-1].append(score)
                logger.info(f"Reading {bleu_file}")
            else:
                results[-1].append(0)
            assert len(results[-1]) == j + 1, "{}-{} | {}".format(src, tgt, len(results[-1]))
        assert len(results[-1]) == 102, "{}-{} | {}".format(src, tgt, len(results[-1]))

    x2e_results = calculate_avg_score(x2x, tgt="en", model_name="our")
    e2x_results = calculate_avg_score(x2x, src="en", model_name="our")
    x2y_results = calculate_avg_score(x2x, src="x", tgt="y", model_name="our")
    avg_results = calculate_avg_score(x2x, model_name="our")

    name = args.checkpoint_name.replace("/", "_").replace(".", "_") + "_direct"
    create_excel(results, name=name)







