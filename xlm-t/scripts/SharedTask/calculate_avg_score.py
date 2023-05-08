import argparse
import xlwt
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(",")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', '-log', type=str,
                        default=r'/home/v-jiaya/SharedTask/log/log.txt', help='input stream')
    args = parser.parse_args()
    return args


def create_excel(results, name):
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet("SharedTask", cell_overwrite_ok=True)
    worksheet.write(0, 0, label="e->x")
    worksheet.write(1, 0, label="DeltaLM-Postnorm(Large)")
    for i in range(len(LANGS)):
        worksheet.write(0, i + 1, label=LANGS[i])
        worksheet.write(1, i + 1, label=results[i])
    workbook.save('/home/v-jiaya/SharedTask/xlm-t/{}.xls'.format(name))
    return workbook


if __name__ == "__main__":
    args = parse_args()
    name="x2e"
    with open(args.log, "r", encoding="utf-8") as r:
        lines = r.readlines()
        x2x = []
        for line in lines:
            if 'BLEU+case.mixed' in line:
                score = float(line.split()[4])
                x2x.append(score)

        print("x->y results")
        print(round(sum(x2x)/len(x2x), 2))
        create_excel(x2x, name="x2e")







