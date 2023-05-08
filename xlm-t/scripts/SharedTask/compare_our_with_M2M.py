import argparse
import xlwt
import xlrd
LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(",")
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--our', '-our', type=str,
                        default=r'/home/v-jiaya/SharedTask/m2m.xls', help='baseline')
    parser.add_argument('--baseline', '-baseline', type=str,
                        default=r'/home/v-jiaya/SharedTask/our.xls', help='our results')
    args = parser.parse_args()
    return args
def read_excel(filename):
    results = []
    r = xlrd.open_workbook(filename)
    st = r.sheet_by_name("SharedTask")
    nrows = st.nrows
    ncols = st.ncols
    return results