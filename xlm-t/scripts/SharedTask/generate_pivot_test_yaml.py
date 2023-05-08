LANGS="af,am,ar,as,ast,az,be,bn,bs,bg,ca,ceb,cs,ku,cy,da,de,el,en,et,fa,fi,fr,ff,ga,gl,gu,ha,he,hi,hr,hu,hy,ig,id,is,it,jv,ja,kam,kn,ka,kk,kea,km,ky,ko,lo,lv,ln,lt,lb,lg,luo,ml,mr,mk,mt,mn,mi,ms,my,nl,no,ne,ns,ny,oc,om,or,pa,pl,pt,ps,ro,ru,sk,sl,sn,sd,so,es,sr,sv,sw,ta,te,tg,tl,th,tr,uk,umb,ur,uz,vi,wo,xh,yo,zh,zt,zu".split(",")
if __name__ == "__main__":
    cmds = ""
    for lang in LANGS:
        cmd = "- name: test_{}2x_pivot\n  sku: G1\n  sku_count: 1\n  command: \n    - bash ./shells/aml/multi-node/large_task1/deltalm/test/test_aml_devtest_pivot_all.sh 2x /mnt/input/SharedTask/thunder/large_track/data/model/deltalm/FULL-v1/lr1e-4-deltalm-postnorm-64GPU/checkpoint10.pt 96 4 {}\n".format(lang, lang)
        cmds += cmd
    print(cmds)
