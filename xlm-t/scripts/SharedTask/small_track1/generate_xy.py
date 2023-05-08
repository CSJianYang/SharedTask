LANGS="et,hr,hu,sr,mk".split(",")
x2y=[]
y2x=[]
all=[]
for i in range(len(LANGS)):
    for j in range(i+1, len(LANGS)):
        x2y.append("{}-{}".format(LANGS[i], LANGS[j]))
        y2x.append("{}-{}".format(LANGS[j], LANGS[i]))
all=x2y+y2x
x2y=" ".join(x2y)
y2x=" ".join(y2x)
print(x2y)
print(y2x)
cmds=""
for pair in all:
    src, tgt = pair.split('-')
    cmd = """
- name: filter_{}-{}
  sku: G0
  sku_count: 1
  command:
    - bash ./shells/preprocess/BT/small_track1/filter_x2y_bt.sh {} {} """.format(src, tgt, src, tgt)
    cmds += cmd
print(cmds)