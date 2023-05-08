LANGS="en,et,hr,sr,mk,sr".split(",")
if __name__ == "__main__":
    cmds = ""
    for lang in LANGS:
        cmd = "- name: test_{}2x\n  sku: G1\n  sku_count: 1\n  command: \n    - bash ./shells/aml/multi-node/small_task1/test/test_aml_devtest_all.sh 2x /mnt/input/SharedTask/thunder/small_task1/download/model/deltalm/lr1e-4-deltalm-postnorm/checkpoint_last.pt 128 4 {}\n".format(lang, lang)
        cmds += cmd
    print(cmds)
