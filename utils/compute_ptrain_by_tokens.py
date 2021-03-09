import os


def opt(root, langs, dataset):
    data_dir = os.path.join(root)
    langs = langs.split(",")

    eng_toks = []
    lan_toks = []
    for lang in langs:
        flang = os.path.join(data_dir, "{}_en".format(lang), "spm.train.{}".format(lang))
        fengl = os.path.join(data_dir, "{}_en".format(lang), "spm.train.en")
        neng_toks = len(open(fengl, encoding="utf-8").read().split())
        nlan_toks = len(open(flang, encoding="utf-8").read().split())
        eng_toks.append(neng_toks)
        lan_toks.append(nlan_toks)

    print(dataset)
    print("m2o")
    print({lang:eng_toks[ii]/sum(eng_toks) for ii, lang in enumerate(langs)})
    print("o2m")
    print({lang:lan_toks[ii]/sum(lan_toks) for ii, lang in enumerate(langs)})

opt("/home/chuntinz/tir5/data/mnmt_data/my_opus10/data", "yi,mr,oc,be,ta,ka,gl,ur,bg,is", "opus")
opt("/home/chuntinz/tir5/data/mnmt_data/ted/ted8_related/data", "aze,bel,glg,slk,tur,rus,por,ces", "ted related")
opt("/home/chuntinz/tir5/data/mnmt_data/ted/ted8_diverse/data", "bos,mar,hin,mkd,ell,bul,fra,kor", "ted diverse")