import os

split = "train"
print(split)


def opt(root, langs, dataset):
    data_dir = os.path.join(root)
    langs = langs.split(",")

    eng_toks = []
    lan_toks = []

    sents = []
    for lang in langs:
        flang = os.path.join(data_dir, "{}_en".format(lang), "spm.{}.{}".format(split, lang))
        fengl = os.path.join(data_dir, "{}_en".format(lang), "spm.{}.en".format(split))
        temp_num_eng_toks, temp_num_lang_toks = 0, 0
        temp_num_sents = 0
        with open(fengl, encoding="utf-8") as fen, open(flang, encoding="utf-8") as flan:
            for leng, llang in zip(fen, flan):
                len_eng, len_lang = len(leng.strip().split()), len(llang.strip().split())
                if len_eng < 512 and len_lang < 512:
                    temp_num_eng_toks += len_eng
                    temp_num_lang_toks += len_lang
                    temp_num_sents += 1
        eng_toks.append(temp_num_eng_toks)
        lan_toks.append(temp_num_lang_toks)
        sents.append(temp_num_sents)

    print(dataset)
    print("ptrain by sents")
    print({lang: sents[ii] / sum(sents) for ii, lang in enumerate(langs)})
    print("m2o")
    print({lang:eng_toks[ii]/sum(eng_toks) for ii, lang in enumerate(langs)})
    print("o2m")
    print({lang:lan_toks[ii]/sum(lan_toks) for ii, lang in enumerate(langs)})

opt("/home/chuntinz/tir5/data/mnmt_data/my_opus10/data", "yi,mr,oc,be,ta,ka,gl,ur,bg,is", "opus")
opt("/home/chuntinz/tir5/data/mnmt_data/ted/ted8_related/data", "aze,bel,glg,slk,tur,rus,por,ces", "ted related")
opt("/home/chuntinz/tir5/data/mnmt_data/ted/ted8_diverse/data", "bos,mar,hin,mkd,ell,bul,fra,kor", "ted diverse")