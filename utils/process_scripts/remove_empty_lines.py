import os
import sys

rawdir = "/jet/home/chuntinz/work/data/wmt4"


def read_data(dirname, lang, split):
    bad = 0
    path1 = os.path.join(dirname, "{}.en-{}.{}".format(split, lang, lang))
    path2 = os.path.join(dirname, "{}.en-{}.{}".format(split, lang, "en"))
    data = []
    with open(path1, "r", encoding="utf-8") as f1, open(path2, "r", encoding="utf-8") as f2:
        for xx, en in zip(f1, f2):
            if xx.strip() == "" or en.strip() == "":
                bad += 1
                continue
            data.append((xx.strip(), en.strip()))
    print("split = {}, lang = {}, bad = {}".format(split, lang, bad))
    return data


def write_data(data, dirname, split, lang):
    p1 = os.path.join(dirname, "{}.{}.deemp".format(split, lang))
    p2 = os.path.join(dirname, "{}.en.deemp".format(split))
    with open(p1, "w", encoding="utf-8") as f1, open(p2, "w", encoding="utf-8") as f2:
        for xx, en in data:
            f1.write(xx + "\n")
            f2.write(en + "\n")
    os.rename(p1, os.path.join(dirname, "{}.en-{}.{}".format(split, lang, lang)))
    os.rename(p2, os.path.join(dirname, "{}.en-{}.{}".format(split, lang, "en")))


for dirname in os.listdir(rawdir):
    if not dirname[-4:-2] == "en":
        continue

    fields = dirname.split("_")
    lang = dirname[-2:]
    xx = "en"
    inputdirname = os.path.join(rawdir, dirname)
    train_data = read_data(inputdirname, lang, "train")
    valid_data = read_data(inputdirname, lang, "valid")
    test_data = read_data(inputdirname, lang, "test")

    write_data(train_data, inputdirname, "train", lang)
    write_data(valid_data, inputdirname, "valid", lang)
    write_data(test_data,  inputdirname, "test", lang)