import os

root = "/private/home/chuntinz/work/data/mnmt_data/ted/"

rawdir = os.path.join(root, "raw")
temp = os.path.join(root, "temp")


def read_data(dirname, lang, split):
    bad = 0
    path1 = os.path.join(dirname, "{}.{}".format(split, lang))
    path2 = os.path.join(dirname, "{}.{}".format(split, "en"))
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
    p1 = os.path.join(dirname, "{}.{}".format(split, lang))
    p2 = os.path.join(dirname, "{}.en".format(split))
    with open(p1, "w", encoding="utf-8") as f1, open(p2, "w", encoding="utf-8") as f2:
        for xx, en in data:
            f1.write(xx + "\n")
            f2.write(en + "\n")


for dirname in os.listdir(rawdir):
    if not dirname.endswith("en"):
        continue

    fields = dirname.split("_")
    if len(fields) == 2:
        lang, xx = fields
    else:
        lang = "_".join(fields[:2])
        xx = fields[-1]
    assert xx == "en"
    inputdirname = os.path.join(rawdir, dirname)
    train_data = read_data(inputdirname, lang, "train")
    valid_data = read_data(inputdirname, lang, "valid")
    test_data = read_data(inputdirname, lang, "test")

    optdirname = os.path.join(temp, "{}_en".format(lang))
    if not os.path.exists(optdirname):
        os.mkdir(optdirname)

    write_data(train_data, optdirname, "train", lang)
    write_data(valid_data, optdirname, "valid", lang)
    write_data(test_data, optdirname, "test", lang)