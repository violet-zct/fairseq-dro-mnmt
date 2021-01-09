import numpy as np
import os
import sys

np.random.seed(1)
temperature = 5.
cap_valid = 800

root = "/home/chuntinz/tir5/data/mnmt_data/opus10"
rawdir = os.path.join(root, "raw")

# yi mr oc be ta ka gl ur bg is
# yi,mr,oc,be,ta,ka,gl,ur,bg,is
# en-yi,en-mr,en-oc,en-be,en-ta,en-ka,en-gl,en-ur,en-bg,en-is
# yi-en,mr-en,oc-en,be-en,ta-en,ka-en,gl-en,ur-en,bg-en,is-en
langs = "yi,mr,oc,be,ta,ka,gl,ur,bg,is"
opt_root = "/home/chuntinz/tir5/data/mnmt_data/opus10"
opt_root = os.path.join(opt_root, "data")
langs = langs.split(",")


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


def write_data(data, p1, p2):
    with open(p1, "w", encoding="utf-8") as f1, open(p2, "w", encoding="utf-8") as f2:
        for xx, en in data:
            f1.write(xx + "\n")
            f2.write(en + "\n")


def get_sampling_ratios(dataset_sizes):
    total_size = sum(dataset_sizes)
    ratios = np.array([(size / total_size) ** (1.0/temperature) for size in dataset_sizes])
    ratios = ratios / ratios.sum()
    return ratios


def default_virtual_size_func(sizes, ratios, max_scale_up=1.5):
    largest_idx = np.argmax(sizes)
    largest_r = ratios[largest_idx]
    largest_s = sizes[largest_idx]
    # set virtual sizes relative to the largest dataset
    virtual_sizes = [(r / largest_r) * largest_s for r in ratios]
    vsize = sum(virtual_sizes)
    max_size = sum(sizes) * max_scale_up
    return int(vsize if vsize < max_size else max_size)


def upsample_and_write(data, data_sizes, sample_ratios):
    total_size = default_virtual_size_func(data_sizes, sample_ratios)
    counts = np.array([total_size * r for r in sample_ratios], dtype=np.int64)
    diff = total_size - counts.sum()
    assert diff >= 0
    # due to round-offs, the size might not match the desired sizes
    if diff > 0:
        dataset_indices = np.random.choice(len(sample_ratios), size=diff, p=sample_ratios)
        for i in dataset_indices:
            counts[i] += 1
    indices = [np.random.choice(d, c, replace=(c > d)) for c, d in zip(counts, data_sizes)]

    op1 = os.path.join(opt_root, "raw.combine.xx")
    with open(op1, "w", encoding="utf-8") as fout:
        for ii, lang in enumerate(langs):
            for idx in indices[ii]:
                fout.write(data[lang][0][idx][0] + "\n")
    op2 = os.path.join(opt_root, "raw.combine.en")
    with open(op2, "w", encoding="utf-8") as fout:
        for lang in langs:
            for xx, en in data[lang][0]:
                fout.write(en + "\n")


lang_pack = {}
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

    if lang not in langs:
        continue

    inputdirname = os.path.join(rawdir, dirname)
    train_data = read_data(inputdirname, lang, "train")
    valid_data = read_data(inputdirname, lang, "valid")

    lang_pack[lang] = (train_data, valid_data)

train_data_sizes = [len(lang_pack[lang][0]) for lang in langs]
train_sample_ratios = get_sampling_ratios(train_data_sizes)
upsample_and_write(lang_pack, train_data_sizes, train_sample_ratios)

for lang in langs:
    valid_size = len(lang_pack[lang][1])
    if valid_size < cap_valid:
        indices = list(range(valid_size))
    else:
        indices = np.random.choice(valid_size, size=cap_valid, replace=False)

    optdir = os.path.join(opt_root, "{}_en".format(lang))
    if not os.path.exists(optdir):
        os.mkdir(optdir)
    with open(os.path.join(optdir, "cap.raw.valid.{}".format(lang)), "w", encoding="utf-8") as fxx, \
        open(os.path.join(optdir, "cap.raw.valid.en"), "w", encoding="utf-8") as fen:
        for idx in indices:
            fxx.write(lang_pack[lang][1][idx][0] + "\n")
            fen.write(lang_pack[lang][1][idx][1] + "\n")