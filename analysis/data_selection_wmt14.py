import numpy as np
import sys
import os
from collections import defaultdict
import shutil

bperoot = "/home/chuntinz/tir5/data/opus_wmt14/joint-bpe-37k"
model_path = "/home/chuntinz/tir5/logs/68_erm_train_dynamics_wmt14_ende_ende"
optroot = "/home/chuntinz/tir5/data/opus_wmt14/wmt14_train_dynamics_bpe"

data = []
with open(os.path.join(bperoot, "train.en"), "r", encoding="utf-8") as fen, open(os.path.join(bperoot, "train.de"), "r",
                                                                                 encoding="utf-8") as fde:
    for sen, sde in zip(fen, fde):
        data.append((sen.strip(), sde.strip()))


def get_confidence_and_variability(epochs_of_vecs, s=-1, e=-1):
    if s != -1 and e != -1:
        mat = np.vstack(epochs_of_vecs[s:e+1])
    else:
        mat = np.vstack(epochs_of_vecs)
    mu = np.mean(mat, axis=0)
    var = np.std(mat, axis=0)
    return mu, var


def collect_topk_index(list_of_tensors, title, K=0.5):
    ds_mus, ds_std = get_confidence_and_variability(list_of_tensors)
    print("size = {}".format(len(ds_mus)))
    cutoff = int(len(ds_std) * K)
    sorted_indices = np.argsort(ds_std)[::-1][:cutoff]  # descending

    opt_dir = os.path.join(optroot, title+"_{}".format(K))
    os.mkdir(opt_dir)
    for lang in ["en", "de"]:
        for split in ["test", "dev"]:
            opt_split = split if split == "test" else "valide"
            shutil.copy(os.path.join(bperoot, "{}.{}".format(split, lang)),  os.path.join(opt_dir, "{}.{}".format(opt_split, lang)))
    with open(os.path.join(opt_dir, "train.en"), "w", encoding="utf-8") as fen, open(os.path.join(opt_dir, "train.de"), "w", encoding="utf-8") as fde:
        for ii in sorted_indices:
            fen.write(data[ii][0] + "\n")
            fde.write(data[ii][1] + "\n")


def collect_random_index(K=0.5):
    total = len(data)
    cutoff = int(total * K)
    sorted_indices = np.random.permutation(total)[:cutoff]

    opt_dir = os.path.join(optroot, "random_{}".format(K))
    os.mkdir(opt_dir)
    for lang in ["en", "de"]:
        for split in ["test", "dev"]:
            opt_split = split if split == "test" else "valide"
            shutil.copy(os.path.join(bperoot, "{}.{}".format(split, lang)),  os.path.join(opt_dir, "{}.{}".format(opt_split, lang)))
    with open(os.path.join(opt_dir, "train.en"), "w", encoding="utf-8") as fen, open(os.path.join(opt_dir, "train.de"), "w", encoding="utf-8") as fde:
        for ii in sorted_indices:
            fen.write(data[ii][0] + "\n")
            fde.write(data[ii][1] + "\n")


def process_epochs():
    max_epoch = 0
    for path in os.listdir(model_path):
        if not path.endswith("npy"):
            continue
        epoch = int(path.split(".")[0].split("_")[-1])
        if epoch > max_epoch:
            max_epoch = epoch

    min_probs = []
    avg_probs = []
    med_probs = []

    print("start loading data")
    for eid in range(1, max_epoch+1):
        if not os.path.exists(os.path.join(model_path, "avg_probs_{}.npy".format(eid))):
            continue
        min_probs.append(np.load(os.path.join(model_path, "min_probs_{}.npy".format(eid))))
        avg_probs.append(np.load(os.path.join(model_path, "avg_probs_{}.npy".format(eid))))
        med_probs.append(np.load(os.path.join(model_path, "median_probs_{}.npy".format(eid))))
    print("end loading data")

    collect_topk_index(min_probs, "min_probs_var")
    collect_topk_index(avg_probs, "avg_probs_var")
    collect_topk_index(med_probs, "med_probs_var")
    collect_random_index()

process_epochs()
