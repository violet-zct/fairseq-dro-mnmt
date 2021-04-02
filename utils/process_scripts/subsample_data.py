import numpy as np
import os
import sys

np.random.seed(1)

root = sys.argv[1]
src_file = sys.argv[2]  # xx file
tgt_file = sys.argv[3]  # en file
K = int(sys.argv[4])
suffix = sys.argv[5]

src_out_file = src_file + "." + suffix
tgt_out_file = tgt_file + "." + suffix

bad = 0
path1 = os.path.join(root, src_file)
path2 = os.path.join(root, tgt_file)
data = []
with open(path1, "r", encoding="utf-8") as f1, open(path2, "r", encoding="utf-8") as f2:
    for xx, en in zip(f1, f2):
        if xx.strip() == "" or en.strip() == "":
            bad += 1
            continue
        data.append((xx.strip(), en.strip()))

selected = np.random.permutation(np.arange(len(data)))[:K]

with open(os.path.join(root, src_out_file), "w", encoding="utf-8") as fxx, \
     open(os.path.join(root, tgt_out_file), "w", encoding="utf-8") as fen:
    for idx in selected:
        fxx.write(data[idx][0] + "\n")
        fen.write(data[idx][1] + "\n")