import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
from collections import defaultdict

labelsize = 13
legendsize = 13
mpl.rcParams['xtick.labelsize'] = labelsize
mpl.rcParams['ytick.labelsize'] = labelsize
mpl.rcParams['font.size'] = labelsize

log = sys.argv[1]
root = "/Users/chuntinz/Documents/research/fairseq-dro-mnmt/analysis/"
opt_dir = os.path.join(root, log)
if not os.path.exists(opt_dir):
    os.mkdir(opt_dir)
    os.system("scp tir:/home/chuntinz/tir5/logs/{}/*log.txt {}/".format(log, opt_dir))

langs = []
lang2idx = {}
idx2lang = {}
sum_train = 0
lang_train_size = dict()

valid_ppl = []
lang_valid_ppl = defaultdict(list)

with open(os.path.join(opt_dir, "log.txt").format(root, log)) as fin:
    for line in fin:
        if "INFO | fairseq_cli.train | Namespace(" in line:
            fields = line.strip().split(", ")
            for field in fields:
                if field.startswith("lang_pairs"):
                    field = field.strip().split("=")[-1]
                    lang_idx = 1 if field.strip().split(",")[0].split("-")[0].strip().strip('\'') == "en" else 0
                    langs = [langpair.split("-")[lang_idx].strip().strip('\'') for langpair in field.strip().split(",")]
                    lang2idx = {lang:idx for idx, lang in enumerate(sorted(list(set(langs))))}
                    idx2lang = {v:k for k, v in lang2idx.items()}

        if "INFO | fairseq.data.multilingual.multilingual_data_manager |" in line and "train " in line:
            lang = line.strip().split(" ")[-3].split("-")[lang_idx]
            size = float(line.strip().split(" ")[-2].strip())
            lang_train_size[lang] = size
            sum_train += size

        if "| valid on" in line:
            for ff in line.strip().split("|"):
                fields = ff.strip().split()
                first = fields[0].strip()
                if first == "ppl":
                    valid_ppl.append(float(fields[1].strip()))
                if first.startswith("fg_ppl"):
                    lang = idx2lang[int(first[6])]
                    lang_valid_ppl[lang].append(float(fields[1].strip()))

K = 20
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(10, 5)

x = list(range(len(valid_ppl)-K))
legends = ["all"] + ["{}={:.3f}".format(lang, lang_train_size[lang]/sum_train) for lang in langs]
colors = plt.cm.jet(np.linspace(0, 1, len(langs)+1))

ax.plot(x, valid_ppl[K:], 'o', markersize=1, color=colors[-1])
for idx, lang in enumerate(langs):
    ax.plot(x, lang_valid_ppl[lang][K:], 'o', markersize=1, color=colors[idx])

ax.set(title="valid ppl", xlabel="steps", ylabel="ppl")
ax.legend(legends, loc='best', fontsize=10)

best_ppl = min(valid_ppl)
best_idx = np.argmin(valid_ppl)
ax.plot(best_idx, best_ppl, "rv", markersize=1.5)
ax.annotate("best@{}={:.3f}@{}".format(legends[0], best_ppl, best_idx), (best_idx, best_ppl), fontsize=8)

for idx, lang in enumerate(langs):
    best_ppl = min(lang_valid_ppl[lang][K:])
    best_idx = np.argmin(lang_valid_ppl[lang][K:])
    ax.plot(best_idx, best_ppl, "rv", markersize=1.5)
    ax.annotate("best@{}={:.3f}@{}".format(legends[idx+1].split("=")[0], best_ppl, best_idx), (best_idx, best_ppl), fontsize=8)

fig.savefig(os.path.join(opt_dir, "{}_valid_ppl.pdf".format(log)), bbox_inches='tight')