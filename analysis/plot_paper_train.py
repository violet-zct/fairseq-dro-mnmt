import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
from collections import defaultdict

log = sys.argv[1]
root = "/Users/chuntinz/Documents/documents/research/fairseq-dro-mnmt/figs/"
opt_dir = os.path.join(root, log)
if not os.path.exists(opt_dir):
    os.mkdir(opt_dir)
    os.system("scp tir:/home/chuntinz/tir5/logs/{}/log.txt {}/".format(log, opt_dir))
    # os.system("scp tir:/home/chuntinz/tir5/fairseq-dro-mnmt/saved_models/{}/log.txt {}/".format(log, opt_dir))
    # os.system("scp psc:/jet/home/chuntinz/work/fairseq-dro-mnmt/saved_models/{}/*log.txt {}/".format(log, opt_dir))

if "wmt" in log:
    lang_num = 4
elif "opus" in log:
    lang_num = 10
else:
    lang_num = 8

langs = []
lang_train_size = dict()
lang_ema_losses = dict()
lang_ema_after_bl_losses = dict()
lang_ema_weights = dict()
lang_idx = 0
train_losses = []
train_ppl = []
valid_ppl = []
lang2idx = {}
idx2lang = {}

cutoffs = []
with open(os.path.join(opt_dir, "log.txt").format(root, log)) as fin:
    for line in fin:
        if "INFO | fairseq_cli.train | Namespace(" in line:
            fields = line.strip().split(", ")
            for field in fields:
                if field.startswith("lang_pairs"):
                    field = field.strip().split("=")[-1]
                    lang_idx = 1 if field.strip().split(",")[0].split("-")[0].strip().strip('\'') == "en" else 0
                    langs = [langpair.split("-")[lang_idx].strip().strip('\'') for langpair in field.strip().split(",")]
                    for lang in langs:
                        lang_ema_losses[lang] = []
                        lang_ema_weights[lang] = []
                        lang_ema_after_bl_losses[lang] = []
                    lang2idx = {lang:idx for idx, lang in enumerate(sorted(list(set(langs))))}
                    idx2lang = {v:k for k, v in lang2idx.items()}

        if "INFO | fairseq.data.multilingual.multilingual_data_manager |" in line and "train " in line and "dataset" not in line:
            lang = line.strip().split(" ")[-3].split("-")[lang_idx]
            size = float(line.strip().split(" ")[-2].strip())
            lang_train_size[lang] = size

        if "INFO | train_inner | epoch" in line:
            fields = line.strip().split()
            found_loss = found_ppl = False
            for field in fields:
                if field.strip().startswith("loss"):
                    train_loss = float(field.strip().rstrip(",").split("=")[-1])
                    found_loss = True
                if field.strip().startswith("ppl="):
                    ppl = float(field.strip().rstrip(",").split("=")[-1])
                    found_ppl = True
            assert found_loss and found_ppl
            train_losses.append(train_loss)
            train_ppl.append(ppl)

        if "| valid on" in line:
            for ff in line.strip().split("|"):
                fields = ff.strip().split()
                if fields[0].strip() == "ppl":
                    valid_ppl.append(float(fields[1].strip()))
                    break

        if " EMA before-baseline losses: " in line:
            losses = line.strip().split(" EMA before-baseline losses: ")[-1].split()
            for idx, loss in enumerate(losses):
                lang_ema_losses[idx2lang[idx]].append(float(loss.strip()))

        if " EMA after-baseline losses: " in line:
            losses = line.strip().split(" EMA after-baseline losses: ")[-1].split()
            for idx, loss in enumerate(losses):
                lang_ema_after_bl_losses[idx2lang[idx]].append(float(loss.strip()))

        if " Group loss weights: " in line:
            weights = line.strip().split("Group loss weights:")[-1].split()
            for idx, weight in enumerate(weights):
                lang_ema_weights[idx2lang[idx]].append(float(weight.strip()))

        if " EMA past losses: " in line:
            losses = line.strip().split(" EMA past losses: ")[-1].split()
            for idx, loss in enumerate(losses):
                lang_ema_losses[idx2lang[idx]].append(float(loss.strip()))

        if " Group loss weights: tensor" in line:
            weights = line.strip().split("([")[-1].rstrip("],").split(",")
            for idx, weight in enumerate(weights):
                lang_ema_weights[idx2lang[idx]].append(float(weight.strip()))
            # if lang_num == 10:
            #     line = fin.readline()
            #     lang_ema_weights['yi'].append(float(line.split("],")[0].strip()))

print(lang_train_size)
sum_train = sum(list(lang_train_size.values()))
legends = ["{}={:.3f}".format(lang, lang_train_size[lang]/sum_train) for lang in langs]
print(legends)
labelsize = 16
legendsize = 16
mpl.rcParams['xtick.labelsize'] = labelsize
mpl.rcParams['ytick.labelsize'] = labelsize
mpl.rcParams['font.size'] = labelsize
plt.style.use('seaborn-deep')
colormap = plt.cm.gist_ncar

def plot_ax_ema(ax, y, title):
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(langs)))))
    for lang in langs:
        ax.plot(np.arange(len(y[lang])), y[lang], alpha=0.6)
    ax.legend(legends, loc='best', fontsize=16)
    ax.set(xlabel="epochs", ylabel=title)


def plot_ax(ax, y, title, ylabel, color, step=300):
    if "valid" in title:
        step = 10
    print(title, len(y))
    ax.plot(np.arange(len(y)), y, color=color)
    ax.set(title=title, xlabel="epochs", ylabel=ylabel)
    ax.xaxis.set_ticks(np.arange(0, max(np.arange(len(y)), ), step))

# plot train loss, (train_ppl and valid_ppl)
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(30, 5)
x = list(range(len(train_losses)))
colors = ["royalblue", "indianred", "mediumseagreen"]

K = 10
plot_ax(ax[0], train_losses[K:], "train_losses", "losses", colors[0])
plot_ax(ax[1], train_ppl[K:], "train_ppl", "ppl", colors[1])
plot_ax(ax[2], valid_ppl[K:], "valid_ppl", "ppl", colors[2])
fig.savefig(os.path.join(opt_dir, "{}_loss_ppl.pdf".format(log)), bbox_inches='tight')

### plot ema losses, ema weights
fig, ax = plt.subplots(1)
fig.set_size_inches(10, 5)
if "ted" in log:
    plot_ax_ema(ax, lang_ema_losses, "ema_loss") # comment this for wmt
    plot_ax_ema(ax, lang_ema_after_bl_losses, "Historical Losses")
else:
    plot_ax_ema(ax, lang_ema_after_bl_losses, "Historical Losses")
# plot_ax_ema(ax[1], lang_ema_weights, "ema_weights", True)
fig.savefig(os.path.join(opt_dir, "{}_ema_loss.pdf".format(log)), bbox_inches='tight')

labelsize = 16
legendsize = 16
mpl.rcParams['xtick.labelsize'] = labelsize
mpl.rcParams['ytick.labelsize'] = labelsize-2
mpl.rcParams['font.size'] = labelsize
if "ted" in log:
    nrows = 2
else:
    nrows = 1
row = lang_num // nrows
fig, ax = plt.subplots(len(langs) // row, row)
if "ted" in log:
    fig.set_size_inches(30, 8)
else:
    fig.set_size_inches(20, 3)  # wmt

colors = plt.cm.jet(np.linspace(0, 1, len(langs)))
max_value = []
for lang in langs:
    max_value.extend(lang_ema_weights[lang])
max_value = np.log(max(max_value)) + 0.2

normalizer = []
for ii in range(len(lang_ema_weights[langs[0]])):
    normalizer.append(sum([lang_ema_weights[lang][ii] for lang in langs]))
normalizer = np.array(normalizer)

tau = 5
ratios = [lang_train_size[lang]/sum_train for lang in langs]
temp_norm = sum(r ** tau for r in ratios)

# WMT plot
# steps = 0
# print(len(ax))
# colors = ["purple", "darkblue", "peru", "crimson"]
# for idx, lang in enumerate(langs):
#     i, j = idx // row, idx % row
#     print(i, j)
#     data_ratio = float(legends[idx].split("=")[-1])
#     # ax[i][j].plot(np.arange(len(lang_ema_weights[lang])),
#     #               np.ones(len(lang_ema_weights[lang]))* (data_ratio**tau / temp_norm),
#     #               linestyle='dotted', markersize=1, color=colors[idx], alpha=0.75)
#     if steps > 0:
#         ax[j].plot(np.arange(steps),
#                       np.ones(steps) * (data_ratio),
#                       linestyle='dashed', markersize=1, color=colors[idx], alpha=0.5)
#
#         ax[j].plot(np.arange(steps), np.array(lang_ema_weights[lang][:steps]) / normalizer[:steps], 'o',
#                       markersize=1, color=colors[idx])
#     else:
#         ax[j].plot(np.arange(len(lang_ema_weights[lang])),
#                       np.log(np.ones(len(lang_ema_weights[lang]))*(data_ratio)),
#                       linestyle='dashed', markersize=1, color=colors[idx], alpha=0.5)
#
#         ax[j].plot(np.arange(len(lang_ema_weights[lang])), np.log(np.array(lang_ema_weights[lang])/normalizer), 'o', markersize=1, color=colors[idx])
#     # print(legends[idx])
#     # ax[i][j].legend(legends[idx], loc='best', fontsize=10)
#     # ax[i][j].set_ylim([0, max_value])
#     if i == 0 and j == 0:
#         ax[j].set(title=legends[idx], xlabel="epochs", ylabel="best response")
#     elif i == 0:
#         ax[j].set(title=legends[idx], xlabel="epochs")
#     else:
#         ax[j].set(title=legends[idx])
# plt.subplots_adjust(wspace=0.2,
#                     hspace=0.3)
# fig.savefig(os.path.join(opt_dir, "{}_ema_weights.pdf".format(log)), bbox_inches='tight')

# TED plot
steps = 0
print(len(ax))
for idx, lang in enumerate(langs):
    i, j = idx // row, idx % row
    print(i, j)
    data_ratio = float(legends[idx].split("=")[-1])
    # ax[i][j].plot(np.arange(len(lang_ema_weights[lang])),
    #               np.ones(len(lang_ema_weights[lang]))* (data_ratio**tau / temp_norm),
    #               linestyle='dotted', markersize=1, color=colors[idx], alpha=0.75)
    if steps > 0:
        ax[i][j].plot(np.arange(steps),
                      np.ones(steps) * (data_ratio),
                      linestyle='dashed', markersize=1, color=colors[idx], alpha=0.5)

        ax[i][j].plot(np.arange(steps), np.array(lang_ema_weights[lang][:steps]) / normalizer[:steps], 'o',
                      markersize=1, color=colors[idx])
    else:
        ax[i][j].plot(np.arange(len(lang_ema_weights[lang])),
                      np.log(np.ones(len(lang_ema_weights[lang]))*(data_ratio)),
                      linestyle='dashed', markersize=1, color=colors[idx], alpha=0.5)

        ax[i][j].plot(np.arange(len(lang_ema_weights[lang])), np.log(np.array(lang_ema_weights[lang])/normalizer), 'o', markersize=1, color=colors[idx])
    # print(legends[idx])
    # ax[i][j].legend(legends[idx], loc='best', fontsize=10)
    # ax[i][j].set_ylim([0, max_value])
    if i == 1 and j == 0:
        ax[i][j].set(title=legends[idx], xlabel="epochs", ylabel="best response")
    elif i == 0 and j == 0:
        ax[i][j].set(title=legends[idx], ylabel="best response")
    elif i == 1:
        ax[i][j].set(title=legends[idx], xlabel="epochs")
    else:
        ax[i][j].set(title=legends[idx])
plt.subplots_adjust(wspace=0.15,
                    hspace=0.3)
fig.savefig(os.path.join(opt_dir, "{}_ema_weights.pdf".format(log)), bbox_inches='tight')

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(30, 5)
for idx, lang in enumerate(langs):
    i, j = idx // row, idx % row
    data_ratio = float(legends[idx].split("=")[-1])
    # ax[i][j].plot(np.arange(len(lang_ema_weights[lang])),
    #               np.ones(len(lang_ema_weights[lang]))* (data_ratio**tau / temp_norm),
    #               linestyle='dotted', markersize=1, color=colors[idx], alpha=0.75)
    if steps > 0:
        ax.plot(np.arange(steps),
                      np.ones(steps) * (data_ratio),
                      linestyle='dashed', markersize=1, color=colors[idx], alpha=0.5)

        ax.plot(np.arange(steps), np.array(lang_ema_weights[lang][:steps]) / normalizer[:steps], 'o',
                      markersize=1, color=colors[idx])
    else:
        ax.plot(np.arange(len(lang_ema_weights[lang])),
                      np.ones(len(lang_ema_weights[lang]))*(data_ratio),
                      linestyle='dashed', markersize=1, color=colors[idx], alpha=0.5)

        ax.plot(np.arange(len(lang_ema_weights[lang])), np.array(lang_ema_weights[lang])/normalizer, 'o', markersize=1, color=colors[idx])
    # print(legends[idx])
ax.legend(legends, loc='best', fontsize=10)
ax.set_ylim([0, max_value])
ax.set(title="best response", xlabel="steps", ylabel="ema_weights")
fig.savefig(os.path.join(opt_dir, "{}_single_ema_weights.pdf".format(log)), bbox_inches='tight')

exit(0)
if not os.path.exists(os.path.join(opt_dir, "inner_log.txt")):
    exit(0)


bins = [0.1, 0.4, 0.5, 0.7, 1.0]
def buckets(freqs):
    sizes = [int(len(freqs)*b) for b in bins]
    sorted_freqs = np.argsort(freqs)[::-1]
    bucket_vocab = dict()
    for idx, actual_idx in enumerate(sorted_freqs):
        for ii, b in enumerate(sizes):
            if idx < b:
                bucket_vocab[actual_idx] = ii
                break
    return bucket_vocab

cutoffs = []
word2id = dict()
id2word = dict()
weights = dict()
alpha = 0.5

weights = None
bucket_scores = defaultdict(lambda: defaultdict(list))
with open(os.path.join(opt_dir, "inner_log.txt"), "r", encoding="utf-8") as fin:
    for line in fin:
        if line.startswith("Cutoff"):
            cutoff = int(line.strip().split("=")[0].split("-")[-1].strip())
            cutoffs.append(cutoff)
        if line.startswith("H-"):
            fields = line.strip().split("\t")
            lang = fields[0].strip().split('-')[-1]
            if lang == "x":
                lang = "en"
            else:
                lang = idx2lang[int(lang)]
            weights = list(map(float, fields[-1].strip().split()))
        if line.startswith("F-"):
            fracs = list(map(float, line.strip().split("\t")[-1].strip().split()))
            bucket_vocab = buckets(fracs)
            temp_bucket_weights = defaultdict(list)
            if weights is None:
                weights = np.ones(len(fracs)) * 0.1
                weights[:cutoff] = 1.0 / alpha
            for idx in range(len(fracs)):
                temp_bucket_weights[bucket_vocab[idx]].append(weights[idx])

            for k, v in temp_bucket_weights.items():
                bucket_scores[lang][k].append(np.mean(v))

plot_langs = bucket_scores.keys()
if len(plot_langs) == 1:
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 5)
    row = 1
else:
    assert len(plot_langs) == lang_num
    row = lang_num // 2
    fig, ax = plt.subplots(len(plot_langs) // row, row)
    fig.set_size_inches(30, 10)


bins_legends = [0] + bins
legends = ["{}%-{}%".format(bins_legends[ii-1], bins_legends[ii]) for ii in range(1, len(bins_legends))]

for idx, lang in enumerate(bucket_scores.keys()):
    i, j = idx // row, idx % row
    colors = plt.cm.jet(np.linspace(0, 1, len(bucket_scores[lang].keys())))
    assert len(legends) == len(bucket_scores[lang].keys())
    max_value = 0
    for ii in range(len(bins)):
        y_weight = bucket_scores[lang][ii]
        if max(y_weight) > max_value:
            max_value = max(y_weight)
        ax[i][j].plot(np.arange(len(y_weight)), y_weight, 'o', markersize=1, color=colors[ii])

    max_value += 0.2
    ax[i][j].legend(legends, loc='best', fontsize=10)
    ax[i][j].set_ylim([0, max_value])
    ax[i][j].set(title=lang, xlabel="steps", ylabel="avg ema weights")
fig.savefig(os.path.join(opt_dir, "{}_token_bucket_ema_weights.pdf".format(log)), bbox_inches='tight')
