import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os

log = sys.argv[1]
root = "/Users/chuntinz/Documents/research/fairseq-dro-mnmt/analysis/"
opt_dir = os.path.join(root, log)
if not os.path.exists(opt_dir):
    os.mkdir(opt_dir)
os.system("scp fair:/private/home/chuntinz/work/fairseq-dro-mnmt/saved_models/{}/log.txt {}".format(log, os.path.join(opt_dir, "log")))


langs = []
lang_train_size = dict()
lang_ema_losses = dict()
lang_ema_weights = dict()
lang_idx = 0
sum_train = 0
train_losses = []
train_ppl = []
valid_ppl = []
lang2idx = {}
idx2lang = {}

with open(os.path.join(opt_dir, "log").format(root, log)) as fin:
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
                    lang2idx = {lang:idx for idx, lang in enumerate(sorted(list(set(langs))))}
                    idx2lang = {v:k for k, v in lang2idx.items()}

        if "INFO | fairseq.data.multilingual.multilingual_data_manager |" in line and "train " in line:
            lang = line.strip().split(" ")[-3].split("-")[lang_idx]
            size = float(line.strip().split(" ")[-2].strip())
            lang_train_size[lang] = size
            sum_train += size

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
            valid_ppl.append(float(line.strip().split("|")[-6].strip().split()[-1].strip()))

        if " EMA past losses: tensor" in line:
            losses = line.strip().split("([")[-1].rstrip("],").split(",")
            for idx, loss in enumerate(losses):
                lang_ema_losses[idx2lang[idx]].append(float(loss.strip()))

        if " Group loss weights: tensor" in line:
            weights = line.strip().split("([")[-1].rstrip("],").split(",")
            for idx, weight in enumerate(weights):
                lang_ema_weights[idx2lang[idx]].append(float(weight.strip()))

print(idx2lang)
legends = ["{}={:.3f}".format(lang, lang_train_size[lang]/sum_train) for lang in langs]

labelsize = 14
legendsize = 14
mpl.rcParams['xtick.labelsize'] = labelsize
mpl.rcParams['ytick.labelsize'] = labelsize
mpl.rcParams['font.size'] = labelsize
plt.style.use('seaborn-deep')
colormap = plt.cm.gist_ncar


def plot_ax_ema(ax, y, title, scatter=False):
    plt.gca().set_prop_cycle(plt.cycler('color', plt.cm.jet(np.linspace(0, 1, len(langs)))))
    for lang in langs:
        if scatter:
            ax.plot(np.arange(len(y[lang])), y[lang], 'o', markersize=1)
        else:
            ax.plot(np.arange(len(y[lang])), y[lang])
    ax.legend(legends, loc='best', fontsize=10)
    ax.set(title=title, xlabel="steps", ylabel=title)


def plot_ax(ax, y, title, ylabel, color, step=150):
    ax.plot(np.arange(len(y)), y, color=color)
    ax.set(title=title, xlabel="steps", ylabel=ylabel)
    ax.xaxis.set_ticks(np.arange(0, max(np.arange(len(y)), ), step))

# plot train loss, (train_ppl and valid_ppl)
fig, ax = plt.subplots(1, 3)
fig.set_size_inches(30, 5)
x = list(range(len(train_losses)))
colors = ["royalblue", "indianred", "mediumseagreen"]

plot_ax(ax[0], train_losses, "train_losses", "losses", colors[0])
plot_ax(ax[1], train_ppl, "train_ppl", "ppl", colors[1])
plot_ax(ax[2], valid_ppl, "valid_ppl", "ppl", colors[2], 50)
fig.savefig(os.path.join(opt_dir, "loss_ppl.pdf"), bbox_inches='tight')

# plot ema losses, ema weights
fig, ax = plt.subplots(1)
fig.set_size_inches(20, 10)
plot_ax_ema(ax, lang_ema_losses, "ema_loss")
# plot_ax_ema(ax[1], lang_ema_weights, "ema_weights", True)
fig.savefig(os.path.join(opt_dir, "ema_loss.pdf"), bbox_inches='tight')

labelsize = 10
legendsize = 10
mpl.rcParams['xtick.labelsize'] = labelsize
mpl.rcParams['ytick.labelsize'] = labelsize
mpl.rcParams['font.size'] = labelsize
row = 4
fig, ax = plt.subplots(len(langs) // row, row)
fig.set_size_inches(30, 10)
colors = plt.cm.jet(np.linspace(0, 1, len(langs)))
for idx, lang in enumerate(langs):
    i, j = idx // row, idx % row
    print(len(lang_ema_weights[lang]))
    ax[i][j].plot(np.arange(len(lang_ema_weights[lang])), lang_ema_weights[lang], 'o', markersize=1, color=colors[idx])
    # print(legends[idx])
    # ax[i][j].legend(legends[idx], loc='best', fontsize=10)
    ax[i][j].set(title=legends[idx], xlabel="steps", ylabel="ema_weights")
fig.savefig(os.path.join(opt_dir, "ema_weights.pdf"), bbox_inches='tight')