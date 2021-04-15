import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import os
from collections import defaultdict
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr

labelsize = 14
legendsize = 14
mpl.rcParams['xtick.labelsize'] = labelsize
mpl.rcParams['ytick.labelsize'] = labelsize
mpl.rcParams['font.size'] = labelsize
plt.style.use('seaborn-deep')
colormap = plt.cm.gist_ncar

root = "/home/chuntinz/tir5/logs"
group_level = "target"
model_path = "/home/chuntinz/tir5/temp/61_erm_train_dynamics_wmt4_de_o2m"

# temporary
cumulated_sizes = [2500000, 4300000, 4812608, 5008370]
langs = ['de', 'fr', 'ta', 'tr']

# cumulated_sizes = [5664, 15504, 34302, 59637, 193964, 368408]
# langs = ['bos', 'mar', 'hin', 'mkd', 'ell', 'bul']
# model_path = "/Users/chuntinz/Documents/research/fairseq-dro-mnmt/61_debug_diverse_o2m"

num_langs = len(langs)
n_bins = 10


def get_confidence_and_variability(epochs_of_vecs, s=-1, e=-1):
    if s != -1 and e != -1:
        mat = np.vstack(epochs_of_vecs[s:e+1])
    else:
        mat = np.vstack(epochs_of_vecs)
    mu = np.mean(mat, axis=0)
    var = np.std(mat, axis=0)
    return mu, var


def plot_histogram(values, titles, stat="mu"):
    fig, ax = plt.subplots(num_langs, 3)
    fig.set_size_inches(30, 5 * num_langs)
    colors = ["royalblue", "indianred", "mediumseagreen"]
    for ii, lang in enumerate(langs):
        sid = 0 if ii == 0 else cumulated_sizes[ii - 1]
        eid = cumulated_sizes[ii]

        for jj, (vec, title) in enumerate(zip(values, titles)):
            ax[ii][jj].hist(vec[sid:eid], bins=n_bins, color=colors[jj], label=lang)
            ax[ii][jj].set(xlabel=title+"_"+stat, ylabel='Density')
            ax[ii][jj].legend(loc="best", fontsize=13)
    fig.savefig(os.path.join(model_path, "{}_hist.pdf".format(stat)), bbox_inches='tight')


def create_pd_frame_dict(mu, std):
    res = {}
    column_names = ['index', 'confidence', 'variability']
    for ii, lang in enumerate(langs):
        sid = 0 if ii == 0 else cumulated_sizes[ii - 1]
        eid = cumulated_sizes[ii]

        ds_mu = mu[sid:eid]
        ds_std = std[sid:eid]
        res[lang] = pd.DataFrame([[i, ds_mu[i], ds_std[i]] for i in range(len(ds_mu))], columns=column_names)
    return res


def process_epochs(eid_start, eid_end, plot_data_maps=True, compare=None):
    # compare = [(1, 30), (50, 80)]
    avg_ents = []
    avg_probs = []
    med_probs = []

    for eid in range(eid_start, eid_end+1):
        avg_ents.append(np.load(os.path.join(model_path, "avg_ent_{}.npy".format(eid))))
        avg_probs.append(np.load(os.path.join(model_path, "avg_probs_{}.npy".format(eid))))
        med_probs.append(np.load(os.path.join(model_path, "med_probs_{}.npy".format(eid))))

    ds_ent_mus, ds_ent_std = get_confidence_and_variability(avg_ents)
    ds_avg_prob_mus, ds_avg_prob_std = get_confidence_and_variability(avg_probs)
    ds_med_prob_mus, ds_med_prob_std = get_confidence_and_variability(med_probs)

    plot_histogram([ds_avg_prob_mus, ds_med_prob_mus, ds_ent_mus], ["avg_prob", "median_prob", "entropy"])
    plot_histogram([ds_avg_prob_std, ds_med_prob_std, ds_ent_std], ["avg_prob", "median_prob", "entropy"], "std")

    if plot_data_maps:
        ent_dfs = create_pd_frame_dict(ds_ent_mus, ds_ent_std)
        avg_prob_dfs = create_pd_frame_dict(ds_avg_prob_mus, ds_avg_prob_std)
        med_prob_dfs = create_pd_frame_dict(ds_med_prob_mus, ds_med_prob_std)
        for lang in langs:
            plot_data_map(ent_dfs[lang], 'ent_{}_dm.pdf'.format(lang))
            plot_data_map(avg_prob_dfs[lang], "avg_prob_{}_dm.pdf".format(lang))
            plot_data_map(med_prob_dfs[lang], 'med_prob_{}_dm.pdf'.format(lang))

    if compare is not None:
        keys = ['ent_mu', 'ent_std', 'avg_prob_mu', 'avg_prob_std', 'med_prob_mu', 'med_prob_std']
        for s, e in compare:
            ds_ent_mus_partial, ds_ent_std_partial = get_confidence_and_variability(avg_ents, s, e)
            psr = pearsonr(ds_ent_mus_partial, ds_ent_mus)
            print("Pearson correlation of {} between epochs {} - {} with full epochs = {}".format(keys[0], s, e, psr))
            psr = pearsonr(ds_ent_std_partial, ds_ent_std)
            print("Pearson correlation of {} between epochs {} - {} with full epochs = {}".format(keys[1], s, e, psr))

            ds_avg_prob_mus_partial, ds_avg_prob_std_partial = get_confidence_and_variability(avg_probs, s, e)
            psr = pearsonr(ds_avg_prob_mus_partial, ds_avg_prob_mus)
            print("Pearson correlation of {} between epochs {} - {} with full epochs = {}".format(keys[2], s, e, psr))
            psr = pearsonr(ds_avg_prob_std_partial, ds_avg_prob_std)
            print("Pearson correlation of {} between epochs {} - {} with full epochs = {}".format(keys[3], s, e, psr))

            ds_med_prob_mus_partial, ds_med_prob_std_partial = get_confidence_and_variability(med_probs, s, e)
            psr = pearsonr(ds_med_prob_mus_partial, ds_med_prob_mus)
            print("Pearson correlation of {} between epochs {} - {} with full epochs = {}".format(keys[4], s, e, psr))
            psr = pearsonr(ds_med_prob_std_partial, ds_med_prob_std)
            print("Pearson correlation of {} between epochs {} - {} with full epochs = {}".format(keys[5], s, e, psr))


def plot_data_map(dataframe: pd.DataFrame,
                  title: str = '',
                  max_instances_to_plot=55000):
    # Set style.
    sns.set(style='whitegrid', font_scale=1.6, font='Georgia', context='paper')

    # Subsample data to plot, so the plot is not too busy.
    dataframe = dataframe.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))

    main_metric = 'variability'
    other_metric = 'confidence'

    fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))

    plot = sns.scatterplot(x=main_metric,
                           y=other_metric,
                           ax=ax0,
                           data=dataframe,
                           s=30)
    plot.set_xlabel('variability')
    plot.set_ylabel('confidence')

    fig.tight_layout()
    filename = os.path.join(model_path, title)
    fig.savefig(filename, dpi=300)


process_epochs(1, 73, plot_data_maps=True, compare=[(1, 20), (30, 50), (53, 63)])


