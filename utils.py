import numpy as np
import pandas as pd
import scipy.stats as st 
from matplotlib import pyplot as plt
from scipy import stats
from copy import deepcopy
import seaborn as sns
import statsmodels.formula.api as smf
import warnings
import types
import os
import scipy.io
from scipy._lib._util import rng_integers
from itertools import combinations

import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects import Formula


####################################
# PATHS
####################################

data_dir = './data'
questionnaires_dir = f'{data_dir}/questionnaires'
cortisol_dir = f'{data_dir}/cortisol'
eyetracking_dir = f'{data_dir}/pupil'
behav_dir = f'{data_dir}/behavior'
classifier_dir = f'{data_dir}/classifier'

panas_pre_file = f"{questionnaires_dir}/StressLearn_PANAS_preSECPT.csv"
panas_post_file = f"{questionnaires_dir}/StressLearn_PANAS_postSECPT.csv"
cortisol_data_file = f"{cortisol_dir}/StressLearn_Cortisol.csv"




####################################
# AESTHETICS
####################################

csfont = {'fontname':'Helvetica'}
hfont = {'fontname':'Helvetica'}
plt.rcParams["font.family"] = "Tahoma"

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

SMALL_SIZE = 15
MEDIUM_SIZE = 18
BIGGER_SIZE = 24

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

cmap = plt.get_cmap("plasma")
rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y) + 1)

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

ctrl_color = sns.desaturate(plt.get_cmap("GnBu_r")(0.55), 0.75)
stress_color = sns.desaturate(plt.get_cmap("RdPu_r")(0.5), 0.75)
group_colors = [ctrl_color, stress_color]
ABX_colors = ['#678BDA', '#56BED6', '#BEC3C6']

dark_ctrl_color = adjust_lightness(ctrl_color, 0.5)
dark_stress_color = adjust_lightness(stress_color, 0.5)
dark_group_colors = [dark_ctrl_color, dark_stress_color]
dark_ABX_colors = [adjust_lightness(p) for p in ABX_colors]

secpt_color = '#f069a0'
learn_color = '#72d276'
rest_color = '#a6a8ab'

pd.options.mode.chained_assignment = None


####################################
# BASIC STATS FUNCTIONS
####################################
def pval2star(p, centered=False):
    if centered:
        asterisk = u'\u2217'
    else:
        asterisk = '*'
    
    star = ""
    if p <= 0.001:
        star = asterisk*3
    elif p <= 0.01:
        star = asterisk*2
    elif p <= 0.05:
        star = asterisk

    return star


def simple_bootstrap(samples, iterations=9999, CI_percentile=95, null=0, onetail_hyp='any'):
    '''
    Simple bootstrapping with the percentile method
    Equivalent to: 
        scipy.stats.bootstrap(data = (samples,), statistic=np.mean, confidence_level=0.95, n_resamples=iterations, method='percentile')
    
    Args:
        samples (np.array[float]): a list of samples which we want to analyze
        iterations (int): number of resampling iterations
        CI_percentile (float): between (0, 1], the confidence interval percentile (default = 0.95)
        null (float): the null hypothesis value used to compute p-value
        onetail_hyp (str): ['below', 'above', 'any'] whether the 1-tailed hypothesis is below or above null (used for p-value)

    Returns:
        CI ([float, float]): confidence interval [low, high]
        pval (float): p-value
    '''
    dist = np.zeros(iterations)

    numsamples = len(samples)

    i = rng_integers(None, 0, numsamples, (iterations, numsamples))
    dist = np.mean(samples[..., i], -1)


    CI = [np.percentile(dist, (100-CI_percentile)/2), np.percentile(dist, 100-(100-CI_percentile)/2)]
    if onetail_hyp == 'below':
        pval = np.mean(dist >= null)
    elif onetail_hyp == 'above':
        pval = np.mean(dist <= null)
    elif onetail_hyp == 'any':
        pval1 = np.mean(dist <= null)
        pval2 = np.mean(dist >= null)
        pval = np.min([pval1, pval2])
    else:
        pval = None
        
    return CI, pval


def simple_bootstrap_2group(samples1, samples2, iterations=9999, CI_percentile=95, null=0, onetail_hyp='any'):
    '''
    Simple bootstrapping with the percentile method - resamples and compares means of two groups. 
    For each iteration, subtracts mean of resampled group 1 from mean of resampled group 2 and computes CIs and p-value from this difference

    Args:
        samples1 (list[float]): first list of samples
        samples2 (list[float]): second list of samples
        iterations (int): number of resampling iterations for both groups
        CI_percentile (float): between (0, 1], the confidence interval percentile (default = 0.95)
        null (float): the null hypothesis value used to compute p-value
        onetail_hyp (str): ['below', 'above', 'any'] whether the 1-tailed hypothesis is below or above null (used for p-value)

    Returns:
        CI ([float, float]): confidence interval [low, high]
        pval (float): p-value
    '''
    dist = np.zeros(iterations)

    numsamples1 = len(samples1)
    numsamples2 = len(samples2)

    i1 = rng_integers(None, 0, numsamples1, (iterations, numsamples1))
    i2 = rng_integers(None, 0, numsamples2, (iterations, numsamples2))
    dist1 = np.mean(samples1[..., i1], -1)
    dist2 = np.mean(samples2[..., i2], -1)


    CI = [np.percentile(dist1-dist2, (100-CI_percentile)/2), np.percentile(dist1-dist2, 100-(100-CI_percentile)/2)]
    if onetail_hyp == 'below':
        pval = np.mean(dist1-dist2 >= null)
    elif onetail_hyp == 'above':
        pval = np.mean(dist1-dist2 <= null)
    elif onetail_hyp == 'any':
        pval1 = np.mean(dist1-dist2 <= null)
        pval2 = np.mean(dist1-dist2 >= null)
        pval = np.min([pval1, pval2])
    else:
        pval = None
        
    return CI, pval



def weighted_subject_bootstrap(subjects, sub2samp, sub2trialnum, iterations=9999, CI_percentile=95, onetail="any", null=0):
    '''
    Computes confidence intervals and p-values via bootstrapping; resampling is performed at the subject, and subject means are weighted by the number of trials they contribute. 
    Outputs the trial-level mean, bootstrapped CIs, and bootstrapped p-value in comparison to null. 
    
    Args:
        subjects (list[int]): list of subject numbers
        sub2samp (dict{sub: np.array}): dictionary mapping subject to 1xn timeseries sequence
        sub2trialnum (dict{sub: numtrials}): dictionary mapping subject number to number of trials used for computing differentiation; set all to 1 for subject-level bootstrap

    Returns:
        grand_mean, CIs, pvals
    '''

    numtimepoints = sub2samp[subjects[0]].shape[-1]
    weighted_sub2samp = {s: sub2samp[s]*sub2trialnum[s] for s in subjects}
    
    # compute the mean HFA timeseries over pooled trials
    # mean HFA timeseries over pooled trials
    grand_mean = np.sum([weighted_sub2samp[s] for s in subjects], axis=0)/np.sum([sub2trialnum[s] for s in subjects])

    # resample over subjects and take the mean over pooled trials
    # start bootstrapping
    numsamples = len(subjects)

    # generate resampled subject IDs
    i = rng_integers(None, 0, numsamples, (iterations, numsamples))
    sub_dist = np.array(subjects)[..., i]

    mean_series = np.full([iterations, numtimepoints], np.nan)
    
    # loop through resampled subIDs, pool trials for each resample, and compute new mean
    for r in range(iterations):
        resamp = sub_dist[r]
        # pool the trials corresponding to the resampled subjects 
        tottrials = np.sum([sub2trialnum[s] for s in resamp])
        weightedsamps = np.sum([weighted_sub2samp[s] for s in resamp], axis=0)

        mean_series[r,:] = weightedsamps/tottrials

    CIs = []
    pvals = []
    # compute 95% confidence intervals and p-value compared to null at each timepoint
    for t in range(mean_series.shape[1]):
        cur_samples = mean_series[:, t]
        CI = [np.percentile(cur_samples, (100-CI_percentile)/2), np.percentile(cur_samples, 100-(100-CI_percentile)/2)]
        CIs.append(CI)

        if onetail == "below":
            pval = np.sum(cur_samples >= null)/iterations
        elif onetail == "above": 
            pval = np.sum(cur_samples <= null)/iterations
        elif onetail == "any":
            pval_raw = np.sum(cur_samples >= null)/iterations
            pval = np.min([pval_raw, 1-pval_raw], axis=0)

        pvals.append(pval)
        
    CIs = np.array(CIs).T
    pvals = np.array(pvals)

    return grand_mean, CIs, pvals


def mannwhitneyu_effsize(grp1, grp2, alternative="two-sided", verbose=False):
    '''
    alternative = one of ["two-sided", "less", "greater"]
    Returns rank-biserial correlation (r; https://doi.org/10.2466/11.IT.3.1)
    '''

    if alternative == "any":
        alternative = "two-sided"
    elif alternative == "above":
        alternative = "greater"
    elif alternative == "below":
        alternative = "less"

    out = scipy.stats.mannwhitneyu(grp1, grp2, alternative=alternative)
    U1 = out.statistic
    p = out.pvalue
    if verbose:
        print(f'U1 = {U1}; p = {p}')

    rbc = (2*U1)/(len(grp1)*len(grp2)) - 1

    return rbc

def wilcoxon_effsize(sample, alternative="two-sided", verbose=False):
    '''
    alternative = one of ["two-sided", "less", "greater"]
    Returns rank-biserial correlaton (r; https://doi.org/10.2466/11.IT.3.1)
    '''

    if alternative == "any":
        alternative = "two-sided"
    elif alternative == "above":
        alternative = "greater"
    elif alternative == "below":
        alternative = "less"

    out = scipy.stats.wilcoxon(sample, alternative=alternative)
    T = out.statistic
    p = out.pvalue
    if verbose:
        print(f'T = {T}; p = {p}')

    d = sample[sample != 0]
    r = scipy.stats.rankdata(abs(d))
    rsum = r.sum()
    r_plus = np.sum((d > 0) * r)
    r_minus = np.sum((d < 0) * r)
    rbc = r_plus / rsum - r_minus / rsum

    return rbc



####################################
# DATA LOADING
####################################

def labels2gencats(subID):
    lab2gen = {}
    
    AB_df = pd.read_csv(f'{behav_dir}/AB.csv', sep='\t')
    X_df = pd.read_csv(f'{behav_dir}/X.csv', sep='\t')
    
    As = list(AB_df[AB_df.subid == subID].A)
    Bs = list(AB_df[AB_df.subid == subID].B)
    Xs = list(X_df[X_df.subid == subID].X)
    
    for i in range(len(As)):
        lab2gen[As[i]] = "A" + str(i+1)
    for i in range(len(Bs)):
        lab2gen[Bs[i]] = "B" + str(i+1)
    for i in range(len(Xs)):
        lab2gen[Xs[i]] = "X" + str(i+1)
    
    return lab2gen


def get_ABX_labels(subid, run, exclude_noresp=True, nan_noresp=False):
    """
    Get ABX labels from CSVs in labels
    Args:
        subid (int): subject ID
        run (int): which scene run are these labels from (1-5)

    Returns:
        list: containing ABX label for each trial for given subject and run
    """
    behav_file = f'{behav_dir}/catStats_sub{subid}.csv'
    behav_data = pd.read_csv(behav_file, sep='\t')
    behav_data_run = behav_data[behav_data.block == run].reset_index(drop=True)
    
    if exclude_noresp:
        behav_data_run = behav_data_run[behav_data_run.encoded == 1]
    
    if nan_noresp:
        nanidxs = behav_data_run[behav_data_run.encoded == 0].index
        behav_data_run.iloc[nanidxs] = np.nan
    
    ABX_labels = list(behav_data_run.ABX)
    
    return ABX_labels


def get_category_labels(subid, run, exclude_noresp=True, nan_noresp=False):
    """
    Get category labels from CSVs in labels
    Args:
        subid (int): subject ID
        run (int): which scene run are these labels from (1-5)

    Returns:
        list: containing category label for each trial for given subject and run
    """
    behav_file = f'{behav_dir}/catStats_sub{subid}.csv'
    behav_data = pd.read_csv(behav_file, sep='\t')
    behav_data_run = behav_data[behav_data.block == run].reset_index(drop=True)
    
    if exclude_noresp:
        behav_data_run = behav_data_run[behav_data_run.encoded == 1]
    
    if nan_noresp:
        nanidxs = behav_data_run[behav_data_run.encoded == 0].index
        behav_data_run.iloc[nanidxs] = np.nan
    
    category_labels = list(behav_data_run.category)
    
    return category_labels


def get_recog_labels(subid, run, conf_thr=0, exclude_noresp=True, nan_noresp=False):
    """
    Get category labels from CSVs in labels
    Args:
        subid (int): subject ID
        conf_thr(int): threshold for confidence to be counted as remembered (exclusive; conf > conf_thr)
        run (int): which scene run are these labels from (1-5)

    Returns:
        list: containing category label for each trial for given subject and run
    """
    behav_file = f'{behav_dir}/catStats_sub{subid}.csv'
    behav_data = pd.read_csv(behav_file, sep='\t')
    behav_data_run = behav_data[behav_data.block == run].reset_index(drop=True)
    
    if exclude_noresp:
        behav_data_run = behav_data_run[behav_data_run.encoded == 1]
    
    if nan_noresp:
        nanidxs = behav_data_run[behav_data_run.encoded == 0].index
        behav_data_run.iloc[nanidxs] = np.nan
    
    recog_labels = list(behav_data_run.recogAcc)
    recog_conf = list(behav_data_run.recogConf)
    recog_labels_named = []
    for l in range(len(recog_labels)):
        if recog_labels[l] == 0:
            recog_labels_named.append("forgotten")
        elif recog_labels[l] == 1:
            if recog_conf[l] > conf_thr:
                recog_labels_named.append("remembered")
            else:
                recog_labels_named.append("lowconf")
        elif np.isnan(recog_labels[l]):
            recog_labels_named.append(np.nan)
        else:
            raise ValueError("recogAcc value is " + str(l) + ", must be 0 or 1")
    
    return recog_labels_named
                            

def get_recog_confidence(subid, run, exclude_noresp=True, nan_noresp=False):
    behav_file = f'{behav_dir}/catStats_sub{subid}.csv'
    behav_data = pd.read_csv(behav_file, sep='\t')
    behav_data_run = behav_data[behav_data.block == run].reset_index(drop=True)

    if exclude_noresp:
        behav_data_run = behav_data_run[behav_data_run.encoded == 1]
    
    if nan_noresp:
        nanidxs = behav_data_run[behav_data_run.encoded == 0].index
        behav_data_run.iloc[nanidxs] = np.nan

    confs = list(behav_data_run.recogConf)

    return confs


def get_stimulus_TRs(subid, run, exclude_noresp=True):
    """
    Get TRs from CSVs in labels. 
    This does NOT include shift to account for hemodynamic lag.
    Args:
        subid (int): subject ID
        run (int): which scene run are these labels from (1-5)

    Returns:
        list: containing 0-indexed TR for each trial for given subject and run
    """
    behav_file = f'{behav_dir}/catStats_sub{subid}.csv'
    behav_data = pd.read_csv(behav_file, sep='\t')
    behav_data_run = behav_data[behav_data.block == run]
    
    if exclude_noresp:
        behav_data_run = behav_data_run[behav_data_run.encoded == 1]
    
    TRs = list(behav_data_run.TR)
    
    return TRs
    

def get_cortisol_samples_df(stress_subIDs, ctrl_subIDs):
    '''
    Loads in and cleans up cortisol data, renaming cols, nan-ing outliers with |SD|>3, and returning a df with a sample for a subject on each row. 
    Args:
        stress_subIDs (list(int)): list of subIDs in the stress group
        ctrl_subIDs (list(int)): list of subIDs in the ctrl group
    '''

    subIDs = list(stress_subIDs) + list(ctrl_subIDs)
    
    cort_df = pd.read_csv("saliva/StressLearn_Cortisol.csv")[["Subject", "Label", "Time", "Cortisol (nmol/l) Mean"]]
    cort_df = cort_df.rename(columns={"Cortisol (nmol/l) Mean": "mean_cort", "Subject": "subID", "Label": "label", "Time": "time"})
    cort_df = cort_df[cort_df["subID"].isin(subIDs)]

    sub2group = {}
    for s in stress_subIDs:
        sub2group[s] = "Stress"
    for s in ctrl_subIDs:
        sub2group[s] = "Control"

    cort_df["day"] = cort_df["label"].apply(lambda s: s[0])
    cort_df["sample"] = cort_df["label"].apply(lambda s: s[2])
    cort_df["group"] = cort_df["subID"].apply(lambda s: sub2group[s])
    cort_d1_df = cort_df.copy()[cort_df["day"] == '1']
    cort_d1_df["z_cort"] = scipy.stats.zscore(list(cort_d1_df["mean_cort"]), nan_policy='omit')
    
    cort_d1_df_out = cort_d1_df[["subID", "group", "sample", "mean_cort", "z_cort"]].reset_index(drop=True)

    # nan out any outlier rows with z_cort > 3
    outlier_idx = cort_d1_df_out[np.abs(cort_d1_df_out.z_cort) > 3].index
    cort_d1_df_out.loc[outlier_idx, "z_cort"] = np.nan
    cort_d1_df_out.loc[outlier_idx, "mean_cort"] = np.nan

    return cort_d1_df_out

def get_cortisol_reactivity_df(stress_subIDs, ctrl_subIDs):
    '''
    Computes cortisol reactivity by taking the natural log of cort values from Day 1, averaging over samples 3 and 4, and subtracting sample 1
    '''
    
    cort_d1_df_out = get_cortisol_samples_df(stress_subIDs, ctrl_subIDs)
    
    # compute reactivity
    cort_d1_df_react = cort_d1_df_out[["subID", "group", "sample", "mean_cort"]]
    cort_d1_df_react["log_cort"] = np.log(cort_d1_df_react["mean_cort"])
    cort_d1_df_react = cort_d1_df_react.pivot(index=["subID","group"], columns="sample", values="log_cort").reset_index()
    cort_d1_df_react["cort_react"] = (cort_d1_df_react["4"] + cort_d1_df_react["3"])/2 - cort_d1_df_react["1"]
    
    return cort_d1_df_react


####################################
# VISUALIZATION
####################################

def get_star_x_y(ax, df, xcol, huecol, valcol, xorder, hueorder, points=True):
    barxs = np.reshape([line.get_data()[0][0] for line in ax.get_lines()[:len(hueorder)*len(xorder)]], (len(hueorder), len(xorder))).T
    if points:
        barys = []
        for hue in hueorder:
            for x in xorder:
                barys.append(np.max(df.loc[(df[xcol] == x) & (df[huecol] == hue)][valcol]))
        barys = np.reshape(barys, (len(hueorder), len(xorder))).T
        
    else:
        barys = np.reshape([line.get_data()[1][1] for line in ax.get_lines()[:len(hueorder)*len(xorder)]], (len(hueorder), len(xorder))).T

    return barxs, barys


def add_sig_compare_line(ax, linex, stary, yshift=None):
    if yshift is None:
        yshift = np.diff(ax.get_ylim())[0]*0.05
    
    ax.hlines(y=stary, xmin=linex[0], xmax=linex[1], color='k', linewidth=1)
    ax.vlines(x=linex[0], ymin=stary-yshift*0.2, ymax=stary, color='k', capstyle='projecting', linewidth=1)
    ax.vlines(x=linex[1], ymin=stary-yshift*0.2, ymax=stary, color='k', capstyle='projecting', linewidth=1)
    

def lm_bootstrap(df, x, y, iterations=9999, CI_percentile=95, two_tailed=True, verbose=False):
    '''
    Args:
        df (pd.DataFrame): dataframe mapping subject IDs to x and y values. Must contain column "subID"
        x (string): name of column in df for x axis
        y (string): name of column in df for y axis
        iterations (int): number of resampling iterations
    '''

    df = df[['subID', x, y]].rename(columns={x:'x', y:'y'}).dropna()

    # get the true x coefficient
    true_coeff = scipy.stats.pearsonr(df['x'], df['y'])[0] #.statistic
    if verbose:
        print(f'r={true_coeff}')

    subjects = df['subID']
    numsamples = len(subjects)
    
    # generate resampled subject IDs
    sub_dist = rng_integers(None, 0, numsamples, (iterations, numsamples))

    resamp_p = []
    resamp_coeffs = []
    
    # loop through resampled subIDs, pool trials for each resample, and compute new regression
    for resamp in sub_dist:
        # pool the trials corresponding to the resampled subjects 
        cur_df = df.iloc[resamp]
        
        # compute a regression with linear and quadratic terms (x=reinstatement, y=differentiation), and output the regression coefficients as a Series
        resamp_coeff = scipy.stats.pearsonr(cur_df['x'], cur_df['y'])[0] #.statistic
        resamp_coeffs.append(resamp_coeff)
        resamp_p.append(np.sign(resamp_coeff) != np.sign(true_coeff))

    resamp_p = np.mean(resamp_p)
    if two_tailed:
        resamp_p = resamp_p*2

    CI = [np.percentile(resamp_coeffs, (100-CI_percentile)/2), np.percentile(resamp_coeffs, 100-(100-CI_percentile)/2)]

    return CI, resamp_p

    

def plot_lm(df, xvar, yvar, groups, colormap=None, pretty=False, savefile=None, legend=False, boot=False, figsize=(4, 4), verbose=False):
    '''
    Args:
        df (pd.DataFrame): dataframe
        xvar (str): column from df for the x-axis
        yvar (str): column from df for the y-axis
        groups (list(str)): subset of ["Stress", "Control"] or None; if None, collapses over groups
        xname (str): x-axis label (if None, defaults to xvar)
        yname (str): y=axis label (if None, defaults to yvar)
        colormap (dict): maps groups to colors for sns.regplot (scatterplot points and linear regression line); use default if None
        pretty (bool): if True, seaborn uses a much higher resampling number for bootstrap, for prettier confidence intervals
        savepath (str): where to save figure, if at all
    '''

    cur_dfs = []
    colors = []

    if colormap is None:
        colormap = {"Stress": stress_color, 
                    "Control": ctrl_color, 
                    None: "mediumpurple"}

    if groups is not None:
        for group in groups:
            cur_dfs.append(df[df.group == group].rename(columns = {xvar: "x", yvar: "y"}))
            colors.append(colormap[group])
    else:
        groups = ["Both"]
        cur_dfs.append(df.copy().rename(columns = {xvar: "x", yvar: "y"}))
        colors.append(colormap[None])

    fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    if pretty:
        n_boot = 100000
    else:
        n_boot = 1000

    for i in range(len(cur_dfs)):
        cur_df = cur_dfs[i]
        color = colors[i]
        group = groups[i]
        
        ax.set_zorder(i*100)
        sns.regplot(ax=ax, 
                    data=cur_df, 
                    x="x", 
                    y="y", 
                    color=color, 
                    ci=95, 
                    n_boot=n_boot, 
                    label=None, 
                    scatter_kws={"s": 20, "alpha": 0.4, "edgecolor": "None", 'label': None}, 
                    line_kws={'solid_capstyle': 'butt', 'label': group})

        ax.collections[-2].set_zorder(-i*5)
        ax.lines[-1].set_zorder((i+1)*5)
        ax.collections[-1].set_zorder((i+2)*5)
        ax.collections[-1].set_alpha(0.25)
        
        model = smf.ols(formula=f'y ~ x', data=cur_df).fit()
        if verbose:
            print(model.summary())

        if not boot:
            pval = model.pvalues["x"]
        else:
            print(f'{group}:')
            CI, pval = lm_bootstrap(cur_df, 'x', 'y', iterations = 9999, verbose=True)
            print(f'95% CI {CI}; p={pval}\n')

        starx = ax.lines[-1].get_data()[0][-1] + plt.xlim()[1]*0.02#plt.xlim()[1]
        stary = model.params["x"]*starx + model.params["Intercept"]
        star = pval2star(pval, centered=True)
        ax.text(x=starx, y=stary, s=star, fontname='DejaVu Sans', horizontalalignment='left', verticalalignment='center', fontsize=15)
    
    ax.set_xlabel(xvar)
    ax.set_ylabel(yvar)
    if legend and len(cur_dfs) > 1:
        ax.legend(fontsize=15, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
        
    sns.despine()

    if savefile is not None:
        plt.savefig(savefile)

    return ax



def r_lme(lme_df, formula, random_formula="~ 1 | subid", effsize=False):
    data = lme_df
    
    nlme = importr('nlme')
    base = importr('base')
    rstats = importr('stats')
    if effsize:
        lme4 = importr('lme4')
        sjstats = importr('sjstats')

    with (ro.default_converter + pandas2ri.converter).context():
      r_data = ro.conversion.get_conversion().py2rpy(data)
    
    formula_lme = Formula(formula)
    random_lme = Formula(random_formula)
    
    model_lme = nlme.lme(fixed=formula_lme, random=random_lme, data=r_data)
    anova_lme = rstats.anova(model_lme)

    if effsize:
        formula_lmer = Formula(f'{formula} + ({random_formula[1:]})')
        print(formula_lmer)
        model_lmer = lme4.lmer(formula=formula_lmer, data=r_data)
        effsize_lme = sjstats.anova_stats(model=model_lmer)
    
    with (ro.default_converter + pandas2ri.converter).context():
        anova_lme = ro.conversion.get_conversion().rpy2py(anova_lme)
        if effsize:
            effsize_lme = ro.conversion.get_conversion().rpy2py(effsize_lme)

    if effsize:
        return anova_lme, effsize_lme
    else:
        return anova_lme