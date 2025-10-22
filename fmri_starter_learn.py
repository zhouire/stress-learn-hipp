import numpy as np
import pandas as pd
import scipy.stats as st 
from matplotlib import pyplot as plt
from scipy import stats
import seaborn as sns
from plotnine import *
import statsmodels.formula.api as sm
from utils import *


subinfo = pd.read_csv('./data/subject_info.txt', sep="\t")
usable_subinfo = subinfo[subinfo.fmri_scenes_usable == 1].reset_index(drop=True)
usable_subinfo = usable_subinfo[usable_subinfo.subid != 51].reset_index(drop=True)

# filter out subjects with low response rate
subIDs = []
for sub in list(usable_subinfo.subid):
    behav_df = pd.read_csv(f'./data/behavior/catStats_sub{sub}.csv', sep='\t')
    goodresp_prop = len(behav_df[behav_df["rt"] < 1])/len(behav_df)
    if goodresp_prop > 0.6:
        subIDs.append(sub)
usable_subinfo = usable_subinfo[np.isin(usable_subinfo["subid"], subIDs)].reset_index(drop=True)

# split into control and stress groups
ctrl_subIDs = list(usable_subinfo[usable_subinfo.group == 0].subid)
stress_subIDs = list(usable_subinfo[usable_subinfo.group == 1].subid)

print("Controls: ",len(ctrl_subIDs))
print("Stress: ",len(stress_subIDs))

runs = ["rest1", "rest2", "scene1", "scene2", "scene3", "scene4", "scene5"]

subid2group = subinfo[["subid", "group"]].rename(columns={'subid': 'subID'})
def convertgrp(grp):
    if grp == 0:
        return "Control"
    elif grp == 1:
        return "Stress"
    else:
        return None
        
subid2group["group"] = subid2group["group"].apply(convertgrp)
