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
usable_subinfo = subinfo[subinfo.behav_usable == 1].reset_index(drop=True)

subIDs = list(usable_subinfo.subid)
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
