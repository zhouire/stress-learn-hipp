# stress-learn-hipp
Data and code for "Stress drives the hippocampus to prioritize statistical prediction over episodic encoding".

**Directories:**  
* `conda_env`: Contains YML files listing specific packages used in analysis. `environment_clean.yml` contains a clean export of packages used to run all analyses. `environment_mirror.yml` contains an exact mirror of the conda environment used for analysis.
* `data`: Contains preprocessed data used as starting points for the analyses described in the paper. Raw fMRI data is available in the NIMH Data Archive (https://dx.doi.org/10.15154/5sec-q259).  
* `midpoints`: Midpoint files generated using the provided notebooks. All midpoint files necessary to reproduce figures from the paper have been pre-computed for convenience. Code used to produce these files is provided in notebooks. 

**Notebooks:** All analyses in the paper can be reproduced by running the provided notebooks in the following order. Later notebooks have dependencies on midpoint files generated from earlier notebooks. Notebooks can be run in any order with the provided midpoint files.   
* `0_demographics.ipynb`: Summarizes and compares participant demographics and experimental parameters from the control and stress groups, as described in Methods.  
* `1_PANAS.pynb`: Analyzes PANAS data collected immediately before and after SECPT. Corresponds to Figure 1C and related results.  
* `2_cortisol.ipynb`: Analyzes salivary cortisol collected throughout the experiment. Corresponds to Figures 1D and 1E.  
* `3_pupil.ipynb`: Analyzes pupil size during statistical learning runs. Corresponds to Figure 2A.  
* `4_classifier.ipynb`: Trains subject-specific classifiers to distinguish B scene categories in the pre-learning run from hippocampal subfield activity patterns, which are used to assess prediction of upcoming B categories during A scene viewing in the post-learning run. Corresponds to Figure 2B and Supplemental Figure S1.
* `5_behavior.ipynb`: Analyzes behavioral data from the episodic memory (item recognition) and statistical learning (pair familiarity) tests on Day 2. Corresponds to Figure 3A and Supplemental Figure S5.  
* `6_patternsimilarity.ipynb`: Uses a multivariate pattern similarity approach to assess pattern separation and evidence of category representation in DG during learning runs. Corresponds to Figures 3B-E. Can be used to generate Supplemental Figures S2-4 with instructions provided in relevant sections.   
* `7_edgetimeseries.ipynb`: Uses an edge timeseries approach to determine if momentary cofluctuation between pairs of hippocampal subfields tracks episodic encoding and statistical learning during the learning runs. Corresponds to Figures 4A-B and Supplemental Figure S6. Can be used to generate Supplemental Figure S7 with instructions provided in relevant sections.  


*Notes:*  
Bootstrapped statistics may differ slightly from those reported in the paper due to minor variations in resampling.  


Irene Zhou, June 2026
