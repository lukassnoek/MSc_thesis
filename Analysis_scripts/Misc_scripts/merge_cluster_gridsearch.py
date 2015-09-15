__author__ = 'lukas'

import sys
import pandas as pd
import numpy as np
from os.path import join as opj
import os
import glob

home = os.path.expanduser('~')
feat_dir = opj(home, 'DecodingEmotions')
ROI_dir = opj(home, 'ROIs')
os.chdir(feat_dir)

csvs = glob.glob(opj(feat_dir, 'opt_clust', '*csv'))

df_list = []
for csv in csvs:
    df_list.append(pd.read_csv(csv, sep='\t'))

df = pd.concat(df_list)

score = []
df_list2 = []
for fs_arg in np.unique(df['zval']):
    for cluster_min in np.unique(df['cluster_min']):
        tmp = df[df['zval'] == fs_arg]
        tmp = tmp[tmp['cluster_min'] == cluster_min]
        s = np.mean(tmp['score'])
        df_tmp = {'score': s, 'zval': fs_arg, 'cluster_min': cluster_min}
        df_tmp = pd.DataFrame(df_tmp, index=[0])
        df_list2.append(df_tmp)

df_list2 = pd.concat(df_list2)

filename = opj(feat_dir, 'opt_clust', 'mean_cluster_optimization_results.csv')
with open(filename, 'w') as f:
    df_list2.to_csv(f, header=True, sep='\t', index=False)
