__author__ = 'c6386806'

import sys
import platform
from os.path import expanduser
home = expanduser("~")
import os

if 'Windows' in platform.platform():
    data_dir = os.path.join('Z:/','fMRI Projects','fMRI Project DynamicAffect')
    script_dir = os.path.join(home,'Dropbox','ResMas_UvA','Thesis','Git',
                              'Analysis_scripts')

if 'precise' in platform.platform():
    data_dir = os.path.join(home,'.gvfs','storage on bigbrother','fMRI Projects','fMRI Project DynamicAffect')
    script_dir = os.path.join(home,'Git','Analysis_scripts')

if 'vivid' in platform.platform():
    data_dir = os.path.join('/run/user/1000/gvfs',
                            'smb-share:server=bigbrother.fmg.uva.nl,share=fmri_projects$',
                            'fMRI Project DynamicAffect')
    script_dir = os.path.join(home,'Dropbox','ResMas_UvA','Thesis','Git',
                              'Analysis_scripts')

sys.path.append(script_dir)
#######################################

import nibabel as nib
import pandas as pd

file_name = os.path.join(data_dir,'diff_vec.nii.gz').replace(' ','\ ')
cluster_file = os.path.join(data_dir,'clustered').replace(' ','\ ')
size_file = os.path.join(data_dir,'cluster_size').replace(' ','\ ')
info_file = os.path.join(data_dir,'cluster_info').replace(' ','\ ')

command = 'cluster -i %s -o %s -t %f --osize=%s > %s' % \
          (file_name, cluster_file, 2.3, size_file, info_file)
os.system(command)

df = pd.read_table(info_file.replace('\ ',' '), sep = '\t', header = 0, skip_blank_lines=True)
print df['Voxels']

