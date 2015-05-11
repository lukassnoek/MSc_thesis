# -*- coding: utf-8 -*-
"""
Module with functions to call FSL command line utilities

Lukas Snoek
"""

import glob
import os

def call_flirt(file_type):
    '''Docstring: to insert'''

    feat_dirs = [os.path.abspath(fpath) for fpath in glob.glob(os.getcwd() + '/*/*.feat')]
    
    for fDir in feat_dirs:
        reg_dir = fDir + '/reg'
        stats_dir = fDir + '/stats'
    
        if not os.path.exists(reg_dir):
            print 'There is no registration info; skipping!'
        elif not os.path.exists(stats_dir):
            print 'The stats directory seems to be missing; skipping!'

        stats_mni_dir = os.path.abspath(fDir) + '/stats_new'
        if not os.path.exists(stats_mni_dir):
            os.mkdir(stats_mni_dir)
            print "Created " + stats_mni_dir
                
        mni_file = reg_dir + '/standard.nii.gz'
        premat_file = reg_dir + '/example_func2highres.mat'
        
        if file_type == 'res4d':
            warp_file = reg_dir + '/highres2standard_warp.nii.gz'
            to_transform = stats_dir + '/res4d.nii.gz'
            out_name = stats_mni_dir + '/res4d_mni.nii.gz'
        elif file_type == 'cope':
            warp_file = reg_dir + '/example_func2standard_warp.nii.gz'
            to_transform = glob.glob(stats_dir + '/cope*.nii.gz')
        elif file_type == 'varcope':
            warp_file = reg_dir + '/example_func2standard_warp.nii.gz'
            to_transform = glob.glob(stats_dir + '/varcope*.nii.gz')

        if file_type == 'cope' or file_type == 'varcope':        
            for c_file in to_transform:        
                out_name = stats_mni_dir + '/' + os.path.basename(c_file)[:-7] + "_mni" + '.nii.gz'
                os.system("applywarp --ref=" + mni_file + 
                      " --in=" + c_file + 
                      " --out=" + out_name + 
                      " --warp=" + warp_file + 
                      " --interp=trilinear")
            print "Transformed " + str(len(to_transform)) + " files"         
        
        elif file_type == 'res4d':
            os.system("applywarp --ref=" + mni_file + 
                      " --in=" + to_transform + 
                      " --out=" + out_name + 
                      " --warp=" + warp_file +
                      " --premat=" + premat_file +
                      " --interp=trilinear")


        

