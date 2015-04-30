# -*- coding: utf-8 -*-
"""
Module to create the directory structure for the nipype processing pipeline
"""

import os
import glob
import shutil
import warnings 

def convert_parrec2nii(directory, remove_nifti = 1):
    """Converts Philips PAR/REC files to nifti files"""
    
    os.chdir(directory)
    os.system("parrec2nii *.PAR")
    os.system("gzip *.nii")
    nPR = len(glob.glob('*.PAR'))
    print "Converted {0} files".format(nPR)

    if remove_nifti:
        for file in glob.glob('*.nii'):
            os.remove(file)

def backup_parrec(directory):
    """Create back-up dir with all original par/recs"""
    os.chdir(directory)
    curDir = os.getcwd()
    PARREC_backup = curDir + '/PARREC_backup'

    if os.path.exists(PARREC_backup) == 0:
        os.mkdir(PARREC_backup)

    parFiles = glob.glob('*.PAR')
    recFiles = glob.glob('*.REC')

    for file in parFiles:
        shutil.move(file, PARREC_backup)
    print "Back-up completed for {0} files".format(len(parFiles))

    for file in recFiles:
        shutil.move(file, PARREC_backup)

def movefiles_ToProcess(directory, subject_stem):
    """Move files to ToProcess"""
    os.chdir(directory)
    ToProcess = os.getcwd() + '/ToProcess'

    if os.path.exists(ToProcess) == 0:
        os.mkdir(ToProcess)

    # Move all files to ToProcess dir
    nFiles = 0;    
    for file in glob.glob(subject_stem + '*'):
        shutil.move(file, ToProcess)
        nFiles =+ 1
    
    print "Moved {0} files to ToProcess".format(nFiles)
    
    # Make dir for bash-files (BET bashfile-etc)
    Log_dir = os.getcwd() + '/Logs'
    if os.path.exists(Log_dir) == 0:
        os.mkdir(Log_dir)
        
def movefiles_subjectdirs(subject_stem, ToProcess):
    '''
    Moves files to subject specific directories and creates run-specific
    subdirectories
    '''
    os.chdir(ToProcess)
    allFiles = glob.glob(subject_stem + '*')
    prefixes = [pref[0:7] for pref in allFiles]
    subjects = list(set(prefixes))
    subjectDir = []

    for idx, subs in enumerate(subjects):
    
        subjectDir.append(subs[0:len(subject_stem) + 3])
    
        if os.path.exists(subjectDir[idx]) == 0:
            os.mkdir(subjectDir[idx])
    
        toMove = glob.glob(subs[0:len(subject_stem) + 3] + '*')
    
        # Move all files to subject-specific dirs
        for mFile in toMove:
            if os.path.isfile(mFile):
                shutil.move(mFile, subjectDir[idx])
            
            for entry in glob.glob('*'):
                if os.path.isfile(entry):
                    print "Not allocated: " + entry
                    
    subject_dir_list = [os.getcwd() + '/' + sDir for sDir in subjectDir]
    
    # Create subdirectories
    for subjectDir in subject_dir_list:
        os.chdir(subjectDir)
    
        mri_files = glob.glob('*.nii.gz')
        subdir_names = []
        
        for mriFile in mri_files:
            split_file = mriFile.split('_')
            from_idx = split_file.index('WIP')
            to_idx = split_file.index('SENSE')
            toAppend = "_".join(split_file[from_idx+1:to_idx])            
            subdir_names.append(toAppend)
            
            os.mkdir(toAppend)
            shutil.move(mriFile, toAppend)
        
        print "Created the following subdirs for {0}: ".format(os.path.basename(subjectDir))
        for subdir in subdir_names:
            print subdir
        print "\n"
        
def get_filepaths(keyword, directory):
    matches = []
    filenames_total = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if keyword in filename:
                matches.append(root + '/' + filename)
                filenames_total.append(filename)
    return matches, filenames_total