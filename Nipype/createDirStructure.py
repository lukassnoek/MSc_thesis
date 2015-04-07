# -*- coding: utf-8 -*-
"""
Module to create the directory structure for the nipype processing pipeline
"""

import os
import glob
import shutil

def convertParrec2nii(directory, remove_nifti):
    """Converts Philips PAR/REC files to nifti files"""
    os.chdir(directory)
    os.system("parrec2nii *.PAR")
    os.system("gzip *.nii")
    nPR = len(glob.glob('*.PAR'))
    print "Converted {0} files".format(nPR)

    if remove_nifti:
        for file in glob.glob('*.nii'):
            os.remove(file)

def backup_Parrec(directory):
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

def moveToProcess(directory):
    """Move files to ToProcess"""
    os.chdir(directory)
    ToProcess = os.getcwd() + '/ToProcess'

    if os.path.exists(ToProcess) == 0:
        os.mkdir(ToProcess)

    # Move all files to ToProcess dir
    for file in glob.glob(subject_stem + '*'):
        shutil.move(file, ToProcess)

    # Make dir for bash-files (BET bashfile-etc)
    Log_dir = os.getcwd() + '/Logs'
    if os.path.exists(Log_dir) == 0:
        os.mkdir(Log_dir)
