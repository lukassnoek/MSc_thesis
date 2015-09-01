"""
Extracts info from cluster-corrected files
"""

import os
from os.path import join as opj
import nibabel as nib
import numpy as np
import glob
import pandas as pd


def main(gfeat, gfeat_clust, mask_list, thres):
    """
    Extracts info from activation-map per ROI

    Args:
        gfeat (str): statistic-file which info needs to be extracted
        gfeat_clust (str): clustered file corresponding to gfeat
        mask_list (list): list of ROIs which is looped over
        thres (int/float): lower-bound for probabilistic masks such as Harvard-Oxford

    Returns:
        Nothing (but creates gfeat_info file in pwd)
    """

    # Create file
    filename = opj(os.getcwd(), 'roi_info.csv')
    f = open(filename, 'w')
    f.write('ROI\tk\t Max(X)\tMAX(Y)\tMAX(Z)\tmax. val.\n')

    # Load activation-file and corresponding cluster file
    # Cluster-file is needed because ROI may contain > 1 clusters
    stat = nib.load(gfeat).get_data()
    clust = nib.load(gfeat_clust).get_data()

    # Loop over masks
    for mask in mask_list:
        mask_name = os.path.basename(mask)[:-7]
        print "Processing %s" % mask_name

        # Get mask index, index stat-file and clusterfile
        mask_idx = nib.load(mask).get_data() > thres
        masked = stat[mask_idx]
        clust_IDs = np.unique(clust[mask_idx])

        # Ignore 0-index of clusters (is default)
        if len(clust_IDs) > 1 and clust_IDs[0] == 0:
            clust_IDs = np.delete(clust_IDs, 0)

        # Loop over cluster-IDs
        for i in clust_IDs:

            # If 0, fill in zeros
            analyze = np.sum(masked>0)

            if analyze > 0:
                mx = np.max(masked[clust[mask_idx]==i])

                # Fill in 1 in tmp where the mask == max and get MNI-coordinates
                tmp = np.zeros(stat.shape)
                tmp[mask_idx] = stat[mask_idx] == mx
                X = np.where(tmp == 1)[0]
                Y = np.where(tmp == 1)[1]
                Z = np.where(tmp == 1)[2]
                clustersize = np.sum(masked[clust[mask_idx]==i])
            else:
                mx = 0
                X = 0
                Y = 0
                Z = 0
                clustersize = 0

            f.write('%s \t %i \t %i \t %i \t %i \t %f \n' % \
                    (mask_name, clustersize, X, Y, Z, np.round(mx, 3)))

    f.close()

    # Some pandas magic to sort
    df = pd.read_csv(filename, sep='\t', header=False)
    df = df.sort('k', ascending=False)
    print df

    with open(filename, 'w') as f:
        df.to_csv(f, header=True, sep='\t', index=False)


if __name__ == '__main__':

    # SOME TEST FILES
    gfeat = '/home/c6386806/Desktop/DECODING_EMOTIONS/Univariate_def/FLAME_FE_FSL_FirstLevel_Zinnen/cope1.gfeat/cope1.feat/thresh_zstat1.nii.gz'
    gfeat_clust = opj(os.path.dirname(gfeat), 'cluster_mask_zstat1.nii.gz')
    mask_list = glob.glob(opj('/home/c6386806/ROIs/Harvard_Oxford_atlas/unilateral', '*nii.gz*'))
    thres = 30 # this actually matters A LOT

    main(gfeat, gfeat_clust, mask_list, thres)
