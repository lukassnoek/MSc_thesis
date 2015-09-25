"""
Extracts info from cluster-corrected files
"""

import os
from os.path import join as opj
import nibabel as nib
import numpy as np
import glob
import pandas as pd


def main(gfeat, gfeat_clust, mask_list, thres, per_cluster):
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

    print "Processing %s ..." % (gfeat)

    if per_cluster:
        filename = opj(os.path.dirname(gfeat), 'roi_info_%s.csv') % os.path.basename(gfeat)[:-7]
        f = open(filename, 'w')
        f.write('Cluster nr.\tk (clust)\tMax (clust)\tMax(X)\tMAX(Y)\tMAX(Z)\tROIs\tk (ROI)\tMax (ROI)\n')

        stat = nib.load(gfeat).get_data()
        clust = nib.load(gfeat_clust).get_data()
        clust_IDs = np.unique(clust)

        # Ignore 0-index of clusters (is default)
        if len(clust_IDs) > 1 and clust_IDs[0] == 0:
            clust_IDs = np.delete(clust_IDs, 0)

        clust_IDs = clust_IDs[::-1]

        for i, ID in enumerate(clust_IDs):

            if len(clust_IDs) == 1 and clust_IDs[0] == 0:
                f.write('%i\t%i\t%f\t%i\t%i\t%i\t' % (i+1, 0, 0, 0, 0, 0))
                break

            k = np.sum(clust==ID)
            mx = np.max(stat[clust==ID])

            tmp = np.zeros(stat.shape)
            tmp[clust==ID] = stat[clust==ID] == mx

            X = np.where(tmp == 1)[0]
            Y = np.where(tmp == 1)[1]
            Z = np.where(tmp == 1)[2]

            f.write('%i\t%i\t%f\t%i\t%i\t%i\t' % (i+1, k, mx, X, Y, Z))

            k_roi = np.zeros(len(mask_list))
            mx_roi = np.zeros(len(mask_list))
            mask_names = np.empty(len(mask_list), dtype=object)

            for j, mask in enumerate(mask_list):
                mask_name = os.path.basename(mask)[:-7]

                # Get mask index, index stat-file and clusterfile
                mask_idx = nib.load(mask).get_data() > thres
                clust_idx = clust==ID

                tmp = np.zeros(clust.shape)
                tmp[mask_idx] += 1
                tmp[clust_idx] += 1

                full_idx = tmp == 2
                k_roi[j] = np.sum(full_idx)

                if np.sum(full_idx) == 0:
                    mx_roi[j] = 0
                else:
                    mx_roi[j] = np.max(stat[full_idx])

                mask_names[j] = mask_name

            zipped = zip(mask_names, k_roi, mx_roi)

            zipped.sort(key=lambda tup: tup[1])
            zipped = zipped[::-1]
            summed_k = 0

            for roi, k_roi2, mx_roi2, in zipped:

                summed_k += k_roi2

                if k_roi2 > 20:
                    f.write('%s\t%i\t%f\n\t\t\t\t\t\t' % (roi, k_roi2, mx_roi2))
                else:
                    f.write('%s\t%i\t%s\n' % ('Unspecified', k-summed_k, 'n/a'))
                    break

        f.close()

    else:

        # Create file
        filename = opj(os.path.dirname(gfeat), 'roi_info.csv')
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
                analyze = np.sum(masked > 0)

                if analyze > 0:
                    mx = np.max(masked[clust[mask_idx] == i])

                    # Fill in 1 in tmp where the mask == max and get MNI-coordinates
                    tmp = np.zeros(stat.shape)
                    tmp[mask_idx] = stat[mask_idx] == mx
                    X = np.where(tmp == 1)[0]
                    Y = np.where(tmp == 1)[1]
                    Z = np.where(tmp == 1)[2]
                    clustersize = np.sum(masked[clust[mask_idx] == i])
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
    # gfeat = '/home/c6386806/Desktop/DecEmo_uni/Conjunction_analysis_p005/Action_min_baseline/thresh_conjunction.nii.gz'
    # gfeat_clust = opj(os.path.dirname(gfeat), 'cluster_mask_conjunction.nii.gz')
    mask_list = glob.glob(opj('/home/c6386806/ROIs/Harvard_Oxford_atlas/unilateral', '*nii.gz*'))
    thres = 20  # this actually matters A LOT
    per_clust = True
    # main(gfeat, gfeat_clust, mask_list, thres, per_clust)

    import glob

    gfeats = glob.glob('/home/c6386806/Desktop/DecEmo_uni/FSL_ThirdLevel_Zinnen_p005/*/cope1.feat/thresh_zstat*.nii.gz')
    gfeats_clust = glob.glob('/home/c6386806/Desktop/DecEmo_uni/FSL_ThirdLevel_Zinnen_p005/*/cope1.feat/cluster_mask_zstat*.nii.gz')

    for gfeat, gfeat_clust in zip(gfeats,gfeats_clust):
        main(gfeat, gfeat_clust, mask_list, thres, per_clust)

    gfeats = glob.glob('/home/c6386806/Desktop/DecEmo_uni/FSL_ThirdLevel_HWW_p005.gfeat/*/thresh_zstat*.nii.gz')
    gfeats_clust = glob.glob('/home/c6386806/Desktop/DecEmo_uni/FSL_ThirdLevel_HWW_p005.gfeat/*/cluster_mask_zstat*.nii.gz')

    for gfeat, gfeat_clust in zip(gfeats,gfeats_clust):
        main(gfeat, gfeat_clust, mask_list, thres, per_clust)

