#
# This file is implemented based on the author code of
#    Lee et al., "A simple unified framework for detecting out-of-distribution samples and adversarial attacks", in NeurIPS 2018.
#

import os
import torch
import numpy as np
import torch.nn.functional as F

def compute_discrepancy(model, test_loader, outdir, id_flag):
    total = 0
    if id_flag == True:
        outfile = os.path.join(outdir, 'discrepancy_id.txt')
    else:
        outfile = os.path.join(outdir, 'discrepancy_pert.txt')

    f = open(outfile, 'w')
    
    for data, _ in test_loader:
        dists1, _ = model(data.cuda())
        discrepancy, _ = torch.min(dists1, dim=1)
        total += data.size(0)

        for i in range(data.size(0)):
            f.write("{}\n".format(-discrepancy[i]))
    
    f.close()


def get_auroc_curve(indir):
    known = np.loadtxt(os.path.join(indir, 'discrepancy_id.txt'), delimiter='\n')
    novel = np.loadtxt(os.path.join(indir, 'discrepancy_pert.txt'), delimiter='\n')
    known.sort()
    novel.sort()
    
    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known),np.min(novel)])
    
    num_k = known.shape[0]
    num_n = novel.shape[0]
    
    tp = -np.ones([num_k+num_n+1], dtype=int)
    fp = -np.ones([num_k+num_n+1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k+num_n):
        if k == num_k:
            tp[l+1:] = tp[l]
            fp[l+1:] = np.arange(fp[l]-1, -1, -1)
            break
        elif n == num_n:
            tp[l+1:] = np.arange(tp[l]-1, -1, -1)
            fp[l+1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l+1] = tp[l]
                fp[l+1] = fp[l] - 1
            else:
                k += 1
                tp[l+1] = tp[l] - 1
                fp[l+1] = fp[l]
    tpr85_pos = np.abs(tp / num_k - .85).argmin()
    tpr95_pos = np.abs(tp / num_k - .95).argmin()
    tnr_at_tpr85 = 1. - fp[tpr85_pos] / num_n
    tnr_at_tpr95 = 1. - fp[tpr95_pos] / num_n
    return tp, fp, tnr_at_tpr85, tnr_at_tpr95

def compute_metrics(dir_name, verbose=False):
    tp, fp, tnr_at_tpr85, tnr_at_tpr95 = get_auroc_curve(dir_name)
    results = dict()
    mtypes = ['TNR85', 'TNR95', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    if verbose:
        print('      ', end='')
        for mtype in mtypes:
            print(' {mtype:6s}'.format(mtype=mtype), end='')
        print('')
        
    if verbose:
        print('{stype:5s} '.format(stype=stype), end='')
    results = dict()
    
    # TNR85
    mtype = 'TNR85'
    results[mtype] = tnr_at_tpr85
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')

    # TNR95
    mtype = 'TNR95'
    results[mtype] = tnr_at_tpr95
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp/tp[0], [0.]])
    fpr = np.concatenate([[1.], fp/fp[0], [0.]])
    results[mtype] = -np.trapz(1. - fpr, tpr)
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # DTACC
    mtype = 'DTACC'
    results[mtype] = .5 * (tp/tp[0] + 1. - fp/fp[0]).max()
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # AUIN
    mtype = 'AUIN'
    denom = tp + fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp/denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
    
    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0] - fp)/denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])
    if verbose:
        print(' {val:6.3f}'.format(val=100.*results[mtype]), end='')
        print('')

    return results

def print_perturbation_results(pert_result):

    for mtype in ['TNR85', 'TNR95', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']:
        print(' {mtype:6s}'.format(mtype=mtype), end='')
    print('\n{val:6.2f}'.format(val=100.*pert_result['TNR85']), end='')
    print(' {val:6.2f}'.format(val=100.*pert_result['TNR95']), end='')
    print(' {val:6.2f}'.format(val=100.*pert_result['AUROC']), end='')
    print(' {val:6.2f}'.format(val=100.*pert_result['DTACC']), end='')
    print(' {val:6.2f}'.format(val=100.*pert_result['AUIN']), end='')
    print(' {val:6.2f}\n'.format(val=100.*pert_result['AUOUT']), end='')