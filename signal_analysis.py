# signal_analysis.py
# Author: John Parker
# Script for simple functions to help gather certain HR cupolet information.
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
import os
import time
from datetime import datetime, timedelta

def cross_correlation(x,y,normalize):
    '''
    Performs correlation between x and y, if normalize then correlation is between -1 and 1.
    Returns the x and y signals (normalized if applicable) the correlation coefficients, and
    the index of the largest correlation coefficient.
    '''
    if normalize:
        xnorm = (x - np.mean(x)) / (np.std(x) * len(x))
        ynorm = (y - np.mean(y)) /  np.std(y)
        corr = np.correlate(xnorm, ynorm,'full')
        return xnorm, ynorm, corr, int(np.where(corr == max(corr))[0])
    else:
        corr = np.correlate(x, y,'full')
        return x,y,corr, int(np.where(corr == max(corr))[0])


def get_period(signal,dt):
    '''
    Reads in a signal and time integration dt to recover the period. Assumes nonrandom
    data and find period based on max norm correlation coeff. Returns value of max corr
    coeff, the index of it, and the period based on the index * dt.
    '''
    xnorm, ynorm, corr, corr_max = cross_correlation(signal,signal,True)
    corrs = corr[int(len(corr)/2):]
    pks,_ = find_peaks(corrs)
    if len(pks > 0):
        mxpeak = np.max(corrs[pks])
        mxpeakind = np.where(corrs == mxpeak)[0][0]
        return mxpeak, mxpeakind, mxpeakind*dt
    else:
        return -1, -1, -1
