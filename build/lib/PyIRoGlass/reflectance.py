# %% 
__author__ = 'Sarah Shi, Henry Towbin'

import os
import copy
import numpy as np
import pandas as pd
import mc3
import warnings

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
from matplotlib import pyplot as plt
mpl.use('pgf')
import mc3.utils as mu
import mc3.stats as ms

from pykrige import OrdinaryKriging
import scipy.signal as signal
from scipy.linalg import solveh_banded
import scipy.interpolate as interpolate

from ._version import __version__

from PyIRoGlass.dataload import *
from PyIRoGlass.core import *

# %% Reflectance FTIR - Interference Fringe Processing for Thicknesses

def PeakID(ref_spec, wn_high, wn_low, peak_heigh_min_delta, peak_search_width, savgol_filter_width, smoothing_wn_width = None, remove_baseline = False, plotting = False, filename = None):
    
    """
    Identifies peaks based on the peakdetect package which identifies local 
    maxima and minima in noisy signals. Based on: https://github.com/avhn/peakdetect
    
    Parameters:
        ref_spec (pd.DataFrame): A Pandas DataFrame indexed by wavenumber and containing absorbance values.
        wn_high (int): The upper wavenumber limit for the analysis.
        wn_low (int): The lower wavenumber limit for the analysis.
        smoothing_wn_width (int): The window size for the Savitzky-Golay smoothing filter. Default is None.
        remove_baseline (bool): Whether to remove the baseline from the spectrum. Default is False.
        peak_heigh_min_delta (float): Minimum difference between a peak and its neighboring points for it to be
            considered a peak. Default is 0.008.
        peak_search_width (int): The size of the region around each point to search for a peak. Default is 50.
        plotting (bool): Whether to create a plot of the spectrum with identified peaks and troughs. Default is False.
        filename (str): The name of the plot file. If None, the plot is not saved. Default is None.
    
    Returns:
        Tuple containing the following elements:
            Peaks and troughs identified as local maxima and minima, respectively.
    """

    from peakdetect import peakdetect


    spec = ref_spec[wn_low:wn_high].copy() # dataframe indexed by wavenumber
    spec_filt = pd.DataFrame(columns = ['Wavenumber', 'Absorbance']) 
    baseline = 0

    spec_filter = signal.medfilt(spec.Absorbance, 3)

    if remove_baseline == True:
        baseline = signal.savgol_filter(spec_filter, savgol_filter_width, 3)
        spec_filter = spec_filter - baseline

    if smoothing_wn_width is not None:
        spec_filter = signal.savgol_filter(spec_filter, smoothing_wn_width, 3)

    spec_filt['Absorbance'] = spec_filter
    spec_filt.index = spec.index
    spec['Subtracted'] = spec['Absorbance'] - baseline

    pandt = peakdetect(spec_filt.Absorbance, spec_filt.index, lookahead = peak_search_width, delta = peak_heigh_min_delta)
    peaks = np.array(pandt[0])
    troughs = np.array(pandt[1])

    if plotting == False: 
        pass
    else: 
        fig, ax = plt.subplots(1, 1, figsize = (8, 6))
        ax.plot(spec.index, spec['Subtracted'], linewidth = 1)
        ax.plot(spec_filt.index, spec_filt.Absorbance)
        ax.plot(peaks[:,0], peaks[:,1], 'ro')
        ax.plot(troughs[:,0], troughs[:,1], 'ko')
        ax.set_title(filename)
        ax.set_xlabel('Wavenumber')
        ax.set_ylabel('Absorbance')
        ax.invert_xaxis()

    return peaks, troughs

def Thickness_Calc(n, positions):

    """
    Calculates thicknesses of glass wafers based on the refractive index of the 
    glass and the positions of the peaks or troughs in the FTIR spectrum.
    
    Parameters:
        n (float): Refractive index of the glass.
        positions (np.ndarray): Array of positions of the peaks or troughs in the FTIR spectrum.
    
    Returns:
        np.ndarray: Array of thicknesses of glass wafers.
    """

    return 1/(2 * n * np.abs(np.diff(positions)))

def Thickness_Processing(dfs_dict, n, wn_high, wn_low, remove_baseline=False, plotting=False, phaseol=True):

    """
    Calculates thickness of glass wafers based on the refractive index of the glass and the positions of the
    peaks or troughs in the FTIR spectrum. Thicknesses for each interference fringe, starting at both the peaks
    and troughs of the fringes are determined. These thicknesses are then averaged over the interval of interest.
    Parameters: 
        dfs_dict (dictionary): dictionary containing FTIR data for each file
        n (float): refractive index of the glass
        wn_high (float): the high wavenumber cutoff for the analysis
        wn_low (float): the low wavenumber cutoff for the analysis
        remove_baseline (bool): whether or not to remove the baseline from the data
        plotting (bool): whether or not to plot the data and detected peaks and troughs
    
    Returns:
        ThickDF (pd.DataFrame): a dataframe containing the thickness calculations for each file

    Notes: 
        smoothing_wn_width (float): width of the Savitzky-Golay smoothing window, if not used, set to None
        peak_heigh_min_delta (float): minimum height difference between a peak and its surrounding points
        peak_search_width (float): the distance (in wavenumbers) to look on either side of a peak to find the
        corresponding trough
    """

    ThickDF = pd.DataFrame(columns=['Peak_Thicknesses', 'Peak_Thickness_M', 'Peak_Thickness_STD',
                                    'Trough_Thicknesses', 'Trough_Thickness_M', 'Trough_Thickness_STD',
                                    'Thickness_M', 'Thickness_STD'])

    failures = []

    # If phase is olivine, set these parameters. If phase glass, set other parameters. 
    if phaseol == True: 
        savgol_filter_width = 99
        smoothing_wn_width = 15 
        peak_heigh_min_delta = 0.002
        peak_search_width = 10
    elif phaseol == False: 
        savgol_filter_width = 449
        smoothing_wn_width = 71
        peak_heigh_min_delta = 0.008
        peak_search_width = 50


    for filename, data in dfs_dict.items(): 
        try:
            peaks, troughs = PeakID(data, wn_high, wn_low,  filename=filename, plotting=plotting, savgol_filter_width=savgol_filter_width, smoothing_wn_width = smoothing_wn_width, remove_baseline = True, peak_heigh_min_delta = peak_heigh_min_delta, peak_search_width = peak_search_width)
            peaks_loc = peaks[:, 0].round(2)
            troughs_loc = troughs[:, 0].round(2)
            peaks_diff = np.diff(peaks[:, 0]).round(2)
            troughs_diff = np.diff(troughs[:, 0]).round(2)

            peaks_loc_filt = np.array([x for x in peaks_loc if abs(x - np.mean(peaks_loc)) < 2 * np.std(peaks_loc)])
            troughs_loc_filt = np.array([x for x in troughs_loc if abs(x - np.mean(troughs_loc)) < 2 * np.std(troughs_loc)])
            peaks_diff_filt = np.array([x for x in peaks_diff if abs(x - np.mean(peaks_diff)) < 2 * np.std(peaks_diff)])
            troughs_diff_filt = np.array([x for x in troughs_diff if abs(x - np.mean(troughs_diff)) < 2 * np.std(troughs_diff)])

            t_peaks = (Thickness_Calc(n, peaks[:,0]) * 1e4).round(2)
            mean_t_peaks = np.mean(t_peaks)
            std_t_peaks = np.std(t_peaks)
            t_peaks_filt = np.array([x for x in t_peaks if abs(x - np.mean(t_peaks)) < np.std(t_peaks)])
            mean_t_peaks_filt = np.mean(t_peaks_filt).round(2)
            std_t_peaks_filt = np.std(t_peaks_filt).round(2)

            t_troughs = (Thickness_Calc(n, troughs[:,0]) * 1e4).round(2)
            mean_t_troughs = np.mean(t_troughs)
            std_t_troughs = np.std(t_troughs)
            t_troughs_filt = [x for x in t_troughs if abs(x - np.mean(t_troughs)) < np.std(t_troughs)]
            mean_t_troughs_filt = np.mean(t_troughs_filt).round(2)
            std_t_troughs_filt = np.std(t_troughs_filt).round(2)

            mean_t = np.mean(np.concatenate([t_peaks_filt, t_troughs_filt])).round(2)
            std_t = np.std(np.concatenate([t_peaks_filt, t_troughs_filt])).round(2)

            ThickDF.loc[f"{filename}"] = pd.Series({'Peak_Thicknesses': t_peaks_filt, 'Peak_Thickness_M': mean_t_peaks_filt, 'Peak_Thickness_STD': std_t_peaks_filt, 'Peak_Loc': peaks_loc_filt, 'Peak_Diff': peaks_diff_filt, 'Trough_Thicknesses': t_troughs_filt, 'Trough_Thickness_M': mean_t_troughs_filt, 'Trough_Thickness_STD': std_t_troughs_filt, 'Trough_Loc': troughs_loc_filt, 'Trough_Diff': troughs_diff_filt, 'Thickness_M': mean_t, 'Thickness_STD': std_t})

        except Exception as e:
            print(f"Error: {e}")
            print(e)
            failures.append(filename)
            ThickDF.loc[filename] = pd.Series({'V1':np.nan,'V2':np.nan,'Thickness':np.nan})


    return ThickDF

def Reflectance_Index(XFo):

    """
    Calculates the reflectance index for a given forsterite composition.
    The reflectance index is calculated based on values from Deer, Howie, and Zussman, 3rd Edition.

    Parameters:
        XFo (float): The mole fraction of forsterite in the sample.

    Returns:
        n (float): The calculated reflectance index.
    """

    n_alpha = 1.827 - 0.192*XFo
    n_beta = 1.869 - 0.218*XFo
    n_gamma = 1.879 - 0.209*XFo
    n = (n_alpha+n_beta+n_gamma) / 3

    return n
