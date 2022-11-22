# %% -*- coding: utf-8 -*-

""" Created on July 29, 2022 // @author: Sarah Shi and Henry Towbin"""

import numpy as np
import pandas as pd 
import scipy.signal as signal
from peakdetect import peakdetect

import os
import glob 
from pathlib import Path

from matplotlib import pyplot as plt
from matplotlib import rc, cm
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 18})
plt.rcParams['pdf.fonttype'] = 42

# %% 

def Load_SampleCSV(paths, wn_high, wn_low): 

    """The Load_SampleCSV function takes the inputs of the path to a directory with 
    all sample CSVs, wavenumber high, wavenumber low values. The function outputs a 
    dictionary of each sample's associated wavenumbers and absorbances."""

    dfs = []
    files = []

    for path in paths:
        head_tail = os.path.split(path)
        file = head_tail[1][0:-4]

        df = pd.read_csv(path, names= ['Wavenumber', 'Absorbance'])
        df.set_index('Wavenumber', inplace = True)
        spec = df.loc[wn_low:wn_high]
        dfs.append(spec)
        files.append(file)

    df_zipobj = zip(files, dfs)
    dfs_dict = dict(df_zipobj)

    return files, dfs_dict


def PeakID(ref_spec, wn_high, wn_low, smoothing_wn_width = None, remove_baseline = False, 
    peak_heigh_min_delta = 0.008, peak_search_width = 50, plotting = False, filename = None):
    
    """Identifies peaks based on the peakdetect package which identifies local 
    maxima and minima in noisy signals.
    Based on: https://github.com/avhn/peakdetect"""

    spec = ref_spec[wn_low:wn_high] # dataframe indexed by wavenumber
    spec_filt = pd.DataFrame(columns = ['Wavenumber', 'Absorbance']) 
    baseline = 0
 
    spec_filter = signal.medfilt(spec.Absorbance, 3)
    
    if remove_baseline == True:
        baseline = signal.savgol_filter(spec_filter, 450, 3)
        spec_filter = spec_filter - baseline

    if smoothing_wn_width is not None:
        spec_filter = signal.savgol_filter(spec_filter, smoothing_wn_width, 3)


    spec_filt['Absorbance'] = spec_filter
    spec_filt.index = spec.index

    pandt = peakdetect(spec_filt.Absorbance, spec_filt.index, lookahead=peak_search_width, delta = peak_heigh_min_delta)
    peaks = np.array(pandt[0])
    troughs = np.array(pandt[1])

    if plotting == True: 
        fig, ax = plt.subplots(1, 1, figsize = (8, 6))
        ax.plot(spec.index, spec.Absorbance- baseline, linewidth = 1)
        ax.plot(spec_filt.index, spec_filt.Absorbance)
        ax.plot(peaks[:,0], peaks[:,1], 'ro')
        ax.plot(troughs[:,0], troughs[:,1], 'ko')
        ax.set_title(filename)
        ax.set_xlabel('Wavenumber')
        ax.set_ylabel('Absorbance')
        ax.invert_xaxis()

    return peaks, troughs


def ThicknessCalc(n, positions):

    """Calculates thicknesses of glass wafers based on the refractive index of the 
    glass and the positions of the peaks or troughs in the FTIR spectrum."""

    return 1/(2 * n * np.abs(np.diff(positions)))


def ThicknessProcessing(dfs_dict, n, wn_high, wn_low, smoothing_wn_width = None, 
peak_heigh_min_delta = 0.008, peak_search_width = 50, remove_baseline = False, plotting=False):

    """Calculates thickness of glass wafers based on the refractive index of the glass and the positions 
    of the peaks or troughs in the FTIR spectrum. Thicknesses for each interference fringe, starting at 
    both the peaks and troughs of the fringes are determined. These thicknesses are then averaged over the 
    interval of interest."""

    ThickDF = pd.DataFrame(columns=['Peak_Thicknesses', 'Peak_Thickness_M', 'Peak_Thickness_STD', 
                                    'Trough_Thicknesses', 'Trough_Thickness_M', 'Trough_Thickness_STD', 
                                    'Thickness_M', 'Thickness_STD'])
    failures = []

    for filename, data in dfs_dict.items(): 
        try:
            peaks, troughs = PeakID(data, wn_high, wn_low,  filename=filename, plotting=plotting, 
                smoothing_wn_width = smoothing_wn_width, remove_baseline = remove_baseline, 
                peak_heigh_min_delta = peak_heigh_min_delta, peak_search_width = peak_search_width)
            
            t_peaks = (ThicknessCalc(n, peaks[:,0]) * 1e4).round(2)
            mean_t_peaks = np.mean(t_peaks).round(2)
            std_t_peaks = np.std(t_peaks).round(2)

            t_troughs = (ThicknessCalc(n, troughs[:,0]) * 1e4).round(2)
            mean_t_troughs = np.mean(t_troughs).round(2)
            std_t_troughs = np.std(t_troughs).round(2)

            mean_t = np.mean(np.concatenate([t_peaks, t_troughs])).round(2)
            std_t = np.std(np.concatenate([t_peaks, t_troughs])).round(2)

            ThickDF.loc[f"{filename}"] = pd.Series({'Peak_Thicknesses': t_peaks, 'Peak_Thickness_M': mean_t_peaks, 'Peak_Thickness_STD': std_t_peaks, 
            'Trough_Thicknesses': t_troughs, 'Trough_Thickness_M': mean_t_troughs, 'Trough_Thickness_STD': std_t_troughs, 
            'Thickness_M': mean_t, 'Thickness_STD': std_t})
            
    
        except Exception as e:
            print(f"Error: {e}")
            print(e)
            failures.append(filename)
            ThickDF.loc[filename] = pd.Series({'V1':np.nan,'V2':np.nan,'Thickness':np.nan})

    return ThickDF


def ReflectanceIndex(XFo):
    
    """Calculates reflectance index for given forsterite composition. 
    Values based on those from Deer, Howie, and Zussman, 3rd Edition.
    Input forsterite in mole fraction."""

    n_alpha = 1.827 - 0.192*XFo
    n_beta = 1.869 - 0.218*XFo
    n_gamma = 1.879 - 0.209*XFo
    n = (n_alpha+n_beta+n_gamma) / 3

    return n

# %% 

path_parent = os.path.dirname(os.getcwd())
path_input = os.getcwd() + '/Inputs'

# Change paths to direct to folder with SampleSpectra -- last bit should be whatever your folder with spectra is called. 
PATH = path_input + '/ReflectanceSpectra/rf_ND70/'
FILES = glob.glob(PATH + "*")
FILES.sort()

DFS_FILES, DFS_DICT = Load_SampleCSV(FILES, wn_high = 2850, wn_low = 1700)

# %% BASALTIC GLASS

# n=1.546 in the range of 2000-2700 cm^-1 following Nichols and Wysoczanski, 2007 for basaltic glass
# Provide a wavenumber buffer in these applications. Assess if the range is appropriate by looking at
# the standard deviations associated with each thickness. 

smoothing_wn_width = 71
peak_heigh_min_delta = 0.008
peak_search_width = 50

ThickDF = ThicknessProcessing(DFS_DICT, n = 1.546, wn_high = 2850, wn_low = 1700, smoothing_wn_width = smoothing_wn_width, 
    peak_heigh_min_delta= peak_heigh_min_delta, peak_search_width = peak_search_width, 
    remove_baseline = True, plotting = True)

ThickDF

# %% 

# ThickDF.to_csv('ND70_Thickness_HighQuality.csv')

# %%


path_parent = os.path.dirname(os.getcwd())
path_input = os.getcwd() + '/Inputs'

# Change paths to direct to folder with SampleSpectra -- last bit should be whatever your folder with spectra is called. 
PATH = path_input + '/ReflectanceSpectra/rf_ND70_lowq/'
FILES = glob.glob(PATH + "*")
FILES.sort()

DFS_FILES, DFS_DICT = Load_SampleCSV(FILES, wn_high = 2850, wn_low = 1700)

# %% BASALTIC GLASS

# n=1.546 in the range of 2000-2700 cm^-1 following Nichols and Wysoczanski, 2007 for basaltic glass
# Provide a wavenumber buffer in these applications. Assess if the range is appropriate by looking at
# the standard deviations associated with each thickness. 

smoothing_wn_width = 71
peak_heigh_min_delta = 0.008
peak_search_width = 50

ThickDF = ThicknessProcessing(DFS_DICT, n = 1.546, wn_high = 2300, wn_low = 1950, smoothing_wn_width = smoothing_wn_width, 
    peak_heigh_min_delta= peak_heigh_min_delta, peak_search_width = peak_search_width, 
    remove_baseline = True, plotting = True)

ThickDF

# %% 

# ThickDF.to_csv('ND70_Thickness_LowQuality.csv')

# %% OLIVINE

# n=XFo dependent in the range of 2100-2700 cm^-1 following Nichols and Wysoczanski, 2007 for basaltic glass

n = ReflectanceIndex(0.70)
n

# Thickness otherwise calculated in the same manner, with a slightly narrower range. 
# Provide a 100 cm^-1 buffer in these applications. 

# %%
