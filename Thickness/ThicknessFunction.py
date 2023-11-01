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

import scipy 
from sklearn.metrics import mean_squared_error

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams["xtick.major.size"] = 4 # Sets length of ticks
plt.rcParams["ytick.major.size"] = 4 # Sets length of ticks
plt.rcParams["xtick.labelsize"] = 20 # Sets size of numbers on tick marks
plt.rcParams["ytick.labelsize"] = 20 # Sets size of numbers on tick marks
plt.rcParams["axes.titlesize"] = 22
plt.rcParams["axes.labelsize"] = 22 # Axes labels

# %% 


def Load_SampleCSV(paths, wn_high, wn_low): 

    """The Load_SampleCSV function takes the inputs of the path to a directory with all sample CSVs, wavenumber high, wavenumber low values. The function outputs a dictionary of each sample's associated wavenumbers and absorbances."""

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


def PeakID(ref_spec, wn_high, wn_low, peak_heigh_min_delta, peak_search_width, savgol_filter_width, smoothing_wn_width = None, remove_baseline = False, plotting = False, filename = None):

    """Identifies peaks based on the peakdetect package which identifies local maxima and minima in noisy signals.
    Based on: https://github.com/avhn/peakdetect"""

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

    if plotting == True: 
        fig, ax = plt.subplots(1, 1, figsize = (8, 6))
        ax.plot(spec.index, spec['Subtracted'], linewidth = 1)
        ax.plot(spec_filt.index, spec_filt.Absorbance)
        ax.plot(peaks[:,0], peaks[:,1], 'ro')
        ax.plot(troughs[:,0], troughs[:,1], 'ko')
        ax.set_title(filename)
        ax.set_xlabel('Wavenumber')
        ax.set_ylabel('Absorbance')
        ax.invert_xaxis()

    # spec.to_csv(filename + '_spec.csv')
    # spec_filt.to_csv(filename + '_specfilt.csv')
    # pd.DataFrame(peaks).to_csv(filename + '_peaks.csv')
    # pd.DataFrame(troughs).to_csv(filename + '_troughs.csv')

    return peaks, troughs


def ThicknessCalc(n, positions):

    """Calculates thicknesses of glass wafers based on the refractive index of the glass and the positions of the peaks or troughs in the FTIR spectrum."""

    return 1/(2 * n * np.abs(np.diff(positions)))


def ThicknessProcessing(dfs_dict, n, wn_high, wn_low, savgol_filter_width, smoothing_wn_width = None, 
    peak_heigh_min_delta = 0.008, peak_search_width = 50, remove_baseline = False, plotting=False):

    """Calculates thickness of glass wafers based on the refractive index of the glass and the positions of the peaks or troughs in the FTIR spectrum. Thicknesses for each interference fringe, starting at both the peaks and troughs of the fringes are determined. These thicknesses are then averaged over the interval of interest."""

    ThickDF = pd.DataFrame(columns=['Peak_Thicknesses', 'Peak_Thickness_M', 'Peak_Thickness_STD', 'Peak_Loc', 'Peak_Diff', 'Trough_Thicknesses', 'Trough_Thickness_M', 'Trough_Thickness_STD', 'Trough_Loc', 'Trough_Diff','Thickness_M', 'Thickness_STD'])
    failures = []

    for filename, data in dfs_dict.items(): 
        try:
            peaks, troughs = PeakID(data, wn_high, wn_low,  filename=filename, plotting=plotting, savgol_filter_width=savgol_filter_width, smoothing_wn_width = smoothing_wn_width, remove_baseline = True, peak_heigh_min_delta = peak_heigh_min_delta, peak_search_width = peak_search_width)
            peaks_loc = peaks[:, 0].round(2)
            troughs_loc = troughs[:, 0].round(2)
            peaks_diff = np.diff(peaks[:, 0]).round(2)
            troughs_diff = np.diff(troughs[:, 0]).round(2)

            peaks_loc_filt = np.array([x for x in peaks_loc if abs(x - np.mean(peaks_loc)) < np.std(peaks_loc)])
            troughs_loc_filt = np.array([x for x in troughs_loc if abs(x - np.mean(troughs_loc)) < np.std(troughs_loc)])
            peaks_diff_filt = np.array([x for x in peaks_diff if abs(x - np.mean(peaks_diff)) < np.std(peaks_diff)])
            troughs_diff_filt = np.array([x for x in troughs_diff if abs(x - np.mean(troughs_diff)) < np.std(troughs_diff)])

            t_peaks = (ThicknessCalc(n, peaks[:,0]) * 1e4).round(2)
            mean_t_peaks = np.mean(t_peaks)
            std_t_peaks = np.std(t_peaks)
            t_peaks_filt = np.array([x for x in t_peaks if abs(x - np.mean(t_peaks)) < np.std(t_peaks)])
            mean_t_peaks_filt = np.mean(t_peaks_filt).round(2)
            std_t_peaks_filt = np.std(t_peaks_filt).round(2)

            t_troughs = (ThicknessCalc(n, troughs[:,0]) * 1e4).round(2)
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


def ReflectanceIndex(XFo):
    
    """Calculates reflectance index for given forsterite composition. Values based on those from Deer, Howie, and Zussman, 3rd Edition. Input forsterite in mole fraction."""

    n_alpha = 1.827 - 0.192*XFo
    n_beta = 1.869 - 0.218*XFo
    n_gamma = 1.879 - 0.209*XFo
    n = (n_alpha+n_beta+n_gamma) / 3

    return n

# %% 


path_parent = os.path.dirname(os.getcwd())
path_input = path_parent + '/Inputs'

# Change paths to direct to folder with SampleSpectra -- last bit should be whatever your folder with spectra is called. 
PATH = path_input + '/ReflectanceSpectra/FuegoOl/'
FILES = glob.glob(PATH + "*")
FILES.sort()

DFS_FILES, DFS_DICT = Load_SampleCSV(FILES, wn_high = 2800, wn_low = 2000)

# n=1.546 in the range of 2000-2700 cm^-1 following Nichols and Wysoczanski, 2007 for basaltic glass
# Provide a wavenumber buffer in these applications. Assess if the range is appropriate by looking at
# the standard deviations associated with each thickness. 

smoothing_wn_width = 15 # lower smoothing 
peak_heigh_min_delta = 0.002
peak_search_width = 10

n_ol = ReflectanceIndex(0.72)

Fuego1 = ThicknessProcessing(DFS_DICT, n = n_ol, wn_high = 2700, wn_low = 2100, savgol_filter_width = 99, smoothing_wn_width = smoothing_wn_width, peak_heigh_min_delta = peak_heigh_min_delta, peak_search_width = 10, remove_baseline = True, plotting = True)

Fuego1.to_csv('FuegoOlThickness.csv')

# %% remove values outside 3 SD

Fuego1 

# %% 

def relative_root_mean_squared_error(true, pred):
    num = np.sum(np.square(true - pred))
    den = np.sum(np.square(pred))
    squared_error = num/den
    rrmse_loss = np.sqrt(squared_error)
    return rrmse_loss

def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Remove NaNs
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    df = df.dropna()
    y_true = df['y_true']
    y_pred = df['y_pred']
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1] 
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator

micro = pd.read_csv('FuegoOlMicrometer.csv')

slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(micro.Thickness_Micrometer.values, Fuego1.Thickness_M.astype(float))
ccc1 = concordance_correlation_coefficient(micro.Thickness_Micrometer.values, Fuego1.Thickness_M.astype(float))
rmse1 = mean_squared_error(micro.Thickness_Micrometer.values, Fuego1.Thickness_M.astype(float), squared=False)

range = [0, 90]

sz = 150
fig, ax = plt.subplots(1, 1, figsize = (7.5, 7.5))
ax.plot(range, range, 'k', lw = 1, zorder = 0)

ax.errorbar(micro.Thickness_Micrometer, Fuego1.Thickness_M, yerr = Fuego1.Thickness_STD, xerr = 3, ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.scatter(micro.Thickness_Micrometer, Fuego1.Thickness_M, s = sz, c = '#0C7BDC', edgecolors='black', linewidth = 0.5, zorder = 15)
ax.set_xlim([20, 90])
ax.set_ylim([20, 90])

ax.annotate("$\mathregular{R^{2}}$="+str(np.round(r_value1**2, 2)), xy=(0.02, 0.8775), xycoords="axes fraction", fontsize=16)
ax.annotate("CCC="+str(np.round(ccc1, 2)), xy=(0.02, 0.96), xycoords="axes fraction", fontsize=16)
ax.annotate("RMSE="+str(np.round(rmse1, 2))+"; RRMSE="+str(np.round(relative_root_mean_squared_error(micro.Thickness_Micrometer.values, Fuego1.Thickness_M.values)*100, 2))+'%', xy=(0.02, 0.92), xycoords="axes fraction", fontsize=16)
ax.annotate("m="+str(np.round(slope1, 2)), xy=(0.02, 0.84), xycoords="axes fraction", fontsize=16)
ax.annotate("b="+str(np.round(intercept1, 2)), xy=(0.02, 0.80), xycoords="axes fraction", fontsize=16)

ax.set_xlabel('Micrometer Thickness (µm)')
ax.set_ylabel('Reflectance FTIR Thickness (µm)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('OlThicknessTest.pdf',bbox_inches='tight', pad_inches = 0.025)


# %% 

path_parent = os.path.dirname(os.getcwd())
path_input = path_parent + '/Inputs'

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

HighThickDF = ThicknessProcessing(DFS_DICT, n = 1.546, wn_high = 2850, wn_low = 1700, savgol_filter_width = 449, smoothing_wn_width = smoothing_wn_width, peak_heigh_min_delta= peak_heigh_min_delta, peak_search_width = peak_search_width, remove_baseline = True, plotting = True)

HighThickDF

# %% 

HighThickDF.to_csv('ND70_Thickness_HighQuality.csv')

# %%


path_parent = os.path.dirname(os.getcwd())
path_input = path_parent + '/Inputs'

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

LowThickDF = ThicknessProcessing(DFS_DICT, n = 1.546, wn_high = 2300, wn_low = 1950, savgol_filter_width = 449, smoothing_wn_width = smoothing_wn_width, peak_heigh_min_delta= peak_heigh_min_delta, peak_search_width = peak_search_width, remove_baseline = True, plotting = True)

LowThickDF

# %% 

LowThickDF.to_csv('ND70_Thickness_LowQuality.csv')

# %% OLIVINE

# n=XFo dependent in the range of 2100-2700 cm^-1 following Nichols and Wysoczanski, 2007 for basaltic glass

n = ReflectanceIndex(0.72)
n

# Thickness otherwise calculated in the same manner, with a slightly narrower range. 
# Provide a 100 cm^-1 buffer in these applications. 


# %%
