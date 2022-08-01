# %% -*- coding: utf-8 -*-
""" Created on July 29, 2022 // @author: Sarah Shi """

import os, glob
import numpy as np
import pandas as pd 
import scipy.signal as signal
from peakdetect import peakdetect

from matplotlib import pyplot as plt
from matplotlib import rc, cm
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 18})
plt.rcParams['pdf.fonttype'] = 42

# %% 

def Load_SampleCSV(paths, wn_high, wn_low): 

    """The Load_SampleCSV function takes the inputs of the path to a directory with all sample CSVs, 
    wavenumber high, wavenumber low values. The function outputs a dictionary of each sample's associated 
    wavenumbers and absorbances."""

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


def PeakID(ref_spec, wn_high, wn_low, plotting, file = None):
    
    """Identifies peaks based on the peakdetect package which identifies local 
    maxima and minima in noisy signals."""
    # https://github.com/avhn/peakdetect

    spec = ref_spec[wn_low:wn_high]
    spec_filt = pd.DataFrame(columns = ['Wavenumber', 'Absorbance']) 
    spec_filt['Absorbance'] = signal.medfilt(spec.Absorbance, 21)
    spec_filt.index = spec.index

    pandt = peakdetect(spec_filt.Absorbance, spec_filt.index, lookahead=31, delta = 0.000001)
    peaks = np.array(pandt[0])
    troughs = np.array(pandt[1])

    if plotting == True: 
        plt.figure(figsize = (8, 6))
        plt.plot(spec.index, spec.Absorbance)
        plt.plot(spec_filt.index, spec_filt.Absorbance)
        plt.plot(peaks[:,0], peaks[:,1], 'ro')
        plt.plot(troughs[:,0], troughs[:,1], 'ko')
        plt.title(file)
        plt.xlabel('Wavenumber')
        plt.ylabel('Absorbance')
    else: 
        pass

    return peaks, troughs


def ThicknessCalc(m, n, v1, v2): 
    """ Calculates thicknesses of glass wafers
    """
    # t = thickness, m = number of waves, n = refractive index 
    # v1 = highest wavenumber, v2 = lowest wavenumber
    t = m / (2 * n * np.abs(v1-v2))
    return t


def Thickness_Processing(dfs_dict, m, n, wn_high, wn_low, plotting): 

    ThickDF = pd.DataFrame(columns=['V1', 'V2', 'Thickness'])
    failures = []

    for files, data in dfs_dict.items(): 
        try: 
            peaks, troughs = PeakID(data, wn_high, wn_low, plotting, files)
            t = ThicknessCalc(m, n, peaks[0, 0], peaks[m+1, 0]) * 1e4
            ThickDF.loc[files] = pd.Series({'V1':peaks[0, 0],'V2':peaks[m+1, 0],'Thickness':t})
        except:
            failures.append(files)
            pass

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
PATH = path_input + '/RefSpectra/rf_ND70/'
FILES = glob.glob(PATH + "*")
FILES.sort()

DFS_FILES, DFS_DICT = Load_SampleCSV(FILES, wn_high = 2800, wn_low = 1900)

# %% BASALTIC GLASS

# n=1.546 in the range of 2000-2700 cm^-1 following Nichols and Wysoczanski, 2007 for basaltic glass
# Provide a 100 cm^-1 buffer in these applications. 

ThickDF_2 = Thickness_Processing(DFS_DICT, m = 2, n = 1.546, wn_high = 2800, wn_low = 1900, plotting = True)

ThickDF_3 = Thickness_Processing(DFS_DICT, m = 3, n = 1.546, wn_high = 2800, wn_low = 1900, plotting = False)

ThickDF_4 = Thickness_Processing(DFS_DICT, m = 4, n = 1.546, wn_high = 2800, wn_low = 1900, plotting = False)

# %% 

thicknesscomp = pd.DataFrame(columns = ['Thickness_2', 'Thickness_3', 'Thickness_4'])

thicknesscomp['Thickness_2'] = ThickDF_2['Thickness']
thicknesscomp['Thickness_3'] = ThickDF_3['Thickness']
thicknesscomp['Thickness_4'] = ThickDF_4['Thickness']
thicknesscomp 

# %% OLIVINE

# n=XFo dependent in the range of 2100-2700 cm^-1 following Nichols and Wysoczanski, 2007 for basaltic glass

n = ReflectanceIndex(0.70)
n

# Thickness otherwise calculated in the same manner, with a slightly narrower range. 
# Provide a 100 cm^-1 buffer in these applications. 

# %%
