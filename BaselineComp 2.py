# %% -*- coding: utf-8 -*-
""" Created on June 12, 2021 // @author: Sarah Shi """

# Import packages

import os
import sys
import glob
import numpy as np
import pandas as pd
import mc3

sys.path.append('src/')
import PyIRoGlass as pig

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc, cm
import seaborn as sns

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 16})
plt.rcParams['pdf.fonttype'] = 42


# %% 

path_beg = os.getcwd() + '/'

standards_path = path_beg + 'Inputs/TransmissionSpectra/Standards/'
standards_FILES = sorted(glob.glob(standards_path + "*.csv"))

hj_path = path_beg + 'HJ_fitted_baseline/'
FILES = sorted(glob.glob(hj_path + "*"))
DFS_FILES, DFS_DICT = pig.Load_SampleCSV(FILES, wn_high = 2400, wn_low = 1250)

Wavenumber = pig.Load_Wavenumber('BaselineAvgPC.npz')
PCmatrix = pig.Load_PC('BaselineAvgPC.npz')
npz_path = path_beg + 'NPZTXTFILES/STD/'
npz_FILES = sorted(glob.glob(npz_path + "*.npz"))


# Iterate over the files in the HJ_fitted_baseline folder
for hj_file_path, data in DFS_DICT.items():
    # Extract the common part of the filename (e.g., samplename_fitted_baseline)
    hj_filename = os.path.basename(hj_file_path)
    common_part = hj_filename.split('_fitted_baseline')[0]

    fig, ax = plt.subplots(1, 1, figsize = (8, 8))
    ax.set_title(common_part)
    ax.set_xlabel('Wavenumber $(\mathregular{cm^{-1}})$')
    ax.set_ylabel('Absorbance')

    column_names = ["Wavenumber", "Absorbance"]
    csv_data = pd.read_csv(standards_path+common_part+'.csv', names=column_names)
    csv_data = csv_data.set_index('Wavenumber')
    ax.plot(csv_data[1250:2400].index, csv_data[1250:2400].Absorbance, label='FTIR Spectrum')

    hj_data = pd.read_csv(hj_path+common_part+'_fitted_baseline.csv', names=column_names)
    hj_data = hj_data.set_index('Wavenumber')
    ax.plot(hj_data[1250:2400].index, hj_data[1250:2400].Absorbance, label='Devolatilized Baseline')

    mc3_output = np.load(npz_path+common_part+'.npz')
    Nvectors=5
    PC_BP = mc3_output['bestp'][0:Nvectors]
    PC_STD = mc3_output['stdp'][0:Nvectors]
    Baseline_Solve_BP = PC_BP * PCmatrix.T
    Baseline_Solve_BP = np.asarray(Baseline_Solve_BP).ravel()

    m_BP, b_BP = mc3_output['bestp'][-2:None]
    m_STD, b_STD = mc3_output['stdp'][-2:None]
    Line_BP = pig.Linear(Wavenumber, m_BP, b_BP) 
    Baseline_Solve_BP = Baseline_Solve_BP + Line_BP

    ax.plot(Wavenumber, Baseline_Solve_BP, label='PyIRoGlass Baseline')
    ax.legend(loc = 'upper left', labelspacing = 0.5, handletextpad = 0.25, handlelength = 1.00, prop={'size': 14}, frameon=False)
    ax.set_xlim([2400, 1250])
    # ax.set_ylim([-0.25, 2.5])
    ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
    ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
    plt.savefig('BLComp/'+common_part+'_BLcomp.png', bbox_inches='tight', pad_inches = 0.025)


# %%
