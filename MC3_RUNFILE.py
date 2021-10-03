# %% -*- coding: utf-8 -*-
""" Created on June 12, 2021 // @author: Sarah Shi and Henry Towbin """

# %% Import packages

import numpy as np
import pandas as pd
import mc3
import os
import glob
import warnings

import scipy.signal as signal
import scipy.interpolate as interpolate
import Automated_Baselines as Auto_B # Handles automated baseline functions
import MC3_BL_Plotting as mc3plots
import MC3_BACKEND as baselines

from pykrige import OrdinaryKriging
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import rc

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 14})
plt.rcParams['pdf.fonttype'] = 42

# %% Create directories for export file sorting. 
# Load FTIR Baseline Dictionary of decarbonated MIs from Aleutian volcanoes. 
# The average baseline and PCA vectors, determining the components of greatest 
# variability are isolated in CSV. These baselines and PCA vectors are 
# inputted into the Monte Carlo-Markov Chain and the best fits are found. 

output_dir = ["./FIGURES/", "./PLOTFILES/", "./NPZFILES/", "./LOGFILES/"] 

for ii in range(len(output_dir)):
    if not os.path.exists(output_dir[ii]):
       os.makedirs(output_dir[ii])


PCA_matrix = baselines.Load_PCA("Inputs/Baseline_Avg+PCA.csv")
Wavenumber = baselines.Load_Wavenumber("Inputs/Baseline_Avg+PCA.csv")
Peak_1635_PCA_matrix = baselines.Load_PCA("Inputs/Water_Peak_1635_All.csv")
Nvectors = 5
indparams = [Wavenumber, PCA_matrix, Peak_1635_PCA_matrix, Nvectors]

# %% 

# %% Fuego 2018 Samples

# Enter the path to your spectra here
FPATH = 'Inputs/SampleSpectra/Fuego/'
FFILES_ALL = glob.glob(FPATH + "*")
FFILES_ALL.sort()

FTHICKNESS = baselines.Load_ChemistryThickness('Inputs/FuegoThickness.csv')
F_MI_Composition_H2O = baselines.Load_ChemistryThickness('Inputs/FuegoChemistry_NEW_H2O_Fe2O3.csv')

# Load all sample CSVs into this dictionary
FUEGO_FILES, FUEGO_DFS_DICT = baselines.Load_SampleCSV(FFILES_ALL, H2O_wn_high = 5500, H2O_wn_low = 1000)

# %% 

paths = ["Inputs/Baseline_Avg+PCA.csv", "Inputs/Water_Peak_1635_All.csv", 'FUEGO_test3']

plt.close('all')
FUEGO_DF_OUTPUT, FUEGO_VOLATILES_PH, FUEGO_NEAR_IR, FFAILURES = baselines.Run_All_Spectra(FUEGO_DFS_DICT, paths)

# %%

# FUEGO_DF_OUTPUT.to_csv('./FINALDATA/Fuego_DF_Output_OF_F.csv')
# FUEGO_VOLATILES_PH.to_csv('./FINALDATA/Fuego_Volatiles_PH_OF_F.csv')
# FUEGO_NEAR_IR.to_csv('./FINALDATA/Fuego_NIR_OF_F.csv')

# FUEGO_DF_OUTPUT = pd.read_csv('./FINALDATA/F18_DF_Output_OF_Fe2O3.csv', index_col=0)
# FUEGO_VOLATILES_PH = pd.read_csv('./FINALDATA/F18_Volatiles_PH_OF_Fe2O3.csv', index_col=0)
# FUEGO_NEAR_IR = pd.read_csv('./FINALDATA/F18_NIR_OF.csv_Fe2O3', index_col=0)


# %% 

N = 500000
FUEGO_EPSILON, FUEGO_MEGA_SPREADSHEET = Concentration_Output(FUEGO_VOLATILES_PH, N, FTHICKNESS, F_MI_Composition_H2O)
FUEGO_MEGA_SPREADSHEET

FUEGO_MEGA_SPREADSHEET.to_csv('./FINALDATA/F18_H2OCO2_OF_F.csv')
FUEGO_EPSILON.to_csv('./FINALDATA/F18_Epsilon_OF_F.csv')

FUEGO_EPSILON_2, FUEGO_MEGA_SPREADSHEET_2 = H2O_Concentration_Output(FUEGO_NEAR_IR, FUEGO_MEGA_SPREADSHEET, FUEGO_DF_OUTPUT, N, FTHICKNESS, F_MI_Composition_H2O, FUEGO_EPSILON['Density'])
FUEGO_MEGA_SPREADSHEET_2.to_csv('./FINALDATA/F18_5200_4500_H2O_OF_F.csv')
FUEGO_EPSILON_2.to_csv('./FINALDATA/F18_5200_4500_Epsilon_OF_F.csv')

