# %% -*- coding: utf-8 -*-
""" Created on June 12, 2021 // @author: Sarah Shi and Henry Towbin """

# %% Import packages

import os
import sys
import time
import glob

import mc3
import numpy as np
import pandas as pd

import scipy.signal as signal
import scipy.interpolate as interpolate
import MC3_BACKEND as baselines

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc, cm

# %% Create directories for export file sorting. 
# Load FTIR Baseline Dictionary of decarbonated MIs from Aleutian volcanoes. 
# The average baseline and PCA vectors, determining the components of greatest 
# variability are isolated in CSV. These baselines and PCA vectors are 
# inputted into the Monte Carlo-Markov Chain and the best fits are found. 

path_parent = os.path.dirname(os.getcwd())
path_grandparent = os.path.dirname(path_parent)
path_beg = path_parent + '/BASELINES/'
output_dir = ["FIGURES", "PLOTFILES", "NPZFILES", "LOGFILES", "FINALDATA"] 

for ii in range(len(output_dir)):
    if not os.path.exists(path_beg + output_dir[ii]):
       os.makedirs(path_beg + output_dir[ii], exist_ok=True)

PATHS = [path_beg+'Inputs/SampleSpectra/Fuego/', path_beg+'Inputs/SampleSpectra/Standards/', path_beg+'Inputs/SampleSpectra/Fuego1974RH/']
THICKNESS_PATH = [path_beg+'Inputs/FuegoThickness.csv', path_beg+'Inputs/StandardThickness.csv', path_beg+'Inputs/DanRHThickness.csv']
MICOMP_PATH = [path_beg+'Inputs/FuegoChemistry_NEW_H2O_Fe2O3.csv', path_beg+'Inputs/StandardChemistry.csv', path_beg+'Inputs/DanRHChemistry.csv']
INPUT_PATHS = [[path_beg+"Inputs/Baseline_Avg+PCA.csv", path_beg+"Inputs/Water_Peak_1635_All.csv", path_beg, 'FUEGO_F'], 
    [path_beg+"Inputs/Baseline_Avg+PCA.csv", path_beg+"Inputs/Water_Peak_1635_All.csv", path_beg, 'STD_F'], 
    [path_beg+"Inputs/Baseline_Avg+PCA.csv", path_beg+"Inputs/Water_Peak_1635_All.csv", path_beg, 'FRH_F']]
OUTPUT_PATH = ['F18', 'STD', 'FRH', 'Henry']

i = 1 # int(sys.argv[1]) - 1
start_time = time.time()

PATH = PATHS[i]
FILES = glob.glob(PATH + "*")
FILES.sort()

THICKNESS = baselines.Load_ChemistryThickness(THICKNESS_PATH[i])
MICOMP = baselines.Load_ChemistryThickness(MICOMP_PATH[i])

DFS_FILES, DFS_DICT = baselines.Load_SampleCSV(FILES, H2O_wn_high = 5500, H2O_wn_low = 1000)
DF_OUTPUT, FAILURES = baselines.Run_All_Spectra(DFS_DICT, INPUT_PATHS[i])
DF_OUTPUT.to_csv(path_beg + output_dir[-1] + '/' + OUTPUT_PATH[i] + '_DF_F.csv')

N = 500000
DENSITY_EPSILON, MEGA_SPREADSHEET = baselines.Concentration_Output(DF_OUTPUT, N, THICKNESS, MICOMP)
MEGA_SPREADSHEET.to_csv(output_dir[-1] + '/' + OUTPUT_PATH[i] + '_H2OCO2_F.csv')
DENSITY_EPSILON.to_csv(output_dir[-1] + '/' + OUTPUT_PATH[i] + '_DensityEpsilon_F.csv')

# %%

# %% 

MS1 = pd.DataFrame(index = MEGA_SPREADSHEET.index, columns = ['H2O_EXP', 'H2O_EXP_STD', 'CO2_EXP', 'CO2_EXP_STD'])
for j in MEGA_SPREADSHEET.index: 
    if '21ALV1846' in j: 
        H2O_EXP= 1.89
        H2O_EXP_STD = 0.19
        CO2_EXP = np.nan
        CO2_EXP_STD = np.nan
    elif '23WOK5-4' in j: 
        H2O_EXP = 1.6
        H2O_EXP_STD = 0.16
        CO2_EXP = 64	
        CO2_EXP_STD = 6.4
    elif 'ALV1833-11' in j:
        H2O_EXP = 1.2
        H2O_EXP_STD = 0.12
        CO2_EXP = 102
        CO2_EXP_STD = 10.2
    elif 'CD33_12-2-2' in j: 
        H2O_EXP = 0.27
        H2O_EXP_STD = 0.03
        CO2_EXP = 170
        CO2_EXP_STD = 17
    elif 'CD33_22-1-1' in j: 
        H2O_EXP = 0.49
        H2O_EXP_STD = 0.05
        CO2_EXP = 109
        CO2_EXP_STD = 10.9
    elif 'ETFSR_Ol8' in j: 
        H2O_EXP = 4.16
        H2O_EXP_STD = 0.42
        CO2_EXP = np.nan
        CO2_EXP_STD = np.nan
    elif 'Fiege63' in j: 
        H2O_EXP = 3.10
        H2O_EXP_STD = 0.31
        CO2_EXP = np.nan
        CO2_EXP_STD = np.nan
    elif 'Fiege73' in j: 
        H2O_EXP = 4.47
        H2O_EXP_STD = 0.45
        CO2_EXP = np.nan
        CO2_EXP_STD = np.nan
    elif 'STD_C1' in j: 
        H2O_EXP = 3.26
        H2O_EXP_STD = 0.33
        CO2_EXP = 169
        CO2_EXP_STD = 16.9
    elif 'STD_CN92C_OL2' in j: 
        H2O_EXP = 4.55
        H2O_EXP_STD = 0.46
        CO2_EXP = 270
        CO2_EXP_STD = 27
    elif 'STD_D1010' in j: 
        H2O_EXP = 1.13
        H2O_EXP_STD = 0.11
        CO2_EXP = 139
        CO2_EXP_STD = 13.9
    elif 'STD_ETFS' in j: 
        H2O_EXP = np.nan
        H2O_EXP_STD = np.nan
        CO2_EXP = np.nan
        CO2_EXP_STD = np.nan
    elif 'VF74_127-7' in j: 
        H2O_EXP = 3.98
        H2O_EXP_STD = 0.39
        CO2_EXP = 439
        CO2_EXP_STD = 43.9
    elif 'VF74_131-1' in j: 
        H2O_EXP = np.nan
        H2O_EXP_STD = np.nan
        CO2_EXP = np.nan
        CO2_EXP_STD = np.nan
    elif 'VF74_131-9' in j: 
        H2O_EXP = np.nan
        H2O_EXP_STD = np.nan
        CO2_EXP = np.nan
        CO2_EXP_STD = np.nan
    elif 'VF74_132-1' in j: 
        H2O_EXP = np.nan
        H2O_EXP_STD = np.nan
        CO2_EXP = np.nan
        CO2_EXP_STD = np.nan
    elif 'VF74_132-2' in j: 
        H2O_EXP = 3.91
        H2O_EXP_STD = 0.39
        CO2_EXP = 198
        CO2_EXP_STD = 19.8
    elif 'VF74-134D-15' in j: 
        H2O_EXP = np.nan
        H2O_EXP_STD = np.nan
        CO2_EXP = np.nan
        CO2_EXP_STD = np.nan
    elif 'VF74-136-3' in j: 
        H2O_EXP = np.nan
        H2O_EXP_STD = np.nan
        CO2_EXP = np.nan
        CO2_EXP_STD = np.nan

    MS1.loc[i] = pd.Series({'H2O_EXP':H2O_EXP,'H2O_EXP_STD':H2O_EXP_STD,'CO2_EXP':CO2_EXP,'CO2_EXP_STD':CO2_EXP_STD})

MEGA_SPREADSHEET1 = pd.concat([MEGA_SPREADSHEET, MS1], axis = 1)
MEGA_SPREADSHEET1.to_csv(output_dir[-1] + '/' + OUTPUT_PATH[i] + 'H2OCO2_FwSTD.csv')

# %%


# %% 

i = 0 # int(sys.argv[1]) - 1
start_time = time.time()

PATH = PATHS[i]
FILES = glob.glob(PATH + "*")
FILES.sort()

THICKNESS = baselines.Load_ChemistryThickness(THICKNESS_PATH[i])
MICOMP = baselines.Load_ChemistryThickness(MICOMP_PATH[i])

DFS_FILES, DFS_DICT = baselines.Load_SampleCSV(FILES, H2O_wn_high = 5500, H2O_wn_low = 1000)
DF_OUTPUT, FAILURES = baselines.Run_All_Spectra(DFS_DICT, INPUT_PATHS[i])
DF_OUTPUT.to_csv(path_beg + output_dir[-1] + '/' + OUTPUT_PATH[i] + '_DF_F.csv')

N = 500000
DENSITY_EPSILON, MEGA_SPREADSHEET = baselines.Concentration_Output(DF_OUTPUT, N, THICKNESS, MICOMP)
MEGA_SPREADSHEET.to_csv(output_dir[-1] + '/' + OUTPUT_PATH[i] + '_H2OCO2_F.csv')
DENSITY_EPSILON.to_csv(output_dir[-1] + '/' + OUTPUT_PATH[i] + '_DensityEpsilon_F.csv')

# %%

i = 2 # int(sys.argv[1]) - 1
start_time = time.time()

PATH = PATHS[i]
FILES = glob.glob(PATH + "*")
FILES.sort()

THICKNESS = baselines.Load_ChemistryThickness(THICKNESS_PATH[i])
MICOMP = baselines.Load_ChemistryThickness(MICOMP_PATH[i])

DFS_FILES, DFS_DICT = baselines.Load_SampleCSV(FILES, H2O_wn_high = 5500, H2O_wn_low = 1000)
DF_OUTPUT, FAILURES = baselines.Run_All_Spectra(DFS_DICT, INPUT_PATHS[i])
DF_OUTPUT.to_csv(path_beg + output_dir[-1] + '/' + OUTPUT_PATH[i] + '_DF_F.csv')

N = 500000
DENSITY_EPSILON, MEGA_SPREADSHEET = baselines.Concentration_Output(DF_OUTPUT, N, THICKNESS, MICOMP)
MEGA_SPREADSHEET.to_csv(output_dir[-1] + '/' + OUTPUT_PATH[i] + '_H2OCO2_F.csv')
DENSITY_EPSILON.to_csv(output_dir[-1] + '/' + OUTPUT_PATH[i] + '_DensityEpsilon_F.csv')
