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


# %% 

# Get working paths 
path_beg = os.getcwd() + '/'
path_input = os.getcwd() + '/Inputs/'
path_spec_input = os.getcwd() + '/Inputs/TransmissionSpectra/'
output_dir = ["FIGURES", "PLOTFILES", "NPZTXTFILES", "LOGFILES", "FINALDATA"] 

# Change paths to direct to folder with SampleSpectra -- last bit should be whatever your folder with spectra is called. 
PATHS = [path_spec_input + string for string in ['Fuego/', 'Standards/', 'Fuego1974RH/']]

# Put ChemThick file in Inputs. Direct to what your ChemThick file is called. 
CHEMTHICK_PATH = [path_input + string for string in ['FuegoChemThick.csv', 'StandardChemThick.csv', 'FuegoRHChemThick.csv']]

# Change last value in list to be what you want your output directory to be called. 
OUTPUT_PATHS = ['FUEGO', 'STD', 'FRH']


# %% 

REF_PATH = path_input + 'ReflectanceSpectra/FuegoOl/'
REF_FILES = sorted(glob.glob(REF_PATH + "*"))

REF_DFS_FILES, REF_DFS_DICT = pig.Load_SampleCSV(REF_FILES, wn_high = 2700, wn_low = 2100)

# Use DHZ parameterization of olivine reflectance index. 
n_ol = pig.Reflectance_Index(0.72)

REF_FUEGO = pig.Thickness_Process(REF_DFS_DICT, n = n_ol, wn_high = 2700, wn_low = 2100, remove_baseline = True, plotting = False, phaseol = True)

REF_PATH = path_input + '/ReflectanceSpectra/rf_ND70/'
REF_FILES = sorted(glob.glob(REF_PATH + "*"))

REF_DFS_FILES, REF_DFS_DICT = pig.Load_SampleCSV(REF_FILES, wn_high = 2850, wn_low = 1700)

# n=1.546 in the range of 2000-2700 cm^-1 following Nichols and Wysoczanski, 2007 for basaltic glass
n_gl = 1.546

REF_FUEGO = pig.Thickness_Process(REF_DFS_DICT, n = n_gl, wn_high = 2850, wn_low = 1700, remove_baseline = True, plotting = False, phaseol = False)

# %% 
# %%

fuegono = 0 
PATH_F = PATHS[fuegono]
FILES_F = sorted(glob.glob(PATH_F + "*"))

MICOMP_F, THICKNESS_F = pig.Load_ChemistryThickness(CHEMTHICK_PATH[fuegono])

DFS_FILES, DFS_DICT = pig.Load_SampleCSV(FILES_F, wn_high = 5500, wn_low = 1000)
DF_OUTPUT_F, FAILURES_F = pig.Run_All_Spectra(DFS_DICT, OUTPUT_PATHS[fuegono])

DF_OUTPUT_F.to_csv(path_beg + output_dir[-1] + '/' + OUTPUT_PATHS[fuegono] + '_DF.csv')

# DF_OUTPUT = pd.read_csv(path_beg + output_dir[-1] + '/' + OUTPUT_PATHS[fuegono] + '_DF.csv', index_col = 0)

T_ROOM = 25 # C
P_ROOM = 1 # Bar

N = 500000
DENSITY_EPSILON_F, MEGA_SPREADSHEET_F = pig.Concentration_Output(DF_OUTPUT_F, N, THICKNESS_F, MICOMP_F, T_ROOM, P_ROOM)
MEGA_SPREADSHEET_F.to_csv(output_dir[-1] + '/' + OUTPUT_PATHS[fuegono] + '_H2OCO2.csv')
DENSITY_EPSILON_F.to_csv(output_dir[-1] + '/' + OUTPUT_PATHS[fuegono] + '_DensityEpsilon.csv')
MEGA_SPREADSHEET_F

# %%
# %% 

stdno = 1

PATH_S = PATHS[stdno]
FILES_S = sorted(glob.glob(PATH_S + "*"))

MICOMP_S, THICKNESS_S = pig.Load_ChemistryThickness(CHEMTHICK_PATH[stdno])

DFS_FILES_S, DFS_DICT_S = pig.Load_SampleCSV(FILES_S, wn_high = 5500, wn_low = 1000)
DF_OUTPUT_S, FAILURES_S = pig.Run_All_Spectra(DFS_DICT_S, OUTPUT_PATHS[stdno])
DF_OUTPUT_S.to_csv(path_beg + output_dir[-1] + '/' + OUTPUT_PATHS[stdno] + '_DF.csv')

# DF_OUTPUT = pd.read_csv(path_beg + output_dir[-1] + '/' + OUTPUT_PATHS[stdno] + '_DF.csv', index_col = 0)

T_ROOM = 25 # C
P_ROOM = 1 # Bar

N = 500000
DENSITY_EPSILON_S, MEGA_SPREADSHEET_S = pig.Concentration_Output(DF_OUTPUT_S, N, THICKNESS_S, MICOMP_S, T_ROOM, P_ROOM)
MEGA_SPREADSHEET_S.to_csv(output_dir[-1] + '/' + OUTPUT_PATHS[stdno] + '_H2OCO2.csv')
DENSITY_EPSILON_S.to_csv(output_dir[-1] + '/' + OUTPUT_PATHS[stdno] + '_DensityEpsilon.csv')
MEGA_SPREADSHEET_S

# %%

stdno = 1

MEGA_SPREADSHEET = pd.read_csv(output_dir[-1] + '/' + OUTPUT_PATHS[stdno] + '_H2OCO2.csv', index_col = 0) 

def STD_DF_MOD(MEGA_SPREADSHEET):
    STD_VAL = pd.DataFrame(index = MEGA_SPREADSHEET.index, columns = ['H2O_EXP', 'H2O_EXP_STD', 'CO2_EXP', 'CO2_EXP_STD'])

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
            H2O_EXP = 2.97
            H2O_EXP_STD = 0.29
            CO2_EXP = np.nan
            CO2_EXP_STD = np.nan
        elif 'Fiege73' in j: 
            H2O_EXP = 4.40
            H2O_EXP_STD = 0.44
            CO2_EXP = np.nan
            CO2_EXP_STD = np.nan
        elif 'STD_C1' in j: 
            H2O_EXP = 3.26
            H2O_EXP_STD = H2O_EXP * 0.054
            CO2_EXP = 169
            CO2_EXP_STD = 16.9
        elif 'STD_CN92C_OL2' in j: 
            H2O_EXP = 4.55
            H2O_EXP_STD = H2O_EXP * 0.054
            CO2_EXP = 270
            CO2_EXP_STD = 27
        elif 'STD_D1010' in j: 
            H2O_EXP = 1.13
            H2O_EXP_STD = 0.11
            CO2_EXP = 139
            CO2_EXP_STD = 13.9
        elif 'STD_ETF' in j: 
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
        elif 'VF74_134D-15' in j: 
            H2O_EXP = np.nan
            H2O_EXP_STD = np.nan
            CO2_EXP = np.nan
            CO2_EXP_STD = np.nan
        elif 'VF74_136-3' in j: 
            H2O_EXP = np.nan
            H2O_EXP_STD = np.nan
            CO2_EXP = np.nan
            CO2_EXP_STD = np.nan
        elif 'BF73' in j: 
            H2O_EXP = 0.715
            H2O_EXP_STD = 0.0715
            CO2_EXP = 2170 # 2333
            CO2_EXP_STD = 20 # 233
        elif 'BF76' in j: 
            H2O_EXP = 0.669
            H2O_EXP_STD = 0.0669
            CO2_EXP = 2052 # 1984
            CO2_EXP_STD = 50 # 198
        elif 'BF77' in j: 
            H2O_EXP = 0.696
            H2O_EXP_STD = 0.0696
            CO2_EXP = 708 # 709
            CO2_EXP_STD = 19 # 70
        elif 'FAB1' in j: 
            H2O_EXP = np.nan
            H2O_EXP_STD = np.nan
            CO2_EXP = np.nan
            CO2_EXP_STD = np.nan
        elif 'NS1' in j: 
            H2O_EXP = 0.37
            H2O_EXP_STD = 0.037
            CO2_EXP = 4433 # 3947
            CO2_EXP_STD = 178 # 394
        elif 'M35' in j: 
            H2O_EXP = 4.20
            H2O_EXP_STD = 0.420
            CO2_EXP = 1019
            CO2_EXP_STD = 101.9
        elif 'M43' in j: 
            H2O_EXP = 2.79
            H2O_EXP_STD = 0.279
            CO2_EXP = 3172
            CO2_EXP_STD = 317.2
        else: 
            H2O_EXP = np.nan
            H2O_EXP_STD = np.nan
            CO2_EXP = np.nan
            CO2_EXP_STD = np.nan


        STD_VAL.loc[j] = pd.Series({'H2O_EXP':H2O_EXP,'H2O_EXP_STD':H2O_EXP_STD,'CO2_EXP':CO2_EXP,'CO2_EXP_STD':CO2_EXP_STD})

    MEGA_SPREADSHEET_STD = pd.concat([MEGA_SPREADSHEET, STD_VAL], axis = 1)

    return MEGA_SPREADSHEET_STD

MEGA_SPREADSHEET_STD = STD_DF_MOD(MEGA_SPREADSHEET)

MEGA_SPREADSHEET_STD.to_csv(output_dir[-1] + '/' + OUTPUT_PATHS[stdno] + '_H2OCO2_FwSTD.csv')

MEGA_SPREADSHEET_STD


# %%

fuegorhno = 2 

PATH_R = PATHS[fuegorhno]
FILES_R = sorted(glob.glob(PATH_R + "*"))

MICOMP_R, THICKNESS_R = pig.Load_ChemistryThickness(CHEMTHICK_PATH[fuegorhno])

DFS_FILES_R, DFS_DICT_R = pig.Load_SampleCSV(FILES_R, wn_high = 5500, wn_low = 1000)
DF_OUTPUT_R, FAILURES_R = pig.Run_All_Spectra(DFS_DICT_R, OUTPUT_PATHS[fuegorhno])
DF_OUTPUT_R.to_csv(path_beg + output_dir[-1] + '/' + OUTPUT_PATHS[fuegorhno] + '_DF.csv')

# DF_OUTPUT = pd.read_csv(path_beg + output_dir[-1] + '/' + OUTPUT_PATHS[fuegorhno] + '_DF.csv', index_col = 0)

T_ROOM = 25 # C
P_ROOM = 1 # Bar

N = 500000
DENSITY_EPSILON_R, MEGA_SPREADSHEET_R = pig.Concentration_Output(DF_OUTPUT_R, N, THICKNESS_R, MICOMP_R, T_ROOM, P_ROOM)
MEGA_SPREADSHEET_R.to_csv(output_dir[-1] + '/' + OUTPUT_PATHS[fuegorhno] + '_H2OCO2.csv')
DENSITY_EPSILON_R.to_csv(output_dir[-1] + '/' + OUTPUT_PATHS[fuegorhno] + '_DensityEpsilon.csv')
MEGA_SPREADSHEET_R


# %% 
