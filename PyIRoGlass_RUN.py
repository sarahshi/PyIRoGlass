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
path_input = os.getcwd() + '/Inputs/'
path_spec_input = path_input + 'TransmissionSpectra/'
output_dir = ["FIGURES", "PLOTFILES", "NPZTXTFILES", "LOGFILES", "FINALDATA"] 

# Change paths to direct to folder with SampleSpectra -- last bit should be whatever your folder with spectra is called. 
PATHS = [path_spec_input + string for string in ['testing/', 'Fuego/', 'Standards/', 'Fuego1974RH/']]

# Put ChemThick file in Inputs. Direct to what your ChemThick file is called. 
CHEMTHICK_PATHS = [path_input + string for string in ['FuegoChemThick.csv', 'StandardChemThick.csv', 'FuegoRHChemThick.csv']]

# Change last value in list to be what you want your output directory to be called. 
OUTPUT_PATHS = ['testing', 'FUEGO', 'STD', 'FRH']

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
floader = pig.SampleDataLoader(PATHS[fuegono], CHEMTHICK_PATHS[fuegono], OUTPUT_PATHS[fuegono])
ffiles, fdfs_dict, fchem, fthick, fout, fdataout = floader.load_all_data()

fdf_output, ffailures = pig.calculate_baselines(fdfs_dict, OUTPUT_PATHS[fuegono])
fdf_output.to_csv(fdataout + '_DF.csv')
# fdf_output = pd.read_csv(fout + '_DF.csv', index_col = 0)

fdf_conc = pig.calculate_concentrations(fdf_output, fchem, fthick)
fdf_conc.to_csv(fout + '_H2OCO2.csv')

# %%
# %% 

stdno = 1
sloader = pig.SampleDataLoader(PATHS[stdno], CHEMTHICK_PATHS[stdno], OUTPUT_PATHS[stdno])
sfiles, sdfs_dict, schem, sthick, sout, sdataout = sloader.load_all_data()

sdf_output, sfailures = pig.calculate_baselines(sdfs_dict, OUTPUT_PATHS[stdno])
sdf_output.to_csv(sdataout + '_DF.csv')
# sdf_output = pd.read_csv(sout + '_DF.csv', index_col = 0)

sdf_conc = pig.calculate_concentrations(sdf_output, schem, sthick)
sdf_conc.to_csv(sout + '_H2OCO2.csv')

# %%

stdno = 1
MEGA_SPREADSHEET = pd.read_csv(sout + '_H2OCO2_brounce.csv', index_col = 0) 

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
            CO2_EXP = 2995 # 2170 
            CO2_EXP_STD = 190 # 68 
        elif 'BF76' in j: 
            H2O_EXP = 0.669
            H2O_EXP_STD = 0.0669
            CO2_EXP = 2336 # 2052 
            CO2_EXP_STD = 127 # 68 
        elif 'BF77' in j: 
            H2O_EXP = 0.696
            H2O_EXP_STD = 0.0696
            CO2_EXP = 1030 # 708
            CO2_EXP_STD = 27 # 29 
        elif 'FAB1' in j: 
            H2O_EXP = np.nan
            H2O_EXP_STD = np.nan
            CO2_EXP = np.nan
            CO2_EXP_STD = np.nan
        elif 'NS1' in j: 
            H2O_EXP = 0.37
            H2O_EXP_STD = 0.037
            CO2_EXP = 3154 # 4433
            CO2_EXP_STD = 3154*0.05 # 222
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

MEGA_SPREADSHEET_STD.to_csv(sout + '_H2OCO2_FwSTD_brounce.csv')

MEGA_SPREADSHEET_STD


# %%

frhno = 2 
frhloader = pig.SampleDataLoader(PATHS[frhno], CHEMTHICK_PATHS[frhno], OUTPUT_PATHS[frhno])
frhfiles, frhdfs_dict, frhchem, frhthick, frhout, frhdataout = frhloader.load_all_data()

frhdf_output, frhfailures = pig.calculate_baselines(frhdfs_dict, OUTPUT_PATHS[frhno])
frhdf_output.to_csv(sdataout + '_DF.csv')
# frhdf_output = pd.read_csv(frhout + '_DF.csv', index_col = 0)

frhdf_conc = pig.calculate_concentrations(frhdf_output, frhchem, frhthick)
frhdf_conc.to_csv(frhout + '_H2OCO2.csv')


# %% 
