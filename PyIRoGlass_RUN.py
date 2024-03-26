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

# Change paths to direct to folder with SampleSpectra -- last bit should be whatever your folder with spectra is called. 
PATHS = [path_input + 'TransmissionSpectra/' + string for string in ['Fuego/', 'Standards/', 'Fuego1974RH/']]

# Put ChemThick file in Inputs. Direct to what your ChemThick file is called. 
CHEMTHICK_PATHS = [path_input + string for string in ['FuegoChemThick.csv', 'StandardChemThick.csv', 'FuegoRHChemThick.csv']]

# Change last value in list to be what you want your output directory to be called. 
OUTPUT_PATHS = ['FUEGO', 'STD', 'FRH']

# %%
# %% 

ref_ol_loader = pig.SampleDataLoader(spectrum_path=path_input+'ReflectanceSpectra/FuegoOl/')
ref_ol_dfs_dict = ref_ol_loader.load_spectrum_directory(wn_high=2700, wn_low=2100)

# Use DHZ parameterization of olivine reflectance index. 
n_ol = pig.reflectance_index(0.72)
ref_fuego = pig.calculate_mean_thickness(ref_ol_dfs_dict, n=n_ol, wn_high=2700, wn_low=2100, plotting=False, phaseol=True)

ref_gl_loader = pig.SampleDataLoader(spectrum_path=path_input+'ReflectanceSpectra/rf_ND70/')
ref_gl_dfs_dict = ref_gl_loader.load_spectrum_directory(wn_high=2850, wn_low=1700)

# n=1.546 in the range of 2000-2700 cm^-1 following Nichols and Wysoczanski, 2007 for basaltic glass
n_gl = 1.546
ref_nd70 = pig.calculate_mean_thickness(ref_gl_dfs_dict, n=n_gl, wn_high=2850, wn_low=1700, plotting=False, phaseol=False)

# %% 
# %%

fuegono = 0
floader = pig.SampleDataLoader(PATHS[fuegono], CHEMTHICK_PATHS[fuegono])
fdfs_dict, fchem, fthick = floader.load_all_data()

# fdf_output, ffailures = pig.calculate_baselines(fdfs_dict, OUTPUT_PATHS[fuegono])
fdf_output = pd.read_csv('FINALDATA/FUEGO_DF.csv', index_col=0)
fdf_conc = pig.calculate_concentrations(fdf_output, fchem, fthick, OUTPUT_PATHS[fuegono])

# %%
# %% 

stdno = 1
sloader = pig.SampleDataLoader(PATHS[stdno], CHEMTHICK_PATHS[stdno])
sdfs_dict, schem, sthick = sloader.load_all_data()

# sdf_output, sfailures = pig.calculate_baselines(sdfs_dict, OUTPUT_PATHS[stdno])
sdf_output = pd.read_csv('FINALDATA/STD_DF.csv', index_col=0)
sdf_conc = pig.calculate_concentrations(sdf_output, schem, sthick, OUTPUT_PATHS[stdno])

# %%

MEGA_SPREADSHEET = pd.read_csv('FINALDATA/STD_H2OCO2.csv', index_col = 0) 

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
            CO2_EXP = 2995
            CO2_EXP_STD = 190
        elif 'BF76' in j: 
            H2O_EXP = 0.669
            H2O_EXP_STD = 0.0669
            CO2_EXP = 2336
            CO2_EXP_STD = 127
        elif 'BF77' in j: 
            H2O_EXP = 0.696
            H2O_EXP_STD = 0.0696
            CO2_EXP = 1030
            CO2_EXP_STD = 27
        elif 'FAB1' in j: 
            H2O_EXP = np.nan
            H2O_EXP_STD = np.nan
            CO2_EXP = np.nan
            CO2_EXP_STD = np.nan
        elif 'NS1' in j: 
            H2O_EXP = 0.37
            H2O_EXP_STD = 0.037
            CO2_EXP = 3154 
            CO2_EXP_STD = 3154*0.05
        elif 'M35' in j: 
            H2O_EXP = 4.20
            H2O_EXP_STD = 0.420
            CO2_EXP = 1019
            CO2_EXP_STD = 101.9
        elif 'M43' in j: 
            H2O_EXP = 2.62
            H2O_EXP_STD = 0.26
            CO2_EXP = 3172
            CO2_EXP_STD = 317.2
        elif 'INSOL' in j: 
            H2O_EXP = 0.15
            H2O_EXP_STD = 0.01
            CO2_EXP = 8207
            CO2_EXP_STD = 377
        elif 'ND70_02' in j: 
            H2O_EXP = 2.53
            H2O_EXP_STD = 0.24
            CO2_EXP = 1837
            CO2_EXP_STD = 35
        elif 'ND70_03' in j: 
            H2O_EXP = 3.13
            H2O_EXP_STD = 0.30
            CO2_EXP = 2689
            CO2_EXP_STD = 54
        elif 'ND70_04' in j: 
            H2O_EXP = 3.68
            H2O_EXP_STD = 0.35
            CO2_EXP = 4122
            CO2_EXP_STD = 65
        elif 'ND70_05' in j: 
            H2O_EXP = 5.34
            H2O_EXP_STD = 0.51
            CO2_EXP = 12682 
            CO2_EXP_STD = 105
        elif 'ND70_06' in j: 
            H2O_EXP = 6.26
            H2O_EXP_STD = 0.59
            CO2_EXP = 16847
            CO2_EXP_STD = 120
        else: 
            H2O_EXP = np.nan
            H2O_EXP_STD = np.nan
            CO2_EXP = np.nan
            CO2_EXP_STD = np.nan

        STD_VAL.loc[j] = pd.Series({'H2O_EXP':H2O_EXP,'H2O_EXP_STD':H2O_EXP_STD,'CO2_EXP':CO2_EXP,'CO2_EXP_STD':CO2_EXP_STD})

    MEGA_SPREADSHEET_STD = pd.concat([MEGA_SPREADSHEET, STD_VAL], axis = 1)

    return MEGA_SPREADSHEET_STD

MEGA_SPREADSHEET_STD = STD_DF_MOD(MEGA_SPREADSHEET)

MEGA_SPREADSHEET_STD.to_csv('FINALDATA/STD_H2OCO2_FwSTD.csv')

MEGA_SPREADSHEET_STD

# %%
# %%

frhno = 2 
frhloader = pig.SampleDataLoader(PATHS[frhno], CHEMTHICK_PATHS[frhno])
frhdfs_dict, frhchem, frhthick = frhloader.load_all_data()

# frhdf_output, frhfailures = pig.calculate_baselines(frhdfs_dict, OUTPUT_PATHS[frhno])
frhdf_output = pd.read_csv('FINALDATA/FRH_DF.csv', index_col=0)
frhdf_conc = pig.calculate_concentrations(frhdf_output, frhchem, frhthick, OUTPUT_PATHS[frhno])


# %%
