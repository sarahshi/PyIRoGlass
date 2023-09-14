# %% -*- coding: utf-8 -*-
""" Created on June 12, 2021 // @author: Sarah Shi and Henry Towbin """

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
            CO2_EXP = 2411 # 2333
            CO2_EXP_STD = 241.1 # 233
        elif 'BF76' in j: 
            H2O_EXP = 0.669
            H2O_EXP_STD = 0.0669
            CO2_EXP = 2511 # 1984
            CO2_EXP_STD = 251.1 # 198
        elif 'BF77' in j: 
            H2O_EXP = 0.696
            H2O_EXP_STD = 0.0696
            CO2_EXP = 821 # 709
            CO2_EXP_STD = 82.1 # 70
        elif 'FAB1' in j: 
            H2O_EXP = np.nan
            H2O_EXP_STD = np.nan
            CO2_EXP = np.nan
            CO2_EXP_STD = np.nan
        elif 'NS1' in j: 
            H2O_EXP = 0.37
            H2O_EXP_STD = 0.037
            CO2_EXP = 4812 # 3947
            CO2_EXP_STD = 481.2 # 394
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
# %% 

stdno = 1

MEGA_SPREADSHEET = pd.read_csv(output_dir[-1] + '/' + OUTPUT_PATHS[stdno] + '_DF.csv', index_col = 0) 
MEGA_SPREADSHEET['Sample ID'] = MEGA_SPREADSHEET.index

CONC = pd.read_csv(output_dir[-1] + '/' + OUTPUT_PATHS[stdno] + '_H2OCO2.csv', index_col = 0) 
CONC['Sample ID'] = MEGA_SPREADSHEET.index

HJ = pd.read_csv('BLComp/PyIRoGlass-LHJ.csv', index_col=0)
HJ_peaks = HJ[['Sample ID', '1430 cm-1', '1515 cm-1', '3535 cm-1', 'Background', 'Thickness', 'CO2_EA']]

CONC_conc = CONC[['CO2_MEAN', 'CO2_STD']]

merge = MEGA_SPREADSHEET.merge(HJ_peaks, on='Sample ID')
merge = merge.merge(CONC, on='Sample ID')
merge = merge.set_index('Sample ID')
merge.to_csv('BLComp/PHComparison.csv')

merge

# %%

badspec = np.array(['CI_IPGP_B6_2_50x50_256s_sp1', 'CI_IPGP_B6_1_50x50_256s_sp2', 'CI_IPGP_NBO_2_2_1_100x100_256s_sp1', 
                    'CI_Ref_22_1_100x100_256s_sp1', 'CI_Ref_22_1_100x100_256s_sp2', 
                    'CI_Ref_23_1_100x100_256s_sp5_BLcomp', 'CI_Ref_24_1_100x100_256s_sp1',
                    'CI_Ref_bas_1_1_100x100_256s_sp1', 'CI_Ref_bas_1_1_100x100_256s_sp2', 
                    'CI_Ref_bas_1_2_100x100_256s_sp1', 'CI_Ref_bas_1_2_100x100_256s_sp2', 
                    'CI_Ref_bas_2_1_100x100_256s_sp1', 
                    'CI_Ref_bas_2_2_100x100_256s_4sp1', 'CI_Ref_bas_2_2_100x100_256s_sp2', 'CI_Ref_bas_2_2_100x100_256s_sp3', 
                    'CI_Ref_bas_2_3_100x100_256s_sp1', 
                    'CI_Ref_bas_3_3_100x100_256s_sp1', 
                    'CI_Ref_bas_4_2_100x100_256s_sp2',
                    'LMT_BA3_2_50x50_256s_sp1', 'LMT_BA3_2_50x50_256s_sp2', 'CI_LMT_BA5_2_50x50x_256s_sp1'])

merge = merge[~merge.index.isin(badspec)]

# %% 
# %% 


%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams["xtick.major.size"] = 4 # Sets length of ticks
plt.rcParams["ytick.major.size"] = 4 # Sets length of ticks
plt.rcParams["xtick.labelsize"] = 18 # Sets size of numbers on tick marks
plt.rcParams["ytick.labelsize"] = 18 # Sets size of numbers on tick marks
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 18 # Axes labels

import matplotlib as mpl
import matplotlib.cm as mcm
import matplotlib.path as mpath
import matplotlib.colors as mcolors
mpl.rcParams['mathtext.default'] = 'regular'

line = np.array([0, 2.25])
sz = 80
ticks = np.arange(0, 2.5, 0.25)

tab = plt.get_cmap('tab20')
labels_background = list(set(merge['Background']))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(labels_background))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)
background_to_color_idx = {background: idx for idx, background in enumerate(labels_background)}


fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax=ax.flatten()
ax[0].plot(line, line, 'k', lw = 1, zorder = 0, label='1-1 Line')
ax[0].fill_between(line, 0.9*line, 1.1*line, color='gray', edgecolor=None, alpha=0.25, label='10% Uncertainty')
ax[0].scatter(merge['1515 cm-1']/merge.Thickness*50, merge.PH_1515_BP/merge.Thickness*50, s = sz, c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20,)
ax[0].errorbar(merge['1515 cm-1']/merge.Thickness*50, merge.PH_1515_BP/merge.Thickness*50, yerr=merge.PH_1515_STD/merge.Thickness*200, xerr=merge['1515 cm-1']/merge.Thickness*50*0., fmt='none', lw = 0.5, c = 'k', zorder = 10)
ax[0].text(0.035, 2.05, 'A. $\mathregular{CO_{3, 1515}^{2-}}$', ha='left', va ='bottom', size = 20)
ax[0].set_xlim([0, 2.25])
ax[0].set_ylim([0, 2.25])
ax[0].set_xticks(ticks)  # Set x ticks
ax[0].set_yticks(ticks)  # Set y ticks
ax[0].set_ylabel('PyIRoGlass Peak Height')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)


ax[1].plot(line, line, 'k', lw = 1, zorder = 0, label='1-1 Line')
ax[1].fill_between(line, 0.9*line, 1.1*line, color='gray', edgecolor=None, alpha=0.25, label='10% Uncertainty')
ax[1].scatter(merge['1430 cm-1']/merge.Thickness*50, merge.PH_1430_BP/merge.Thickness*50, s = sz, c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20)
ax[1].errorbar(merge['1430 cm-1']/merge.Thickness*50, merge.PH_1430_BP/merge.Thickness*50, yerr=merge.PH_1430_STD/merge.Thickness*200, xerr=merge['1430 cm-1']/merge.Thickness*50*0.05, fmt='none', lw = 0.5, c = 'k', zorder = 10)
ax[1].text(0.035, 2.05, 'B. $\mathregular{CO_{3, 1430}^{2-}}$', ha='left', va ='bottom', size = 20)
ax[1].set_xlim([0, 2.25])
ax[1].set_ylim([0, 2.25])
ax[1].set_xticks(ticks)  # Set x ticks
ax[1].set_yticks(ticks)  # Set y ticks
ax[1].set_xlabel('Devolatilized Peak Height')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].legend(loc='lower right', labelspacing = 0.2, handletextpad = 0.25, handlelength = 1.00, prop={'size': 13}, frameon=False)

plt.tight_layout()
plt.savefig('BLComp/PeakHeightComp.pdf', bbox_inches='tight', pad_inches = 0.025)


# %% 

tab = plt.get_cmap('tab20')
labels_background = list(set(merge['Background']))
cNorm  = mcolors.Normalize(vmin=0, vmax=len(labels_background))
scalarMap = mcm.ScalarMappable(norm=cNorm, cmap=tab)
background_to_color_idx = {background: idx for idx, background in enumerate(labels_background)}
merge['Py_Devol_1430'] = merge['PH_1430_BP'] / merge['1430 cm-1']
merge['Py_Devol_1515'] = merge['PH_1515_BP'] / merge['1515 cm-1']
line = np.array([0, 2.25])
ticks = np.arange(0, 2.5, 0.25)

fig, ax = plt.subplots(1, 2, figsize=(14, 7))
ax.flatten()
sc2 = ax[0].scatter(merge['PH_1515_BP']/merge.Thickness*50, (merge['Py_Devol_1515']), s = sz, 
                    c = '#0C7BDC', ec = '#171008', lw = 0.5, 
                    zorder = 20)
sc1 = ax[1].scatter(merge['PH_1430_BP']/merge.Thickness*50, (merge['Py_Devol_1430']), s = sz, 
                    c = '#0C7BDC', ec = '#171008', lw = 0.5, 
                    zorder = 20)
ax[0].axhline(np.mean(merge['Py_Devol_1515']), color='k', linestyle='--', dashes = (10, 10), linewidth=0.75,)
ax[0].text(0.035, 1.075, 'A. $\mathregular{CO_{3, 1515}^{2-}}$', ha='left', va ='bottom', size = 20)
ax[0].fill_between(line, np.mean(merge['Py_Devol_1515'])-np.std(merge['Py_Devol_1515']), np.mean(merge['Py_Devol_1515'])+np.std(merge['Py_Devol_1515']), color = 'k', alpha=0.10, edgecolor = None,
    zorder = -5, label='68% Confidence Interval')

ax[1].axhline(np.mean(merge['Py_Devol_1430']), color='k', linestyle='--', dashes = (10, 10), linewidth=0.75, label='Mean')
ax[1].text(0.035, 1.075, 'B. $\mathregular{CO_{3, 1430}^{2-}}$', ha='left', va ='bottom', size = 20)
ax[1].fill_between(line, np.mean(merge['Py_Devol_1430'])-np.std(merge['Py_Devol_1430']), np.mean(merge['Py_Devol_1430'])+np.std(merge['Py_Devol_1430']), color = 'k', alpha=0.10, edgecolor = None,
    zorder = -5, label='68% Confidence Interval')
ax[1].legend(loc='lower right', labelspacing = 0.2, handletextpad = 0.25, handlelength = 1.00, prop={'size': 13}, frameon=False)

ax[0].set_xlabel('PyIRoGlass Peak Height')
ax[0].set_ylabel('PyIRoGlass/Devolatilized Peak Height')
ax[0].set_xlim([0, 2.25])
ax[0].set_xticks(ticks)  # Set x ticks
ax[0].set_ylim([0.8, 1.1])
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

ax[1].set_xlim([0, 2.25])
ax[1].set_xticks(ticks)  # Set x ticks
ax[1].set_ylim([0.8, 1.1])
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('BLComp/Ratios.pdf', bbox_inches='tight', pad_inches = 0.025)


# %% 

def NBO_T(MI_Composition): 


    # Define a dictionary of molar masses for each oxide
    molar_mass = {'SiO2': 60.08, 'TiO2': 79.866, 'Al2O3': 101.96, 'Fe2O3': 159.69, 'FeO': 71.844, 'MnO': 70.9374, 
                'MgO': 40.3044, 'CaO': 56.0774, 'Na2O': 61.9789, 'K2O': 94.2, 'P2O5': 141.9445}

    # Create an empty dataframe to store the mole fraction of each oxide in the MI composition
    mol = pd.DataFrame()
    # Calculate the mole fraction of each oxide by dividing its mole fraction by its molar mass
    for oxide in MI_Composition:
        mol[oxide] = MI_Composition[oxide]/molar_mass[oxide]

    # Calculate the total mole fraction for the MI composition
    mol_tot = pd.DataFrame()
    mol_tot = mol.sum(axis = 1)

    t = mol['SiO2'] + 2*mol['Al2O3'] + 2*mol['Fe2O3']
    o = mol.sum(axis=1) + mol['SiO2'] + mol['TiO2'] + 2*mol['Al2O3'] + 2*mol['Fe2O3'] + 4*mol['P2O5']

    nbo_t = ((2*o)-(4*t))/t



    return nbo_t

# %% 

MICOMP0, THICKNESS0 = pig.Load_ChemistryThickness(CHEMTHICK_PATH[fuegono])
MICOMP1, THICKNESS1 = pig.Load_ChemistryThickness(CHEMTHICK_PATH[stdno])
MICOMP2, THICKNESS2 = pig.Load_ChemistryThickness(CHEMTHICK_PATH[fuegorhno])

MICOMP = pd.concat([MICOMP0, MICOMP1, MICOMP2])
THICKNESS = pd.concat([THICKNESS0, THICKNESS1, THICKNESS2])
THICKNESS_only = THICKNESS.Thickness

nbo_t = NBO_T(MICOMP)
nbo_t_lim = nbo_t 
THICKNESS_lim = THICKNESS[MICOMP.Fe2O3 != 0]
nbo_t_lim = nbo_t[MICOMP.Fe2O3 != 0]

MEGA_SPREADSHEET0 = pd.read_csv(output_dir[-1] + '/' + OUTPUT_PATHS[fuegono] + '_DF.csv') #, index_col=0)
MEGA_SPREADSHEET1 = pd.read_csv(output_dir[-1] + '/' + OUTPUT_PATHS[stdno] + '_DF.csv') #, index_col=0)
MEGA_SPREADSHEET2 = pd.read_csv(output_dir[-1] + '/' + OUTPUT_PATHS[fuegorhno] + '_DF.csv') #, index_col=0)
DENSEPS0 = pd.read_csv(output_dir[-1] + '/' + OUTPUT_PATHS[fuegono] + '_DensityEpsilon.csv') #, index_col=0)
DENSEPS1 = pd.read_csv(output_dir[-1] + '/' + OUTPUT_PATHS[stdno] + '_DensityEpsilon.csv') #, index_col=0)
DENSEPS2 = pd.read_csv(output_dir[-1] + '/' + OUTPUT_PATHS[fuegorhno] + '_DensityEpsilon.csv') #, index_col=0)

MEGA_SPREADSHEET = pd.concat([MEGA_SPREADSHEET0, MEGA_SPREADSHEET1, MEGA_SPREADSHEET2])
MEGA_SPREADSHEET = MEGA_SPREADSHEET.rename(columns={'Unnamed: 0': 'Sample'})
MEGA_SPREADSHEET = MEGA_SPREADSHEET.set_index('Sample')
MEGA_SPREADSHEET_lim = MEGA_SPREADSHEET[MICOMP.Fe2O3 != 0]

DENSEPS = pd.concat([DENSEPS0, DENSEPS1, DENSEPS2])
DENSEPS = DENSEPS.rename(columns={'Unnamed: 0': 'Sample'})
DENSEPS = DENSEPS.set_index('Sample')
DENSEPS1 = DENSEPS

THICKNESS_lim = THICKNESS_only.values[~np.isnan(DENSEPS1.Density)]
DENSEPS_lim = DENSEPS[~np.isnan(DENSEPS1.Density)]

MEGA_SPREADSHEET_lim = MEGA_SPREADSHEET_lim[['AVG_BL_BP', 'PC1_BP', 'PC2_BP', 'PC3_BP', 'PC4_BP', 'm_BP', 'b_BP', 'PH_1635_PC1_BP', 'PH_1635_PC2_BP']]
MEGA_SPREADSHEET_norm = MEGA_SPREADSHEET_lim.divide(THICKNESS_lim, axis=0) * 100
plots = MEGA_SPREADSHEET_norm.join([nbo_t_lim])
plots = plots.join([DENSEPS_lim[['Density_Sat', 'Tau', 'Eta']]])
plots = plots.rename(columns={0: 'NBO_T'})
plots = plots[plots.NBO_T < 1]
plots = plots[plots.AVG_BL_BP < 8]

p = sns.pairplot(plots, kind='kde', corner=True)
plt.tight_layout()
plt.show()

# %%


p = sns.pairplot(plots, corner=True)
plt.tight_layout()
plt.show()

# %%


# %%
