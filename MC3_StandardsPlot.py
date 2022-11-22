# %% -*- coding: utf-8 -*-
""" Created on June 12, 2021 // @author: Sarah Shi """

# %% Import packages

import os
import sys
import time
import glob
import warnings 
import mc3
import numpy as np
import pandas as pd

import scipy.signal as signal
import scipy.interpolate as interpolate

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc, cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 18})
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams["xtick.major.size"] = 4 # Sets length of ticks
plt.rcParams["ytick.major.size"] = 4 # Sets length of ticks
plt.rcParams["xtick.labelsize"] = 18 # Sets size of numbers on tick marks
plt.rcParams["ytick.labelsize"] = 18 # Sets size of numbers on tick marks
plt.rcParams["axes.labelsize"] = 20 # Axes labels

# %% Load PC components. 

path_parent = os.path.dirname(os.getcwd())
path_beg = path_parent + '/BASELINES/'
path_input = path_parent + '/BASELINES/Inputs/'
output_dir = ["FIGURES", "PLOTFILES", "NPZFILES", "LOGFILES", "FINALDATA"] 

for ii in range(len(output_dir)):
    if not os.path.exists(path_beg + output_dir[ii]):
       os.makedirs(path_beg + output_dir[ii], exist_ok=True)

PATHS = [path_input+'SampleSpectra/Fuego/', path_input+'SampleSpectra/Standards/', path_input+'SampleSpectra/Fuego1974RH/', path_input+'SampleSpectra/SIMS/']
CHEMTHICK_PATH = [path_input+'FuegoChemThick.csv', path_input+'StandardChemThick.csv', path_input+'DanRHChemThick.csv', path_input+'SIMSChemThick.csv']
INPUT_PATHS = [[path_input+'Baseline_AvgPCA.csv', path_input+"Water_Peak_1635_All.csv", path_beg, 'FUEGO_F'],
                [path_input+'Baseline_AvgPCA.csv', path_input+"Water_Peak_1635_All.csv", path_beg, 'STD_F'], 
                [path_input+'Baseline_AvgPCA.csv', path_input+"Water_Peak_1635_All.csv", path_beg, 'FRH_F'],
                [path_input+'Baseline_AvgPCA.csv', path_input+"Water_Peak_1635_All.csv", path_beg, 'SIMS_F']]
OUTPUT_PATH = ['F18', 'STD', 'FRH', 'SIMSSTD']

stdno = 1
MEGA_SPREADSHEET = pd.read_csv(output_dir[-1] + '/' + OUTPUT_PATH[stdno] + '_H2OCO2_FwSTD.csv', index_col = 0)

# %% 

ALV1846 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('21ALV1846')]
WOK5_4 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('23WOK5-4')]
ALV1833_11 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ALV1833-11')]
CD33_12_2_2 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('CD33_12-2-2')]
CD33_22_1_1 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('CD33_22-1-1')]
ETFSR_Ol8 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ETFSR_Ol8')]
Fiege63 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('Fiege63')]
Fiege73 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('Fiege73')]
STD_C1 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('STD_C1')]
STD_CN92C_OL2 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('STD_CN92C_OL2')]
STD_D1010 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('STD_D1010')]
VF74_127_7 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('VF74_127-7')]
VF74_132_2 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('VF74_132-2')]

# STD_ETFS = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('STD_ETFS')]
# VF74_131_1 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('VF74_131-1')]
# VF74_131_9 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('VF74_131-9')]
# VF74_132_1 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('VF74_132-1')]
# VF74_134D_15 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('VF74_134D-15')]
# VF74_136_3 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('VF74_136-3')]

# %% 

def H2O_mean(DF): 
    return DF['H2OT_MEAN'].mean()
def H2O_std(DF): 
    return np.sqrt(np.sum(np.square(DF['H2OT_STD']), axis=0)) / len(DF)
def CO2_mean(DF): 
    return DF['CO2_MEAN'].mean()
def CO2_std(DF): 
    return np.sqrt(np.sum(np.square(DF['CO2_STD']), axis=0)) / len(DF)
def H2O_expmean(DF): 
    return DF['H2O_EXP'][0]
def H2O_expstd(DF): 
    return DF['H2O_EXP_STD'][0]
def CO2_expmean(DF): 
    return DF['CO2_EXP'][0]
def CO2_expstd(DF): 
    return DF['CO2_EXP_STD'][0]
def H2O_rsd(DF): 
    return np.mean(DF['H2OT_STD'] / DF['H2OT_MEAN'])
def CO2_rsd(DF): 
    return np.mean(DF['CO2_STD'] / DF['CO2_MEAN'])

# %% 

h2o_line = np.array([0, 6])
co2_line = np.array([0, 1400])
sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 2, figsize = (18, 8))
ax = ax.flatten()
ax[0].plot(h2o_line, h2o_line, 'k', lw = 1, zorder = 0)

ax[0].scatter(H2O_expmean(STD_D1010), H2O_mean(STD_D1010), s = sz, c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'STD_D1010 (SN et al., 2000)')
ax[0].errorbar(H2O_expmean(STD_D1010), H2O_mean(STD_D1010), xerr = H2O_expstd(STD_D1010), yerr = H2O_mean(STD_D1010) * H2O_rsd(STD_D1010), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(STD_C1), H2O_mean(STD_C1), s = sz, marker = 's', c = '#5DB147', ec = '#171008', lw = 0.5, zorder = 20, label = 'STD_C1 (AB et al., 2019)')
ax[0].errorbar(H2O_expmean(STD_C1), H2O_mean(STD_C1), xerr = H2O_expstd(STD_C1), yerr = H2O_mean(STD_C1) * H2O_rsd(STD_C1), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(STD_CN92C_OL2), H2O_mean(STD_CN92C_OL2), s = sz, marker = 's', c = '#F9E600', ec = '#171008', lw = 0.5, zorder = 20, label = 'STD_CN92C_OL2 (AB et al., 2019)')
ax[0].errorbar(H2O_expmean(STD_CN92C_OL2), H2O_mean(STD_CN92C_OL2), xerr = H2O_expstd(STD_CN92C_OL2), yerr = H2O_mean(STD_CN92C_OL2) * H2O_rsd(STD_CN92C_OL2), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(VF74_127_7), H2O_mean(VF74_127_7), s = sz, marker = 's', c = '#E42211', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-127-7 (AL et al., 2013)')
ax[0].errorbar(H2O_expmean(VF74_127_7), H2O_mean(VF74_127_7), xerr = H2O_expstd(VF74_127_7), yerr = H2O_mean(VF74_127_7) * H2O_rsd(VF74_127_7), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(VF74_132_2), H2O_mean(VF74_132_2), s = sz, marker = 's', c = '#FE7D10', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-132-2 (AL et al., 2013)')
ax[0].errorbar(H2O_expmean(VF74_132_2), H2O_mean(VF74_132_2), xerr = H2O_expstd(VF74_132_2), yerr = H2O_mean(VF74_132_2) * H2O_rsd(VF74_132_2), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(Fiege63), H2O_mean(Fiege63), s = sz, marker = 's', c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 15, label = 'Fiege 63 (AF et al., 2015)')
ax[0].scatter(H2O_expmean(Fiege63), H2O_mean(Fiege63), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax[0].errorbar(H2O_expmean(Fiege63), H2O_mean(Fiege63), xerr = H2O_expstd(Fiege63), yerr = H2O_mean(Fiege63) * H2O_rsd(Fiege63), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(ETFSR_Ol8), H2O_mean(ETFSR_Ol8), s = sz, marker = 's', c = '#8A8A8A', ec = '#171008', lw = 0.5, zorder = 20, label = 'ETFSR_OL8 (AB Thesis)')
ax[0].errorbar(H2O_expmean(ETFSR_Ol8), H2O_mean(ETFSR_Ol8), xerr = H2O_expstd(ETFSR_Ol8), yerr = H2O_mean(ETFSR_Ol8) * H2O_rsd(ETFSR_Ol8), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(Fiege73), H2O_mean(Fiege73), s = sz, marker = 's', c = '#252525', ec = '#171008', lw = 0.5, zorder = 15, label = 'Fiege 73 (AF et al., 2015)')
ax[0].scatter(H2O_expmean(Fiege73), H2O_mean(Fiege73), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax[0].errorbar(H2O_expmean(Fiege73), H2O_mean(Fiege73), xerr = H2O_expstd(Fiege73), yerr = H2O_mean(Fiege73) * H2O_rsd(Fiege73), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(CD33_12_2_2), H2O_mean(CD33_12_2_2), s = sz, c = '#F7F7F7', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-12-2-2 (JA, pers. comm.)')
ax[0].errorbar(H2O_expmean(CD33_12_2_2), H2O_mean(CD33_12_2_2), xerr = H2O_expstd(CD33_12_2_2), yerr = H2O_mean(CD33_12_2_2) * H2O_rsd(CD33_12_2_2), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(CD33_22_1_1), H2O_mean(CD33_22_1_1), s = sz, c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-12-2-2 (JA, pers. comm.)')
ax[0].errorbar(H2O_expmean(CD33_22_1_1), H2O_mean(CD33_22_1_1), xerr = H2O_expstd(CD33_22_1_1), yerr = H2O_mean(CD33_22_1_1) * H2O_rsd(CD33_22_1_1), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(ALV1833_11), H2O_mean(ALV1833_11), s = sz, c = '#969696', ec = '#171008', lw = 0.5, zorder = 20, label = 'ALV1833-11 (SN et al., 2000)')
ax[0].errorbar(H2O_expmean(ALV1833_11), H2O_mean(ALV1833_11), xerr = H2O_expstd(ALV1833_11), yerr = H2O_mean(ALV1833_11) * H2O_rsd(ALV1833_11), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(WOK5_4), H2O_mean(WOK5_4), s = sz, c = '#636363', ec = '#171008', lw = 0.5, zorder = 20, label = '23WOK5-4 (SN et al., 2000)')
ax[0].errorbar(H2O_expmean(WOK5_4), H2O_mean(WOK5_4), xerr = H2O_expstd(WOK5_4), yerr = H2O_mean(WOK5_4) * H2O_rsd(WOK5_4), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(ALV1846), H2O_mean(ALV1846), s = sz, c = '#252525', ec = '#171008', lw = 0.5, zorder = 20, label = '21ALV1846-9 (SN et al., 2000)')
ax[0].errorbar(H2O_expmean(ALV1846), H2O_mean(ALV1846), xerr = H2O_expstd(ALV1846), yerr = H2O_mean(ALV1846) * H2O_rsd(ALV1846), lw = 0.5, c = 'k', zorder = 10)
ax[0].set_xlim([0, 6])
ax[0].set_ylim([0, 6])
ax[0].set_xlabel('H2O Expected (wt.%)')
ax[0].set_ylabel('H2Ot Measured by FTIR (wt.%)')
l1 = ax[0].legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

ftir_sym = ax[0].scatter(np.nan, np.nan, s = sz, ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = 'FTIR')
sims_sym = ax[0].scatter(np.nan, np.nan, s = sz, marker = 's', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = 'SIMS')
ftirs_sym = ax[0].scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = 'H2Ot, 3550 Saturated')
ax[0].legend([ftir_sym, sims_sym, ftirs_sym], ['FTIR', 'SIMS', 'H2Ot, 3550 Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
ax[0].add_artist(l1)


ax[1].plot(co2_line, co2_line, 'k', lw = 1, zorder = 0)

ax[1].scatter(CO2_expmean(STD_D1010), CO2_mean(STD_D1010), s = sz, c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'STD_D1010 (SN et al., 2000)')
ax[1].errorbar(CO2_expmean(STD_D1010), CO2_mean(STD_D1010), xerr = CO2_expstd(STD_D1010), yerr = CO2_mean(STD_D1010) * CO2_rsd(STD_D1010), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(STD_C1), CO2_mean(STD_C1), s = sz, marker = 's', c = '#5DB147', ec = '#171008', lw = 0.5, zorder = 20, label = 'STD_C1 (AB et al., 2019)')
ax[1].errorbar(CO2_expmean(STD_C1), CO2_mean(STD_C1), xerr = CO2_expstd(STD_C1), yerr = CO2_mean(STD_C1) * CO2_rsd(STD_C1), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(STD_CN92C_OL2), CO2_mean(STD_CN92C_OL2), s = sz, marker = 's', c = '#F9E600', ec = '#171008', lw = 0.5, zorder = 20, label = 'STD_CN92C_OL2 (AB et al., 2019)')
ax[1].errorbar(CO2_expmean(STD_CN92C_OL2), CO2_mean(STD_CN92C_OL2), xerr = CO2_expstd(STD_CN92C_OL2), yerr = CO2_mean(STD_CN92C_OL2) * CO2_rsd(STD_CN92C_OL2), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(VF74_127_7), CO2_mean(VF74_127_7), s = sz, marker = 's', c = '#E42211', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-127-7 (AL et al., 2013)')
ax[1].errorbar(CO2_expmean(VF74_127_7), CO2_mean(VF74_127_7), xerr = CO2_expstd(VF74_127_7), yerr = CO2_mean(VF74_127_7) * CO2_rsd(VF74_127_7), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(VF74_132_2), CO2_mean(VF74_132_2), s = sz, marker = 's', c = '#FE7D10', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-132-2 (AL et al., 2013)')
ax[1].errorbar(CO2_expmean(VF74_132_2), CO2_mean(VF74_132_2), xerr = CO2_expstd(VF74_132_2), yerr = CO2_mean(VF74_132_2) * CO2_rsd(VF74_132_2), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(Fiege63), CO2_mean(Fiege63), s = sz, marker = 's', c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 15)
# ax[1].scatter(CO2_expmean(Fiege63), CO2_mean(Fiege63), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax[1].errorbar(CO2_expmean(Fiege63), CO2_mean(Fiege63), xerr = CO2_expstd(Fiege63), yerr = CO2_mean(Fiege63) * CO2_rsd(Fiege63), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(ETFSR_Ol8), CO2_mean(ETFSR_Ol8), s = sz, marker = 's', c = '#8A8A8A', ec = '#171008', lw = 0.5, zorder = 20)
# ax[1].errorbar(CO2_expmean(ETFSR_Ol8), CO2_mean(ETFSR_Ol8), xerr = CO2_expstd(ETFSR_Ol8), yerr = CO2_mean(ETFSR_Ol8) * CO2_rsd(ETFSR_Ol8), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(Fiege73), CO2_mean(Fiege73), s = sz, marker = 's', c = '#252525', ec = '#171008', lw = 0.5, zorder = 15)
# ax[1].scatter(CO2_expmean(Fiege73), CO2_mean(Fiege73), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax[1].errorbar(CO2_expmean(Fiege73), CO2_mean(Fiege73), xerr = CO2_expstd(Fiege73), yerr = CO2_mean(Fiege73) * CO2_rsd(Fiege73), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), s = sz, c = '#F7F7F7', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-12-2-2 (JA, pers. comm.)')
ax[1].errorbar(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), xerr = CO2_expstd(CD33_12_2_2), yerr = CO2_mean(CD33_12_2_2) * CO2_rsd(CD33_12_2_2), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), s = sz, c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-22-1-1 (JA, pers. comm.)')
ax[1].errorbar(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), xerr = CO2_expstd(CD33_22_1_1), yerr = CO2_mean(CD33_22_1_1) * CO2_rsd(CD33_22_1_1), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11), s = sz, c = '#969696', ec = '#171008', lw = 0.5, zorder = 20, label = 'ALV1833-11 (SN et al., 2000)')
ax[1].errorbar(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11), xerr = CO2_expstd(ALV1833_11), yerr = CO2_mean(ALV1833_11) * CO2_rsd(ALV1833_11), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(WOK5_4), CO2_mean(WOK5_4), s = sz, c = '#636363', ec = '#171008', lw = 0.5, zorder = 20, label = '23WOK5-4 (SN et al., 2000)')
ax[1].errorbar(CO2_expmean(WOK5_4), CO2_mean(WOK5_4), xerr = CO2_expstd(WOK5_4), yerr = CO2_mean(WOK5_4) * CO2_rsd(WOK5_4), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(ALV1846), CO2_mean(ALV1846), s = sz, c = '#252525', ec = '#171008', lw = 0.5, zorder = 20)
# ax[1].errorbar(CO2_expmean(ALV1846), CO2_mean(ALV1846), xerr = CO2_expstd(ALV1846), yerr = CO2_mean(ALV1846) * CO2_rsd(ALV1846), lw = 0.5, c = 'k', zorder = 10)

ax[1].set_xlim([0, 1400])
ax[1].set_ylim([0, 1400])
ax[1].set_xlabel('CO2 Expected (ppm)')
ax[1].set_ylabel('CO2 Measured by FTIR (ppm)')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].legend(loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)

plt.tight_layout()
# plt.savefig('FTIRSIMS_Comparison.pdf')
plt.show()

# %% 

# %% 

CO2_stdexpmean = np.array([CO2_expmean(STD_D1010), CO2_expmean(STD_C1),
    CO2_expmean(STD_CN92C_OL2), CO2_expmean(VF74_127_7), CO2_expmean(VF74_132_2),
    CO2_expmean(Fiege63), CO2_expmean(ETFSR_Ol8), CO2_expmean(Fiege73),
    CO2_expmean(CD33_12_2_2), CO2_expmean(CD33_22_1_1), CO2_expmean(ALV1833_11), 
    CO2_expmean(WOK5_4), CO2_expmean(ALV1846)])

CO2_stdexpstd = np.array([CO2_expstd(STD_D1010), CO2_expstd(STD_C1),
    CO2_expstd(STD_CN92C_OL2), CO2_expstd(VF74_127_7), CO2_expstd(VF74_132_2), 
     CO2_expstd(Fiege63), CO2_expstd(ETFSR_Ol8), CO2_expstd(Fiege73),
    CO2_expstd(CD33_12_2_2), CO2_expstd(CD33_22_1_1), CO2_expstd(ALV1833_11), 
    CO2_expstd(WOK5_4), CO2_expstd(ALV1846)])

H2O_stdexpmean = np.array([H2O_expmean(STD_D1010), H2O_expmean(STD_C1),
    H2O_expmean(STD_CN92C_OL2), H2O_expmean(VF74_127_7), H2O_expmean(VF74_132_2),
    H2O_expmean(Fiege63), H2O_expmean(ETFSR_Ol8), H2O_expmean(Fiege73), 
    H2O_expmean(CD33_12_2_2), H2O_expmean(CD33_22_1_1), H2O_expmean(ALV1833_11), 
    H2O_expmean(WOK5_4), H2O_expmean(ALV1846)])

H2O_stdexpstd = np.array([H2O_expstd(STD_D1010), H2O_expstd(STD_C1),
    H2O_expstd(STD_CN92C_OL2), H2O_expstd(VF74_127_7), H2O_expstd(VF74_132_2),
    H2O_expstd(Fiege63), H2O_expstd(ETFSR_Ol8), H2O_expstd(Fiege73),
    H2O_expstd(CD33_12_2_2), H2O_expstd(CD33_22_1_1), H2O_expstd(ALV1833_11), 
    H2O_expstd(WOK5_4), H2O_expstd(ALV1846)])

CO2_stdmean = np.array([CO2_mean(STD_D1010), CO2_mean(STD_C1),
    CO2_mean(STD_CN92C_OL2), CO2_mean(VF74_127_7), CO2_mean(VF74_132_2),
    CO2_mean(Fiege63), CO2_mean(ETFSR_Ol8), CO2_mean(Fiege73), 
    CO2_mean(CD33_12_2_2), CO2_mean(CD33_22_1_1), CO2_mean(ALV1833_11), 
    CO2_mean(WOK5_4), CO2_mean(ALV1846)])

H2O_stdmean = np.array([H2O_mean(STD_D1010), H2O_mean(STD_C1),
    H2O_mean(STD_CN92C_OL2), H2O_mean(VF74_127_7), H2O_mean(VF74_132_2),
    H2O_mean(Fiege63), H2O_mean(ETFSR_Ol8), H2O_mean(Fiege73), 
    H2O_mean(CD33_12_2_2), H2O_mean(CD33_22_1_1), H2O_mean(ALV1833_11), 
    H2O_mean(WOK5_4), H2O_mean(ALV1846), ])

H2O_stdrsd = np.array([H2O_rsd(STD_D1010), H2O_rsd(STD_C1),
    H2O_rsd(STD_CN92C_OL2), H2O_rsd(VF74_127_7), H2O_rsd(VF74_132_2), 
    H2O_rsd(Fiege63), H2O_rsd(ETFSR_Ol8), H2O_rsd(Fiege73), 
    H2O_rsd(CD33_12_2_2), H2O_rsd(CD33_22_1_1), H2O_rsd(ALV1833_11), 
    H2O_rsd(WOK5_4), H2O_rsd(ALV1846)])

CO2_stdrsd = np.array([CO2_rsd(STD_D1010), CO2_rsd(STD_C1),
    CO2_rsd(STD_CN92C_OL2), CO2_rsd(VF74_127_7), CO2_rsd(VF74_132_2),
    CO2_rsd(Fiege63), CO2_rsd(ETFSR_Ol8), CO2_rsd(Fiege73), 
    CO2_rsd(CD33_12_2_2), CO2_rsd(CD33_22_1_1), CO2_rsd(ALV1833_11), 
    CO2_rsd(WOK5_4), CO2_rsd(ALV1846)])

# %%

h2o_line = np.array([0, 6])
co2_line = np.array([0, 1400])
sz_sm = 80
sz = 150

names = np.array(['STD_D1010', 'STD_C1', 'STD_CN92C_OL2', 'VF74_127_7', 'VF74_132_2',
    'Fiege63', 'ETFSR_Ol8', 'Fiege73', 'CD33_12_2_2', 'CD33_22_1_1', 'ALV1833_11', 'WOK5_4', 'ALV1846'])

h2o_vmin, h2o_vmax = min(H2O_stdexpmean), max(H2O_stdmean)


fig, ax = plt.subplots(1, 2, figsize = (18, 8))
ax = ax.flatten()
sc1 = ax[0].plot(h2o_line, h2o_line, 'k', lw = 1, zorder = 0)

for i in range(len(names)): 
    if names[i] in ('STD_C1', 'STD_CN92C_OL2', 'VF74_127_7', 'VF74_132_2', 'Fiege63', 'ETFSR_Ol8', 'Fiege73'):
        scatter1 = ax[0].scatter(H2O_stdexpmean[i], H2O_stdmean[i], s = sz, marker = 's', c = H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[0].errorbar(H2O_stdexpmean[i], H2O_stdmean[i], marker = 's', xerr = H2O_stdexpstd[i], yerr = H2O_stdmean[i] * H2O_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
        ax[1].scatter(CO2_stdexpmean[i], CO2_stdmean[i], s = sz, marker = 's', c = H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[1].errorbar(CO2_stdexpmean[i], CO2_stdmean[i], xerr = CO2_stdexpstd[i], yerr = CO2_stdmean[i] * CO2_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
    else: 
        ax[0].scatter(H2O_stdexpmean[i], H2O_stdmean[i], s = sz, c = H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[0].errorbar(H2O_stdexpmean[i], H2O_stdmean[i], xerr = H2O_stdexpstd[i], yerr = H2O_stdmean[i] * H2O_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
        ax[1].scatter(CO2_stdexpmean[i], CO2_stdmean[i], s = sz, c = H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[1].errorbar(CO2_stdexpmean[i], CO2_stdmean[i], xerr = CO2_stdexpstd[i], yerr = CO2_stdmean[i] * CO2_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
ax[0].scatter(H2O_expmean(Fiege63), H2O_mean(Fiege63), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax[0].scatter(H2O_expmean(Fiege73), H2O_mean(Fiege73), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)

ax[0].set_xlim([0, 6])
ax[0].set_ylim([0, 6])
ax[0].set_xlabel('H2O Expected (wt.%)')
ax[0].set_ylabel('H2Ot Measured by FTIR (wt.%)')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

ax[1].plot(co2_line, co2_line, 'k', lw = 1, zorder = 0)
ax[1].set_xlim([0, 1400])
ax[1].set_ylim([0, 1400])
ax[1].set_xlabel('CO2 Expected (ppm)')
ax[1].set_ylabel('CO2 Measured by FTIR (ppm)')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)

cbaxes = inset_axes(ax[1], width="15%", height="5%", loc = 'lower right') 
cbar = fig.colorbar(scatter1, cax=cbaxes, orientation='horizontal')
cbaxes.xaxis.set_ticks_position("top")
cbaxes.tick_params(labelsize=12)

ax[1].text(0.905, 0.13, 'H2O (wt.%)', fontsize = 12, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
plt.tight_layout()
# plt.savefig('FTIRSIMS_Comparison_H2O.pdf')
plt.show()

# %%
# %%
