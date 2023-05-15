# %% -*- coding: utf-8 -*-
""" Created on June 12, 2021 // @author: Sarah Shi """

# Import packages

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
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams["xtick.major.size"] = 4 # Sets length of ticks
plt.rcParams["ytick.major.size"] = 4 # Sets length of ticks
plt.rcParams["xtick.labelsize"] = 18 # Sets size of numbers on tick marks
plt.rcParams["ytick.labelsize"] = 18 # Sets size of numbers on tick marks
plt.rcParams["axes.titlesize"] = 20
plt.rcParams["axes.labelsize"] = 20 # Axes labels

# %% Load PC components. 

path_parent = os.path.dirname(os.getcwd())
path_beg =  path_parent + '/'
path_input = path_parent + '/Inputs/'

output_dir = ["FIGURES", "PLOTFILES", "NPZTXTFILES", "LOGFILES", "FINALDATA"] # NPZFILES

# Change paths to direct to folder with SampleSpectra -- last bit should be whatever your folder with spectra is called. 
PATHS = [path_input + string for string in ['TransmissionSpectra/Fuego/', 'TransmissionSpectra/Standards/', 'TransmissionSpectra/Fuego1974RH/', 'TransmissionSpectra/ND70/', 'TransmissionSpectra/HJYM/', 'TransmissionSpectra/YM/']]

# Put ChemThick file in Inputs. Direct to what your ChemThick file is called. 
CHEMTHICK_PATH = [path_input + string for string in ['FuegoChemThick.csv', 'StandardChemThick.csv', 'DanRHChemThick.csv', 'ND70ChemThick.csv', 'HJYMChemThick.csv', 'YMChemThick.csv']]

# Change last value in list to be what you want your output directory to be called. 
INPUT_PATHS = ['FUEGO', 'STD', 'FRH', 'ND70', 'EXPSTD', 'YMSTD']

# Change to be what you want the prefix of your output files to be. 
OUTPUT_PATH = ['FUEGO', 'STD', 'FRH', 'ND70', 'EXPSTD', 'YMSTD']

stdno = 1
MEGA_SPREADSHEET = pd.read_csv(path_parent + '/' + output_dir[-1] + '/' + OUTPUT_PATH[stdno] + '_H2OCO2_FwSTD.csv', index_col = 0)

# %% 

ALV1846 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('21ALV1846')]
WOK5_4 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('23WOK5-4')]
ALV1833_11 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ALV1833-11')]
CD33_12_2_2 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('CD33_12-2-2')]
CD33_22_1_1 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('CD33_22-1-1')]
ETFSR_Ol8 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ETFSR_Ol8')]
Fiege63 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('Fiege63')]
Fiege73 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('Fiege73')]
STD_C1 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('C1')]
STD_CN92C_OL2 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('CN92C_OL2')]
STD_D1010 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('D1010')]
STD_ETF46 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ETF46')]
VF74_127_7 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('VF74_127-7')]
VF74_132_2 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('VF74_132-2')]

# STD_ETFS = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ETFS')]
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

# h2o_line = np.array([0, 6])
# co2_line = np.array([0, 1400])
# sz_sm = 80
# sz = 150
# fig, ax = plt.subplots(1, 2, figsize = (18, 8))
# ax = ax.flatten()
# ax[0].plot(h2o_line, h2o_line, 'k', lw = 1, zorder = 0)

# ax[0].scatter(H2O_expmean(STD_D1010), H2O_mean(STD_D1010), s = sz, c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'D1010 (SN et al., 2000)')
# ax[0].errorbar(H2O_expmean(STD_D1010), H2O_mean(STD_D1010), xerr = H2O_expstd(STD_D1010), yerr = H2O_mean(STD_D1010) * H2O_rsd(STD_D1010), lw = 0.5, c = 'k', zorder = 10)

# ax[0].scatter(H2O_expmean(STD_C1), H2O_mean(STD_C1), s = sz, marker = 's', c = '#5DB147', ec = '#171008', lw = 0.5, zorder = 20, label = "CN_C_OL1' (AB et al., 2019)")
# ax[0].errorbar(H2O_expmean(STD_C1), H2O_mean(STD_C1), xerr = H2O_expstd(STD_C1), yerr = H2O_mean(STD_C1) * H2O_rsd(STD_C1), lw = 0.5, c = 'k', zorder = 10)

# ax[0].scatter(H2O_expmean(STD_CN92C_OL2), H2O_mean(STD_CN92C_OL2), s = sz, marker = 's', c = '#F9E600', ec = '#171008', lw = 0.5, zorder = 20, label = 'CN92C_OL2 (AB et al., 2019)')
# ax[0].errorbar(H2O_expmean(STD_CN92C_OL2), H2O_mean(STD_CN92C_OL2), xerr = H2O_expstd(STD_CN92C_OL2), yerr = H2O_mean(STD_CN92C_OL2) * H2O_rsd(STD_CN92C_OL2), lw = 0.5, c = 'k', zorder = 10)

# ax[0].scatter(H2O_expmean(VF74_127_7), H2O_mean(VF74_127_7), s = sz, marker = 's', c = '#E42211', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-127-7 (AL et al., 2013)')
# ax[0].errorbar(H2O_expmean(VF74_127_7), H2O_mean(VF74_127_7), xerr = H2O_expstd(VF74_127_7), yerr = H2O_mean(VF74_127_7) * H2O_rsd(VF74_127_7), lw = 0.5, c = 'k', zorder = 10)

# ax[0].scatter(H2O_expmean(VF74_132_2), H2O_mean(VF74_132_2), s = sz, marker = 's', c = '#FE7D10', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-132-2 (AL et al., 2013)')
# ax[0].errorbar(H2O_expmean(VF74_132_2), H2O_mean(VF74_132_2), xerr = H2O_expstd(VF74_132_2), yerr = H2O_mean(VF74_132_2) * H2O_rsd(VF74_132_2), lw = 0.5, c = 'k', zorder = 10)

# ax[0].scatter(H2O_expmean(ETFSR_Ol8), H2O_mean(ETFSR_Ol8), s = sz, marker = 's', c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 20, label = 'ETFSR_OL8 (AB Thesis)')
# ax[0].errorbar(H2O_expmean(ETFSR_Ol8), H2O_mean(ETFSR_Ol8), xerr = H2O_expstd(ETFSR_Ol8), yerr = H2O_mean(ETFSR_Ol8) * H2O_rsd(ETFSR_Ol8), lw = 0.5, c = 'k', zorder = 10)

# ax[0].scatter(H2O_expmean(Fiege63), H2O_mean(Fiege63), s = sz, c = '#8A8A8A', ec = '#171008', lw = 0.5, zorder = 15, label = 'ABWCl-F0x (AF et al., 2015)')
# ax[0].scatter(H2O_expmean(Fiege63)+0.01, H2O_mean(Fiege63), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax[0].errorbar(H2O_expmean(Fiege63), H2O_mean(Fiege63), xerr = H2O_expstd(Fiege63), yerr = H2O_mean(Fiege63) * H2O_rsd(Fiege63), lw = 0.5, c = 'k', zorder = 10)

# ax[0].scatter(H2O_expmean(Fiege73), H2O_mean(Fiege73), s = sz, marker = 'D', c = '#252525', ec = '#171008', lw = 0.5, zorder = 15, label = 'ABWB-0x (AF et al., 2015)')
# ax[0].scatter(H2O_expmean(Fiege73)+0.01, H2O_mean(Fiege73), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax[0].errorbar(H2O_expmean(Fiege73), H2O_mean(Fiege73), xerr = H2O_expstd(Fiege73), yerr = H2O_mean(Fiege73) * H2O_rsd(Fiege73), lw = 0.5, c = 'k', zorder = 10)

# ax[0].scatter(H2O_expmean(CD33_12_2_2), H2O_mean(CD33_12_2_2), s = sz, c = '#F7F7F7', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-12-2-2 (JA, pers. comm.)')
# ax[0].errorbar(H2O_expmean(CD33_12_2_2), H2O_mean(CD33_12_2_2), xerr = H2O_expstd(CD33_12_2_2), yerr = H2O_mean(CD33_12_2_2) * H2O_rsd(CD33_12_2_2), lw = 0.5, c = 'k', zorder = 10)

# ax[0].scatter(H2O_expmean(CD33_22_1_1), H2O_mean(CD33_22_1_1), s = sz, c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-12-2-2 (JA, pers. comm.)')
# ax[0].errorbar(H2O_expmean(CD33_22_1_1), H2O_mean(CD33_22_1_1), xerr = H2O_expstd(CD33_22_1_1), yerr = H2O_mean(CD33_22_1_1) * H2O_rsd(CD33_22_1_1), lw = 0.5, c = 'k', zorder = 10)

# ax[0].scatter(H2O_expmean(ALV1833_11), H2O_mean(ALV1833_11), s = sz, c = '#969696', ec = '#171008', lw = 0.5, zorder = 20, label = 'ALV1833-11 (SN et al., 2000)')
# ax[0].errorbar(H2O_expmean(ALV1833_11), H2O_mean(ALV1833_11), xerr = H2O_expstd(ALV1833_11), yerr = H2O_mean(ALV1833_11) * H2O_rsd(ALV1833_11), lw = 0.5, c = 'k', zorder = 10)

# ax[0].scatter(H2O_expmean(WOK5_4), H2O_mean(WOK5_4), s = sz, c = '#636363', ec = '#171008', lw = 0.5, zorder = 20, label = 'WOK5-4 (SN et al., 2000)')
# ax[0].errorbar(H2O_expmean(WOK5_4), H2O_mean(WOK5_4), xerr = H2O_expstd(WOK5_4), yerr = H2O_mean(WOK5_4) * H2O_rsd(WOK5_4), lw = 0.5, c = 'k', zorder = 10)

# ax[0].scatter(H2O_expmean(ALV1846), H2O_mean(ALV1846), s = sz, c = '#252525', ec = '#171008', lw = 0.5, zorder = 20, label = 'ALV1846-9 (SN et al., 2000)')
# ax[0].errorbar(H2O_expmean(ALV1846), H2O_mean(ALV1846), xerr = H2O_expstd(ALV1846), yerr = H2O_mean(ALV1846) * H2O_rsd(ALV1846), lw = 0.5, c = 'k', zorder = 10)
# ax[0].set_xlim([0, 6])
# ax[0].set_ylim([0, 6])
# ax[0].set_xlabel('$\mathregular{H_2O}$ Expected (wt.%)')
# ax[0].set_ylabel('$\mathregular{H_2O_t}$ Measured by FTIR (wt.%)')
# l1 = ax[0].legend(loc = 'upper left', labelspacing = 0.2, handletextpad = 0.25, handlelength = 1.00, prop={'size': 13}, frameon=False)
# ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
# ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

# ftir_sym = ax[0].scatter(np.nan, np.nan, s = sz, ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = 'FTIR')
# sims_sym = ax[0].scatter(np.nan, np.nan, s = 100, marker = 's', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = 'SIMS')
# kft_sym = ax[0].scatter(np.nan, np.nan, s = 100, marker = 'D', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = 'KFT')
# sat_symb = ax[0].scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
# ax[0].legend([ftir_sym, sims_sym, kft_sym, sat_symb], ['FTIR', 'SIMS', 'KFT', '$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.3, handletextpad = 0.25, handlelength = 1.00, prop={'size': 13}, frameon=False)
# ax[0].add_artist(l1)


# ax[1].plot(co2_line, co2_line, 'k', lw = 1, zorder = 0)

# ax[1].scatter(CO2_expmean(STD_D1010), CO2_mean(STD_D1010), s = sz, c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'D1010 (SN et al., 2000)')
# ax[1].errorbar(CO2_expmean(STD_D1010), CO2_mean(STD_D1010), xerr = CO2_expstd(STD_D1010), yerr = CO2_mean(STD_D1010) * CO2_rsd(STD_D1010), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(STD_C1), CO2_mean(STD_C1), s = sz, marker = 's', c = '#5DB147', ec = '#171008', lw = 0.5, zorder = 20, label = "CN_C_OL1' (AB et al., 2019)")
# ax[1].errorbar(CO2_expmean(STD_C1), CO2_mean(STD_C1), xerr = CO2_expstd(STD_C1), yerr = CO2_mean(STD_C1) * CO2_rsd(STD_C1), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(STD_CN92C_OL2), CO2_mean(STD_CN92C_OL2), s = sz, marker = 's', c = '#F9E600', ec = '#171008', lw = 0.5, zorder = 20, label = 'CN92C_OL2 (AB et al., 2019)')
# ax[1].errorbar(CO2_expmean(STD_CN92C_OL2), CO2_mean(STD_CN92C_OL2), xerr = CO2_expstd(STD_CN92C_OL2), yerr = CO2_mean(STD_CN92C_OL2) * CO2_rsd(STD_CN92C_OL2), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(VF74_127_7), CO2_mean(VF74_127_7), s = sz, marker = 's', c = '#E42211', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-127-7 (AL et al., 2013)')
# ax[1].errorbar(CO2_expmean(VF74_127_7), CO2_mean(VF74_127_7), xerr = CO2_expstd(VF74_127_7), yerr = CO2_mean(VF74_127_7) * CO2_rsd(VF74_127_7), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(VF74_132_2), CO2_mean(VF74_132_2), s = sz, marker = 's', c = '#FE7D10', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-132-2 (AL et al., 2013)')
# ax[1].errorbar(CO2_expmean(VF74_132_2), CO2_mean(VF74_132_2), xerr = CO2_expstd(VF74_132_2), yerr = CO2_mean(VF74_132_2) * CO2_rsd(VF74_132_2), lw = 0.5, c = 'k', zorder = 10)

# # ax[1].scatter(CO2_expmean(Fiege63), CO2_mean(Fiege63), s = sz, marker = 's', c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 15)
# # ax[1].scatter(CO2_expmean(Fiege63), CO2_mean(Fiege63), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# # ax[1].errorbar(CO2_expmean(Fiege63), CO2_mean(Fiege63), xerr = CO2_expstd(Fiege63), yerr = CO2_mean(Fiege63) * CO2_rsd(Fiege63), lw = 0.5, c = 'k', zorder = 10)

# # ax[1].scatter(CO2_expmean(ETFSR_Ol8), CO2_mean(ETFSR_Ol8), s = sz, marker = 's', c = '#8A8A8A', ec = '#171008', lw = 0.5, zorder = 20)
# # ax[1].errorbar(CO2_expmean(ETFSR_Ol8), CO2_mean(ETFSR_Ol8), xerr = CO2_expstd(ETFSR_Ol8), yerr = CO2_mean(ETFSR_Ol8) * CO2_rsd(ETFSR_Ol8), lw = 0.5, c = 'k', zorder = 10)

# # ax[1].scatter(CO2_expmean(Fiege73), CO2_mean(Fiege73), s = sz, marker = 's', c = '#252525', ec = '#171008', lw = 0.5, zorder = 15)
# # ax[1].scatter(CO2_expmean(Fiege73), CO2_mean(Fiege73), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# # ax[1].errorbar(CO2_expmean(Fiege73), CO2_mean(Fiege73), xerr = CO2_expstd(Fiege73), yerr = CO2_mean(Fiege73) * CO2_rsd(Fiege73), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), s = sz, c = '#F7F7F7', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-12-2-2 (JA, pers. comm.)')
# ax[1].errorbar(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), xerr = CO2_expstd(CD33_12_2_2), yerr = CO2_mean(CD33_12_2_2) * CO2_rsd(CD33_12_2_2), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), s = sz, c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-22-1-1 (JA, pers. comm.)')
# ax[1].errorbar(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), xerr = CO2_expstd(CD33_22_1_1), yerr = CO2_mean(CD33_22_1_1) * CO2_rsd(CD33_22_1_1), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11), s = sz, c = '#969696', ec = '#171008', lw = 0.5, zorder = 20, label = 'ALV1833-11 (SN et al., 2000)')
# ax[1].errorbar(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11), xerr = CO2_expstd(ALV1833_11), yerr = CO2_mean(ALV1833_11) * CO2_rsd(ALV1833_11), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(WOK5_4), CO2_mean(WOK5_4), s = sz, c = '#636363', ec = '#171008', lw = 0.5, zorder = 20, label = '23WOK5-4 (SN et al., 2000)')
# ax[1].errorbar(CO2_expmean(WOK5_4), CO2_mean(WOK5_4), xerr = CO2_expstd(WOK5_4), yerr = CO2_mean(WOK5_4) * CO2_rsd(WOK5_4), lw = 0.5, c = 'k', zorder = 10)

# # ax[1].scatter(CO2_expmean(ALV1846), CO2_mean(ALV1846), s = sz, c = '#252525', ec = '#171008', lw = 0.5, zorder = 20)
# # ax[1].errorbar(CO2_expmean(ALV1846), CO2_mean(ALV1846), xerr = CO2_expstd(ALV1846), yerr = CO2_mean(ALV1846) * CO2_rsd(ALV1846), lw = 0.5, c = 'k', zorder = 10)

# ax[1].set_xlim([0, 1400])
# ax[1].set_ylim([0, 1400])
# ax[1].set_xlabel('$\mathregular{CO_2}$ Expected (ppm)')
# ax[1].set_ylabel('$\mathregular{CO_2}$ Measured by FTIR (ppm)')
# ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
# ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
# ax[1].legend(loc = 'lower right', labelspacing = 0.2, handletextpad = 0.25, handlelength = 1.00, prop={'size': 13}, frameon=False)

# plt.tight_layout()
# # plt.savefig('FTIRSIMS_Comparison.pdf')
# plt.show()

# %%
# %% no citations


h2o_line = np.array([0, 6])
co2_line = np.array([0, 1400])
sz_sm = 80
sz = 150

fig, ax = plt.subplots(1, 2, figsize = (14, 7))

ax = ax.flatten()
ax[0].plot(h2o_line, h2o_line, 'k', lw = 1, zorder = 0)

ax[0].scatter(H2O_expmean(STD_D1010), H2O_mean(STD_D1010), s = sz, c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'D1010')
ax[0].errorbar(H2O_expmean(STD_D1010), H2O_mean(STD_D1010), xerr = H2O_expstd(STD_D1010), yerr = H2O_mean(STD_D1010) * H2O_rsd(STD_D1010), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(STD_C1), H2O_mean(STD_C1), s = sz, marker = 's', c = '#5DB147', ec = '#171008', lw = 0.5, zorder = 20, label = "CN_C_OL1'")
ax[0].errorbar(H2O_expmean(STD_C1), H2O_mean(STD_C1), xerr = H2O_expstd(STD_C1), yerr = H2O_mean(STD_C1) * H2O_rsd(STD_C1), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(STD_CN92C_OL2), H2O_mean(STD_CN92C_OL2), s = sz, marker = 's', c = '#F9E600', ec = '#171008', lw = 0.5, zorder = 20, label = 'CN92C_OL2')
ax[0].errorbar(H2O_expmean(STD_CN92C_OL2), H2O_mean(STD_CN92C_OL2), xerr = H2O_expstd(STD_CN92C_OL2), yerr = H2O_mean(STD_CN92C_OL2) * H2O_rsd(STD_CN92C_OL2), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(VF74_127_7), H2O_mean(VF74_127_7), s = sz, marker = 's', c = '#E42211', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-127-7')
ax[0].errorbar(H2O_expmean(VF74_127_7), H2O_mean(VF74_127_7), xerr = H2O_expstd(VF74_127_7), yerr = H2O_mean(VF74_127_7) * H2O_rsd(VF74_127_7), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(VF74_132_2), H2O_mean(VF74_132_2), s = sz, marker = 's', c = '#FE7D10', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-132-2')
ax[0].errorbar(H2O_expmean(VF74_132_2), H2O_mean(VF74_132_2), xerr = H2O_expstd(VF74_132_2), yerr = H2O_mean(VF74_132_2) * H2O_rsd(VF74_132_2), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(ETFSR_Ol8), H2O_mean(ETFSR_Ol8), s = sz, marker = 's', c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 20, label = 'ETFSR_OL8')
ax[0].errorbar(H2O_expmean(ETFSR_Ol8), H2O_mean(ETFSR_Ol8), xerr = H2O_expstd(ETFSR_Ol8), yerr = H2O_mean(ETFSR_Ol8) * H2O_rsd(ETFSR_Ol8), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(Fiege63), H2O_mean(Fiege63), s = sz, c = '#8A8A8A', ec = '#171008', lw = 0.5, zorder = 15, label = 'ABWCl-F0x')
ax[0].scatter(H2O_expmean(Fiege63)+0.01, H2O_mean(Fiege63), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax[0].errorbar(H2O_expmean(Fiege63), H2O_mean(Fiege63), xerr = H2O_expstd(Fiege63), yerr = H2O_mean(Fiege63) * H2O_rsd(Fiege63), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(Fiege73), H2O_mean(Fiege73), s = sz, marker = 'D', c = '#252525', ec = '#171008', lw = 0.5, zorder = 15, label = 'ABWB-0x')
ax[0].scatter(H2O_expmean(Fiege73)+0.01, H2O_mean(Fiege73), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax[0].errorbar(H2O_expmean(Fiege73), H2O_mean(Fiege73), xerr = H2O_expstd(Fiege73), yerr = H2O_mean(Fiege73) * H2O_rsd(Fiege73), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(CD33_12_2_2), H2O_mean(CD33_12_2_2), s = sz, c = '#F7F7F7', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-12-2-2')
ax[0].errorbar(H2O_expmean(CD33_12_2_2), H2O_mean(CD33_12_2_2), xerr = H2O_expstd(CD33_12_2_2), yerr = H2O_mean(CD33_12_2_2) * H2O_rsd(CD33_12_2_2), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(CD33_22_1_1), H2O_mean(CD33_22_1_1), s = sz, c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-22-1-1')
ax[0].errorbar(H2O_expmean(CD33_22_1_1), H2O_mean(CD33_22_1_1), xerr = H2O_expstd(CD33_22_1_1), yerr = H2O_mean(CD33_22_1_1) * H2O_rsd(CD33_22_1_1), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(ALV1833_11), H2O_mean(ALV1833_11), s = sz, c = '#969696', ec = '#171008', lw = 0.5, zorder = 20, label = 'ALV1833-11')
ax[0].errorbar(H2O_expmean(ALV1833_11), H2O_mean(ALV1833_11), xerr = H2O_expstd(ALV1833_11), yerr = H2O_mean(ALV1833_11) * H2O_rsd(ALV1833_11), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(WOK5_4), H2O_mean(WOK5_4), s = sz, c = '#636363', ec = '#171008', lw = 0.5, zorder = 20, label = '23WOK5-4')
ax[0].errorbar(H2O_expmean(WOK5_4), H2O_mean(WOK5_4), xerr = H2O_expstd(WOK5_4), yerr = H2O_mean(WOK5_4) * H2O_rsd(WOK5_4), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(ALV1846), H2O_mean(ALV1846), s = sz, c = '#252525', ec = '#171008', lw = 0.5, zorder = 20, label = '21ALV1846-9')
ax[0].errorbar(H2O_expmean(ALV1846), H2O_mean(ALV1846), xerr = H2O_expstd(ALV1846), yerr = H2O_mean(ALV1846) * H2O_rsd(ALV1846), lw = 0.5, c = 'k', zorder = 10)
ax[0].set_xlim([0, 6])
ax[0].set_ylim([0, 6])
ax[0].set_xlabel('$\mathregular{H_2O}$ Expected (wt.%)')
ax[0].set_ylabel('$\mathregular{H_2O_t}$ Measured by FTIR (wt.%)')
l1 = ax[0].legend(loc = (0.01, 0.415), labelspacing = 0.2, handletextpad = 0.25, handlelength = 1.00, prop={'size': 13}, frameon=False)
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

ftir_sym = ax[0].scatter(np.nan, np.nan, s = sz, ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = 'FTIR')
sims_sym = ax[0].scatter(np.nan, np.nan, s = 100, marker = 's', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = 'SIMS')
kft_sym = ax[0].scatter(np.nan, np.nan, s = 100, marker = 'D', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = 'KFT')
sat_symb = ax[0].scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax[0].legend([ftir_sym, sims_sym, kft_sym, sat_symb], ['FTIR', 'SIMS', 'KFT', '$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.3, handletextpad = 0.25, handlelength = 1.00, prop={'size': 13}, frameon=False)
ax[0].add_artist(l1)
ax[0].annotate("A.", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='bold')


ax[1].plot(co2_line, co2_line, 'k', lw = 1, zorder = 0)

ax[1].scatter(CO2_expmean(STD_D1010), CO2_mean(STD_D1010), s = sz, c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'D1010')
ax[1].errorbar(CO2_expmean(STD_D1010), CO2_mean(STD_D1010), xerr = CO2_expstd(STD_D1010), yerr = CO2_mean(STD_D1010) * CO2_rsd(STD_D1010), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(STD_C1), CO2_mean(STD_C1), s = sz, marker = 's', c = '#5DB147', ec = '#171008', lw = 0.5, zorder = 20, label = "CN_C_OL1'")
ax[1].errorbar(CO2_expmean(STD_C1), CO2_mean(STD_C1), xerr = CO2_expstd(STD_C1), yerr = CO2_mean(STD_C1) * CO2_rsd(STD_C1), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(STD_CN92C_OL2), CO2_mean(STD_CN92C_OL2), s = sz, marker = 's', c = '#F9E600', ec = '#171008', lw = 0.5, zorder = 20, label = 'CN92C_OL2')
ax[1].errorbar(CO2_expmean(STD_CN92C_OL2), CO2_mean(STD_CN92C_OL2), xerr = CO2_expstd(STD_CN92C_OL2), yerr = CO2_mean(STD_CN92C_OL2) * CO2_rsd(STD_CN92C_OL2), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(VF74_127_7), CO2_mean(VF74_127_7), s = sz, marker = 's', c = '#E42211', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-127-7')
ax[1].errorbar(CO2_expmean(VF74_127_7), CO2_mean(VF74_127_7), xerr = CO2_expstd(VF74_127_7), yerr = CO2_mean(VF74_127_7) * CO2_rsd(VF74_127_7), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(VF74_132_2), CO2_mean(VF74_132_2), s = sz, marker = 's', c = '#FE7D10', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-132-2')
ax[1].errorbar(CO2_expmean(VF74_132_2), CO2_mean(VF74_132_2), xerr = CO2_expstd(VF74_132_2), yerr = CO2_mean(VF74_132_2) * CO2_rsd(VF74_132_2), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(Fiege63), CO2_mean(Fiege63), s = sz, marker = 's', c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 15)
# ax[1].scatter(CO2_expmean(Fiege63), CO2_mean(Fiege63), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax[1].errorbar(CO2_expmean(Fiege63), CO2_mean(Fiege63), xerr = CO2_expstd(Fiege63), yerr = CO2_mean(Fiege63) * CO2_rsd(Fiege63), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(ETFSR_Ol8), CO2_mean(ETFSR_Ol8), s = sz, marker = 's', c = '#8A8A8A', ec = '#171008', lw = 0.5, zorder = 20)
# ax[1].errorbar(CO2_expmean(ETFSR_Ol8), CO2_mean(ETFSR_Ol8), xerr = CO2_expstd(ETFSR_Ol8), yerr = CO2_mean(ETFSR_Ol8) * CO2_rsd(ETFSR_Ol8), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(Fiege73), CO2_mean(Fiege73), s = sz, marker = 's', c = '#252525', ec = '#171008', lw = 0.5, zorder = 15)
# ax[1].scatter(CO2_expmean(Fiege73), CO2_mean(Fiege73), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax[1].errorbar(CO2_expmean(Fiege73), CO2_mean(Fiege73), xerr = CO2_expstd(Fiege73), yerr = CO2_mean(Fiege73) * CO2_rsd(Fiege73), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), s = sz, c = '#F7F7F7', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-12-2-2')
ax[1].errorbar(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), xerr = CO2_expstd(CD33_12_2_2), yerr = CO2_mean(CD33_12_2_2) * CO2_rsd(CD33_12_2_2), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), s = sz, c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-22-1-1')
ax[1].errorbar(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), xerr = CO2_expstd(CD33_22_1_1), yerr = CO2_mean(CD33_22_1_1) * CO2_rsd(CD33_22_1_1), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11), s = sz, c = '#969696', ec = '#171008', lw = 0.5, zorder = 20, label = 'ALV1833-11')
ax[1].errorbar(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11), xerr = CO2_expstd(ALV1833_11), yerr = CO2_mean(ALV1833_11) * CO2_rsd(ALV1833_11), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(WOK5_4), CO2_mean(WOK5_4), s = sz, c = '#636363', ec = '#171008', lw = 0.5, zorder = 20, label = '23WOK5-4')
ax[1].errorbar(CO2_expmean(WOK5_4), CO2_mean(WOK5_4), xerr = CO2_expstd(WOK5_4), yerr = CO2_mean(WOK5_4) * CO2_rsd(WOK5_4), lw = 0.5, c = 'k', zorder = 10)

# ax[1].scatter(CO2_expmean(ALV1846), CO2_mean(ALV1846), s = sz, c = '#252525', ec = '#171008', lw = 0.5, zorder = 20)
# ax[1].errorbar(CO2_expmean(ALV1846), CO2_mean(ALV1846), xerr = CO2_expstd(ALV1846), yerr = CO2_mean(ALV1846) * CO2_rsd(ALV1846), lw = 0.5, c = 'k', zorder = 10)

ax[1].set_xlim([0, 1400])
ax[1].set_ylim([0, 1400])
ax[1].set_xlabel('$\mathregular{CO_2}$ Expected (ppm)')
ax[1].set_ylabel('$\mathregular{CO_2}$ Measured by FTIR (ppm)')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].legend(loc = 'lower right', labelspacing = 0.2, handletextpad = 0.25, handlelength = 1.00, prop={'size': 13}, frameon=False)
ax[1].annotate("B.", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='bold')

plt.tight_layout()
plt.savefig('FTIRSIMS_Comparison_nocite.pdf')
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

names = np.array(['D1010', 'C1', 'CN92C_OL2', 'VF74_127_7', 'VF74_132_2',
    'Fiege63', 'ETFSR_Ol8', 'Fiege73', 'CD33_12_2_2', 'CD33_22_1_1', 'ALV1833_11', 'WOK5_4', 'ALV1846'])

h2o_vmin, h2o_vmax = min(H2O_stdexpmean), max(H2O_stdmean)


fig, ax = plt.subplots(1, 2, figsize = (18, 8))
ax = ax.flatten()
sc1 = ax[0].plot(h2o_line, h2o_line, 'k', lw = 1, zorder = 0)

for i in range(len(names)): 
    if names[i] in ('C1', 'CN92C_OL2', 'VF74_127_7', 'VF74_132_2', 'ETFSR_Ol8'):
        scatter1 = ax[0].scatter(H2O_stdexpmean[i], H2O_stdmean[i], s = sz, marker = 's', c = H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[0].errorbar(H2O_stdexpmean[i], H2O_stdmean[i], marker = 's', xerr = H2O_stdexpstd[i], yerr = H2O_stdmean[i] * H2O_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
        ax[1].scatter(CO2_stdexpmean[i], CO2_stdmean[i], s = sz, marker = 's', c = H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[1].errorbar(CO2_stdexpmean[i], CO2_stdmean[i], xerr = CO2_stdexpstd[i], yerr = CO2_stdmean[i] * CO2_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
    elif names[i] in ('Fiege73'):
        scatter1 = ax[0].scatter(H2O_stdexpmean[i], H2O_stdmean[i], s = sz, marker = 'D', c = H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[0].errorbar(H2O_stdexpmean[i], H2O_stdmean[i], marker = 's', xerr = H2O_stdexpstd[i], yerr = H2O_stdmean[i] * H2O_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
        ax[1].scatter(CO2_stdexpmean[i], CO2_stdmean[i], s = sz, marker = 's', c = H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[1].errorbar(CO2_stdexpmean[i], CO2_stdmean[i], xerr = CO2_stdexpstd[i], yerr = CO2_stdmean[i] * CO2_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
    else: 
        ax[0].scatter(H2O_stdexpmean[i], H2O_stdmean[i], s = sz, c = H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[0].errorbar(H2O_stdexpmean[i], H2O_stdmean[i], xerr = H2O_stdexpstd[i], yerr = H2O_stdmean[i] * H2O_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
        ax[1].scatter(CO2_stdexpmean[i], CO2_stdmean[i], s = sz, c = H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[1].errorbar(CO2_stdexpmean[i], CO2_stdmean[i], xerr = CO2_stdexpstd[i], yerr = CO2_stdmean[i] * CO2_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
ax[0].scatter(H2O_expmean(Fiege63)+0.01, H2O_mean(Fiege63), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax[0].scatter(H2O_expmean(Fiege73)+0.01, H2O_mean(Fiege73), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)

ax[0].set_xlim([0, 6])
ax[0].set_ylim([0, 6])
ax[0].set_xlabel('$\mathregular{H_2O}$ Expected (wt.%)')
ax[0].set_ylabel('$\mathregular{H_2O_t}$ Measured by FTIR (wt.%)')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

ax[1].plot(co2_line, co2_line, 'k', lw = 1, zorder = 0)
ax[1].set_xlim([0, 1400])
ax[1].set_ylim([0, 1400])
ax[1].set_xlabel('$\mathregular{CO_2}$ Expected (ppm)')
ax[1].set_ylabel('$\mathregular{CO_2}$ Measured by FTIR (ppm)')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)

cbaxes = inset_axes(ax[1], width="15%", height="5%", loc = 'lower right') 
cbar = fig.colorbar(scatter1, cax=cbaxes, orientation='horizontal')
cbaxes.xaxis.set_ticks_position("top")
cbaxes.tick_params(labelsize=12)

ax[1].text(0.905, 0.13, '$\mathregular{H_2O}$ (wt.%)', fontsize = 12, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
plt.tight_layout()
# plt.savefig('FTIRSIMS_Comparison_H2O.pdf')
plt.show()

# %%
# %%




h2o_line = np.array([0, 6])
co2_line = np.array([0, 1400])
sz_sm = 80
sz = 150
fig, ax = plt.subplots(2, 2, figsize = (14, 14))
ax = ax.flatten()
ax[0].plot(h2o_line, h2o_line, 'k', lw = 1, zorder = 0)

ax[0].scatter(H2O_expmean(STD_D1010), H2O_mean(STD_D1010), s = sz, c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'D1010')
ax[0].errorbar(H2O_expmean(STD_D1010), H2O_mean(STD_D1010), xerr = H2O_expstd(STD_D1010), yerr = H2O_mean(STD_D1010) * H2O_rsd(STD_D1010), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(STD_C1), H2O_mean(STD_C1), s = sz, marker = 's', c = '#5DB147', ec = '#171008', lw = 0.5, zorder = 20, label = "CN-C-OL1'")
ax[0].errorbar(H2O_expmean(STD_C1), H2O_mean(STD_C1), xerr = H2O_expstd(STD_C1), yerr = H2O_mean(STD_C1) * H2O_rsd(STD_C1), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(STD_CN92C_OL2), H2O_mean(STD_CN92C_OL2), s = sz, marker = 's', c = '#F9E600', ec = '#171008', lw = 0.5, zorder = 20, label = 'CN92C-OL2')
ax[0].errorbar(H2O_expmean(STD_CN92C_OL2), H2O_mean(STD_CN92C_OL2), xerr = H2O_expstd(STD_CN92C_OL2), yerr = H2O_mean(STD_CN92C_OL2) * H2O_rsd(STD_CN92C_OL2), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(VF74_127_7), H2O_mean(VF74_127_7), s = sz, marker = 's', c = '#E42211', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-127-7')
ax[0].errorbar(H2O_expmean(VF74_127_7), H2O_mean(VF74_127_7), xerr = H2O_expstd(VF74_127_7), yerr = H2O_mean(VF74_127_7) * H2O_rsd(VF74_127_7), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(VF74_132_2), H2O_mean(VF74_132_2), s = sz, marker = 's', c = '#FE7D10', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-132-2')
ax[0].errorbar(H2O_expmean(VF74_132_2), H2O_mean(VF74_132_2), xerr = H2O_expstd(VF74_132_2), yerr = H2O_mean(VF74_132_2) * H2O_rsd(VF74_132_2), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(ETFSR_Ol8), H2O_mean(ETFSR_Ol8), s = sz, marker = 's', c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 20, label = 'ETFSR-OL8')
ax[0].errorbar(H2O_expmean(ETFSR_Ol8), H2O_mean(ETFSR_Ol8), xerr = H2O_expstd(ETFSR_Ol8), yerr = H2O_mean(ETFSR_Ol8) * H2O_rsd(ETFSR_Ol8), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(Fiege63), H2O_mean(Fiege63), s = sz, c = '#8A8A8A', ec = '#171008', lw = 0.5, zorder = 15, label = 'ABWCl-F0x')
ax[0].scatter(H2O_expmean(Fiege63)+0.01, H2O_mean(Fiege63), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax[0].errorbar(H2O_expmean(Fiege63), H2O_mean(Fiege63), xerr = H2O_expstd(Fiege63), yerr = H2O_mean(Fiege63) * H2O_rsd(Fiege63), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(Fiege73), H2O_mean(Fiege73), s = sz, marker = 'D', c = '#252525', ec = '#171008', lw = 0.5, zorder = 15, label = 'ABWB-0x')
ax[0].scatter(H2O_expmean(Fiege73)+0.01, H2O_mean(Fiege73), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax[0].errorbar(H2O_expmean(Fiege73), H2O_mean(Fiege73), xerr = H2O_expstd(Fiege73), yerr = H2O_mean(Fiege73) * H2O_rsd(Fiege73), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(CD33_12_2_2), H2O_mean(CD33_12_2_2), s = sz, marker = 's', c = '#F7F7F7', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-12-2-2')
ax[0].errorbar(H2O_expmean(CD33_12_2_2), H2O_mean(CD33_12_2_2), xerr = H2O_expstd(CD33_12_2_2), yerr = H2O_mean(CD33_12_2_2) * H2O_rsd(CD33_12_2_2), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(CD33_22_1_1), H2O_mean(CD33_22_1_1), s = sz, marker = 's', c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-22-1-1')
ax[0].errorbar(H2O_expmean(CD33_22_1_1), H2O_mean(CD33_22_1_1), xerr = H2O_expstd(CD33_22_1_1), yerr = H2O_mean(CD33_22_1_1) * H2O_rsd(CD33_22_1_1), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(ALV1833_11), H2O_mean(ALV1833_11), s = sz, c = '#969696', ec = '#171008', lw = 0.5, zorder = 20, label = 'ALV1833-11')
ax[0].errorbar(H2O_expmean(ALV1833_11), H2O_mean(ALV1833_11), xerr = H2O_expstd(ALV1833_11), yerr = H2O_mean(ALV1833_11) * H2O_rsd(ALV1833_11), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(WOK5_4), H2O_mean(WOK5_4), s = sz, c = '#636363', ec = '#171008', lw = 0.5, zorder = 20, label = 'WOK5-4')
ax[0].errorbar(H2O_expmean(WOK5_4), H2O_mean(WOK5_4), xerr = H2O_expstd(WOK5_4), yerr = H2O_mean(WOK5_4) * H2O_rsd(WOK5_4), lw = 0.5, c = 'k', zorder = 10)

ax[0].scatter(H2O_expmean(ALV1846), H2O_mean(ALV1846), s = sz, c = '#252525', ec = '#171008', lw = 0.5, zorder = 20, label = 'ALV1846-9')
ax[0].errorbar(H2O_expmean(ALV1846), H2O_mean(ALV1846), xerr = H2O_expstd(ALV1846), yerr = H2O_mean(ALV1846) * H2O_rsd(ALV1846), lw = 0.5, c = 'k', zorder = 10)
ax[0].set_xlim([0, 6])
ax[0].set_ylim([0, 6])
ax[0].annotate("A.", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='bold')
# ax[0].set_xlabel('$\mathregular{H_2O}$ Expected (wt.%)')
ax[0].set_ylabel('$\mathregular{H_2O_t}$ Measured by FTIR (wt.%)')
l1 = ax[0].legend(loc = (0.01, 0.445), labelspacing = 0.2, handletextpad = 0.25, handlelength = 1, prop={'size': 13}, frameon=False)
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

ftir_sym = ax[0].scatter(np.nan, np.nan, s = sz, ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = 'FTIR')
sims_sym = ax[0].scatter(np.nan, np.nan, s = 100, marker = 's', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = 'SIMS')
kft_sym = ax[0].scatter(np.nan, np.nan, s = 100, marker = 'D', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = 'KFT')
sat_symb = ax[0].scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax[0].legend([ftir_sym, sims_sym, kft_sym, sat_symb], ['FTIR', 'SIMS', 'KFT', '$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.2, handletextpad = 0.25, handlelength = 1.00, prop={'size': 13}, frameon=False)
ax[0].add_artist(l1)


ax[1].plot(co2_line, co2_line, 'k', lw = 1, zorder = 0)

ax[1].scatter(CO2_expmean(STD_D1010), CO2_mean(STD_D1010), s = sz, c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'D1010')
ax[1].errorbar(CO2_expmean(STD_D1010), CO2_mean(STD_D1010), xerr = CO2_expstd(STD_D1010), yerr = CO2_mean(STD_D1010) * CO2_rsd(STD_D1010), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(STD_C1), CO2_mean(STD_C1), s = sz, marker = 's', c = '#5DB147', ec = '#171008', lw = 0.5, zorder = 20, label = "CN-C-OL1'")
ax[1].errorbar(CO2_expmean(STD_C1), CO2_mean(STD_C1), xerr = CO2_expstd(STD_C1), yerr = CO2_mean(STD_C1) * CO2_rsd(STD_C1), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(STD_CN92C_OL2), CO2_mean(STD_CN92C_OL2), s = sz, marker = 's', c = '#F9E600', ec = '#171008', lw = 0.5, zorder = 20, label = 'CN92C-OL2')
ax[1].errorbar(CO2_expmean(STD_CN92C_OL2), CO2_mean(STD_CN92C_OL2), xerr = CO2_expstd(STD_CN92C_OL2), yerr = CO2_mean(STD_CN92C_OL2) * CO2_rsd(STD_CN92C_OL2), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(VF74_127_7), CO2_mean(VF74_127_7), s = sz, marker = 's', c = '#E42211', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-127-7')
ax[1].errorbar(CO2_expmean(VF74_127_7), CO2_mean(VF74_127_7), xerr = CO2_expstd(VF74_127_7), yerr = CO2_mean(VF74_127_7) * CO2_rsd(VF74_127_7), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(VF74_132_2), CO2_mean(VF74_132_2), s = sz, marker = 's', c = '#FE7D10', ec = '#171008', lw = 0.5, zorder = 20, label = 'VF74-132-2')
ax[1].errorbar(CO2_expmean(VF74_132_2), CO2_mean(VF74_132_2), xerr = CO2_expstd(VF74_132_2), yerr = CO2_mean(VF74_132_2) * CO2_rsd(VF74_132_2), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), s = sz, marker = 's', c = '#F7F7F7', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-12-2-2')
ax[1].errorbar(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), xerr = CO2_expstd(CD33_12_2_2), yerr = CO2_mean(CD33_12_2_2) * CO2_rsd(CD33_12_2_2), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), s = sz, marker = 's', c = '#CCCCCC', ec = '#171008', lw = 0.5, zorder = 20, label = 'CD33-22-1-1')
ax[1].errorbar(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), xerr = CO2_expstd(CD33_22_1_1), yerr = CO2_mean(CD33_22_1_1) * CO2_rsd(CD33_22_1_1), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11), s = sz, c = '#969696', ec = '#171008', lw = 0.5, zorder = 20, label = 'ALV1833-11')
ax[1].errorbar(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11), xerr = CO2_expstd(ALV1833_11), yerr = CO2_mean(ALV1833_11) * CO2_rsd(ALV1833_11), lw = 0.5, c = 'k', zorder = 10)

ax[1].scatter(CO2_expmean(WOK5_4), CO2_mean(WOK5_4), s = sz, c = '#636363', ec = '#171008', lw = 0.5, zorder = 20, label = 'WOK5-4')
ax[1].errorbar(CO2_expmean(WOK5_4), CO2_mean(WOK5_4), xerr = CO2_expstd(WOK5_4), yerr = CO2_mean(WOK5_4) * CO2_rsd(WOK5_4), lw = 0.5, c = 'k', zorder = 10)

ax[1].set_xlim([0, 1400])
ax[1].set_ylim([0, 1400])
ax[1].annotate("B.", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='bold')

# ax[1].set_title('B.')
# ax[1].set_xlabel('$\mathregular{CO_2}$ Expected (ppm)')
ax[1].set_ylabel('$\mathregular{CO_2}$ Measured by FTIR (ppm)')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].legend(loc = 'lower right', labelspacing = 0.2, handletextpad = 0.25, handlelength = 1.00, prop={'size': 13}, frameon=False)


sc1 = ax[2].plot(h2o_line, h2o_line, 'k', lw = 1, zorder = 0)

for i in range(len(names)): 
    if names[i] in ('CD33_12_2_2', 'CD33_22_1_1', 'C1', 'CN92C_OL2', 'VF74_127_7', 'VF74_132_2', 'ETFSR_Ol8'):
        scatter1 = ax[2].scatter(H2O_stdexpmean[i], H2O_stdmean[i], s = sz, marker = 's', c = H2O_stdmean[i], vmin = 0, vmax = 5.25, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[2].errorbar(H2O_stdexpmean[i], H2O_stdmean[i], marker = 's', xerr = H2O_stdexpstd[i], yerr = H2O_stdmean[i] * H2O_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
        ax[3].scatter(CO2_stdexpmean[i], CO2_stdmean[i], s = sz, marker = 's', c = H2O_stdmean[i], vmin = 0, vmax = 5.25, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[3].errorbar(CO2_stdexpmean[i], CO2_stdmean[i], xerr = CO2_stdexpstd[i], yerr = CO2_stdmean[i] * CO2_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
    elif names[i] in ('Fiege73'):
        scatter2 = ax[2].scatter(H2O_stdexpmean[i], H2O_stdmean[i], s = sz, marker = 'D', c = H2O_stdmean[i], vmin = 0, vmax = 5.25, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[2].errorbar(H2O_stdexpmean[i], H2O_stdmean[i], marker = 's', xerr = H2O_stdexpstd[i], yerr = H2O_stdmean[i] * H2O_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
        ax[3].scatter(CO2_stdexpmean[i], CO2_stdmean[i], s = sz, marker = 's', c = H2O_stdmean[i], vmin = 0, vmax = 5.25, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[3].errorbar(CO2_stdexpmean[i], CO2_stdmean[i], xerr = CO2_stdexpstd[i], yerr = CO2_stdmean[i] * CO2_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
    else: 
        ax[2].scatter(H2O_stdexpmean[i], H2O_stdmean[i], s = sz, c = H2O_stdmean[i], vmin = 0, vmax = 5.25, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[2].errorbar(H2O_stdexpmean[i], H2O_stdmean[i], xerr = H2O_stdexpstd[i], yerr = H2O_stdmean[i] * H2O_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
        ax[3].scatter(CO2_stdexpmean[i], CO2_stdmean[i], s = sz, c = H2O_stdmean[i], vmin = 0, vmax = 5.25, cmap = 'Blues', ec = '#171008', lw = 0.5, zorder = 20)
        ax[3].errorbar(CO2_stdexpmean[i], CO2_stdmean[i], xerr = CO2_stdexpstd[i], yerr = CO2_stdmean[i] * CO2_stdrsd[i], lw = 0.5, ls = 'none', c = 'k', zorder = 10)
ax[2].scatter(H2O_expmean(Fiege63)+0.01, H2O_mean(Fiege63), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax[2].scatter(H2O_expmean(Fiege73)+0.01, H2O_mean(Fiege73), s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)

ax[2].set_xlim([0, 6])
ax[2].set_ylim([0, 6])
ax[2].annotate("C.", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='bold')
ax[2].set_xlabel('$\mathregular{H_2O}$ Expected (wt.%)')
ax[2].set_ylabel('$\mathregular{H_2O_t}$ Measured by FTIR (wt.%)')
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)

ax[3].plot(co2_line, co2_line, 'k', lw = 1, zorder = 0)
ax[3].set_xlim([0, 1400])
ax[3].set_ylim([0, 1400])
ax[3].annotate("D.", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='bold')
ax[3].set_xlabel('$\mathregular{CO_2}$ Expected (ppm)')
ax[3].set_ylabel('$\mathregular{CO_2}$ Measured by FTIR (ppm)')
ax[3].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[3].tick_params(axis="y", direction='in', length=5, pad = 6.5)

cbaxes = inset_axes(ax[2], width="25%", height="7.5%", loc = 'lower right') 
cbar = fig.colorbar(scatter1, cax=cbaxes, orientation='horizontal')
cbaxes.xaxis.set_ticks_position("top")
cbaxes.tick_params(labelsize=16, pad=-2.5)
cbaxes = inset_axes(ax[3], width="25%", height="7.5%", loc = 'lower right') 
cbar = fig.colorbar(scatter1, cax=cbaxes, orientation='horizontal')
cbaxes.xaxis.set_ticks_position("top")
cbaxes.tick_params(labelsize=16, pad=-2.5)

ax[2].text(0.845, 0.16, '$\mathregular{H_2O}$ (wt.%)', fontsize = 18, horizontalalignment='center', verticalalignment='center', transform=ax[2].transAxes)
ax[3].text(0.845, 0.16, '$\mathregular{H_2O}$ (wt.%)', fontsize = 18, horizontalalignment='center', verticalalignment='center', transform=ax[3].transAxes)
plt.tight_layout()
# plt.savefig('FTIRSIMS_Comparison_combined.pdf', bbox_inches='tight', pad_inches = 0.025)

# %%
# %% 
# %% 
# %% 
# %% 
# %% 

df0 = pd.read_csv(path_parent + '/' + output_dir[-1] + '/' + OUTPUT_PATH[0] + '_DF.csv', index_col = 0)
df1 = pd.read_csv(path_parent + '/' + output_dir[-1] + '/' + OUTPUT_PATH[1] + '_DF.csv', index_col = 0)
df2 = pd.read_csv(path_parent + '/' + output_dir[-1] + '/' + OUTPUT_PATH[2] + '_DF.csv', index_col = 0)
df3 = pd.read_csv(path_parent + '/' + output_dir[-1] + '/' + OUTPUT_PATH[3] + '_DF.csv', index_col = 0)
df4 = pd.read_csv(path_parent + '/' + output_dir[-1] + '/' + OUTPUT_PATH[4] + '_DF.csv', index_col = 0)
df5 = pd.read_csv(path_parent + '/' + output_dir[-1] + '/' + OUTPUT_PATH[5] + '_DF.csv', index_col = 0)

t0 = pd.read_csv(CHEMTHICK_PATH[0], index_col = 0)
t1 = pd.read_csv(CHEMTHICK_PATH[1], index_col = 0)
t2 = pd.read_csv(CHEMTHICK_PATH[2], index_col = 0)
t3 = pd.read_csv(CHEMTHICK_PATH[3], index_col = 0)
t4 = pd.read_csv(CHEMTHICK_PATH[4], index_col = 0)
t5 = pd.read_csv(CHEMTHICK_PATH[5], index_col = 0)


col_lim = ['AVG_BL_BP', 'PCA1_BP', 'PCA2_BP', 'PCA3_BP', 'PCA4_BP', 'm_BP', 'b_BP']
chem_lim = ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']

df = pd.concat([df0, df1, df2]) #, df3, df4, df5]) #, df1])
t = pd.concat([t0, t1, t2]) #, t3, t4, t5]) #, t1], axis=0)

dft = pd.concat([df[col_lim], t[chem_lim]], axis=1)
dft_norm = dft.copy()

dft_norm[col_lim[0]] = dft[col_lim[0]] / t.Thickness.ravel() * 100
dft_norm[col_lim[1]] = dft[col_lim[1]] / t.Thickness.ravel() * 100
dft_norm[col_lim[2]] = dft[col_lim[2]] / t.Thickness.ravel() * 100
dft_norm[col_lim[3]] = dft[col_lim[3]] / t.Thickness.ravel() * 100
dft_norm[col_lim[4]] = dft[col_lim[4]] / t.Thickness.ravel() * 100
dft_norm[col_lim[5]] = dft[col_lim[5]] / t.Thickness.ravel() * 100
dft_norm[col_lim[6]] = dft[col_lim[6]] / t.Thickness.ravel() * 100

dft_norm['Fe2O3T'] = np.nan
dft_norm['FeOT'] = np.nan

def Fe_Conversion(df):

    """
    Handle inconsistent Fe speciation in PetDB datasets by converting all to FeOT. 

    Parameters
    --------------
    df:class:`pandas.DataFrame`
        Array of oxide compositions.

    Returns
    --------
    df:class:`pandas.DataFrame`
        Array of oxide compositions with corrected Fe.
    """

    fe_conv = 1.1113
    conditions = [~np.isnan(df['FeO']) & np.isnan(df['FeOT']) & np.isnan(df['Fe2O3']) & np.isnan([df['Fe2O3T']]),
    ~np.isnan(df['FeOT']) & np.isnan(df['FeO']) & np.isnan(df['Fe2O3']) & np.isnan([df['Fe2O3T']]), 
    ~np.isnan(df['Fe2O3']) & np.isnan(df['Fe2O3T']) & np.isnan(df['FeO']) & np.isnan([df['FeOT']]), # 2
    ~np.isnan(df['Fe2O3T']) & np.isnan(df['Fe2O3']) & np.isnan(df['FeO']) & np.isnan([df['FeOT']]), # 2
    ~np.isnan(df['FeO']) & ~np.isnan(df['Fe2O3']) & np.isnan(df['FeOT']) & np.isnan([df['Fe2O3T']]), # 3
    ~np.isnan(df['FeO']) & ~np.isnan(df['FeOT']) & ~np.isnan(df['Fe2O3']) & np.isnan([df['Fe2O3T']]), # 4
    ~np.isnan(df['FeO']) & ~np.isnan(df['Fe2O3']) & ~np.isnan(df['Fe2O3T']) & np.isnan([df['FeOT']]), # 5
    ~np.isnan(df['FeOT']) & ~np.isnan(df['Fe2O3']) & np.isnan(df['Fe2O3T']) & np.isnan([df['FeO']]), # 6
    ~np.isnan(df['Fe2O3']) & ~np.isnan(df['Fe2O3T']) & np.isnan(df['FeO']) & np.isnan([df['FeOT']]) ] # 7

    choices = [ (df['FeO']), (df['FeOT']),
    (df['Fe2O3']),(df['Fe2O3T']),
    (df['FeO'] + (df['Fe2O3'] / fe_conv)), # 3
    (df['FeOT']), # 4 of interest
    (df['Fe2O3T'] / fe_conv), # 5
    (df['FeOT']), # 6
    (df['Fe2O3T'] / fe_conv) ] # 7

    df.insert(10, 'FeOT_F', np.select(conditions, choices))

    return df 


dft_fe_norm = Fe_Conversion(dft_norm)
dft_fe_norm = dft_fe_norm[['AVG_BL_BP', 'PCA1_BP', 'PCA2_BP', 'PCA3_BP', 'PCA4_BP', 'm_BP', 'b_BP',
       'SiO2', 'TiO2', 'Al2O3', 'FeOT_F', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']]
dft_fe_norm = dft_fe_norm.rename(columns={'FeOT_F': 'FeO'})

dft_mol_norm = dft_fe_norm.copy()
dft_mol_norm['SiO2'] = dft_mol_norm['SiO2'] / 60.08
dft_mol_norm['TiO2'] = dft_mol_norm['TiO2'] / 79.866
dft_mol_norm['Al2O3'] = dft_mol_norm['Al2O3'] / 101.96
dft_mol_norm['FeO'] = dft_mol_norm['FeO'] / 71.844
dft_mol_norm['MgO'] = dft_mol_norm['MgO'] / 40.3044
dft_mol_norm['CaO'] = dft_mol_norm['CaO'] / 56.0774
dft_mol_norm['Na2O'] = dft_mol_norm['Na2O'] / 61.9789
dft_mol_norm['K2O'] = dft_mol_norm['K2O'] / 94.2
dft_mol_norm['P2O5'] = dft_mol_norm['P2O5'] / 141.9445


df_norm_lim = dft_fe_norm.copy()
df_norm_lim = df_norm_lim[df_norm_lim['K2O'] < 3]
df_norm_lim = df_norm_lim[df_norm_lim['TiO2'] < 2]
df_norm_lim = df_norm_lim[df_norm_lim['MgO'] < 10]

# %% 

rc('font',**{'family':'Avenir', 'size': 12})
plt.rcParams["xtick.labelsize"] = 12 # Sets size of numbers on tick marks
plt.rcParams["ytick.labelsize"] = 12 # Sets size of numbers on tick marks
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 12 # Axes labels

pd.plotting.scatter_matrix(df_norm_lim, figsize = (15, 15), hist_kwds={'bins':20})
plt.show()

# %% 

dft_mol_norm = dft_mol_norm[dft_mol_norm['AVG_BL_BP'] < 6]
dft_mol_norm = dft_mol_norm[dft_mol_norm['K2O'] < 0.03]

pd.plotting.scatter_matrix(dft_mol_norm, figsize = (15, 15), hist_kwds={'bins':20})
plt.show()


# %% 

df_cat_norm = dft_mol_norm.copy()



# %% 


# %% 
# %%
