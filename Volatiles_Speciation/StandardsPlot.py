# %% -*- coding: utf-8 -*-
""" Created on June 12, 2021 // @author: Sarah Shi """

# Import packages
import os
import sys
import numpy as np
import pandas as pd
import scipy 

sys.path.append('../src/')
import PyIRoGlass as pig

from matplotlib import rc
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams["xtick.major.size"] = 4 # Sets length of ticks
plt.rcParams["ytick.major.size"] = 4 # Sets length of ticks
plt.rcParams["xtick.labelsize"] = 20 # Sets size of numbers on tick marks
plt.rcParams["ytick.labelsize"] = 20 # Sets size of numbers on tick marks
plt.rcParams["axes.titlesize"] = 22
plt.rcParams["axes.labelsize"] = 22 # Axes labels

# %% Load PC components. 

# Get working paths 
path_input = os.getcwd() + '/Inputs/'

# Change paths to direct to folder with SampleSpectra -- last bit should be whatever your folder with spectra is called. 
PATHS = [path_input + 'TransmissionSpectra/' + string for string in ['Fuego/', 'Standards/', 'Fuego1974RH/']]

# Put ChemThick file in Inputs. Direct to what your ChemThick file is called. 
CHEMTHICK_PATHS = [path_input + string for string in ['FuegoChemThick.csv', 'StandardChemThick.csv', 'FuegoRHChemThick.csv']]

# Change last value in list to be what you want your output directory to be called. 
OUTPUT_PATHS = ['FUEGO', 'STD', 'FRH']

# %% 

MEGA_SPREADSHEET = pd.read_csv('../FINALDATA/STD_H2OCO2_FwSTD.csv', index_col = 0)

INSOL = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('INSOL')]
ALV1846 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('21ALV1846')]
WOK5_4 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('23WOK5-4')]
ALV1833_11 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ALV1833-11')]
CD33_12_2_2 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('CD33_12-2-2')]
CD33_22_1_1 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('CD33_22-1-1')]
ETFSR_Ol8 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ETFSR_Ol8')]
Fiege63 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('Fiege63')]
Fiege73 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('Fiege73')]
STD_C1 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('CN_C_OL1')]
STD_CN92C_OL2 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('CN92C_OL2')]
STD_D1010 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('D1010')]
STD_D1010 = STD_D1010.dropna()
STD_ETF46 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ETF46')]
VF74_127_7 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('VF74_127-7')]
VF74_132_2 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('VF74_132-2')]
NS1 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('NS1')]
M35 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('M35')]
M43 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('M43')]
BF73 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('BF73_100x100')]
BF76 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('BF76_100x')]
BF77 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('BF77_50x50')]

ND70_2 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ND70_02')]
ND70_3 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ND70_03')]
ND70_4 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ND70_04')]
ND70_5 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ND70_05')]
ND70_6 = MEGA_SPREADSHEET[MEGA_SPREADSHEET.index.str.contains('ND70_06')]


def H2O_mean(DF): 
    return DF['H2Ot_MEAN'].mean()
def H2O_std(DF): 
    return np.sqrt(np.sum(np.square(DF['H2Ot_STD']), axis=0)) / len(DF)
def CO2_mean(DF): 
    return DF['CO2_MEAN'].mean() / 10000
def CO2_std(DF): 
    return np.sqrt(np.sum(np.square(DF['CO2_STD']), axis=0)) / len(DF) / 10000
def H2O_expmean(DF): 
    return DF['H2O_EXP'].iloc[0]
def H2O_expstd(DF): 
    return DF['H2O_EXP_STD'].iloc[0]
def CO2_expmean(DF): 
    return DF['CO2_EXP'].iloc[0] / 10000
def CO2_expstd(DF): 
    return DF['CO2_EXP_STD'].iloc[0] / 10000
def H2O_rsd(DF): 
    return np.mean(DF['H2Ot_STD'] / DF['H2Ot_MEAN'])
def CO2_rsd(DF): 
    return np.mean(DF['CO2_STD'] / DF['CO2_MEAN'])

def eps_mean(DF): 
    return DF['epsilon_CO2'].iloc[0]

def Error_CO2(DF):
    
    cols = ['CO2_MEAN', 'CO2_STD']
    means = DF[cols].mean()
    std = DF[cols].std()
    mean_std = means['CO2_STD']
    mean_mean = means['CO2_MEAN']
    std_mean = std['CO2_MEAN']
    sigma_analysis = mean_std/mean_mean
    sigma_repeat = std_mean/mean_mean
    sigma_prop = np.sqrt(sigma_analysis**2 + sigma_repeat**2)
    uncert_prop = mean_mean * sigma_prop / 10000

    return uncert_prop

def Error_H2O(DF):
    
    cols = ['H2Ot_MEAN', 'H2Ot_STD']
    means = DF[cols].mean()
    std = DF[cols].std()
    mean_std = means['H2Ot_STD']
    mean_mean = means['H2Ot_MEAN']
    std_mean = std['H2Ot_MEAN']
    sigma_analysis = mean_std/mean_mean
    sigma_repeat = std_mean/mean_mean
    sigma_prop = np.sqrt(sigma_analysis**2 + sigma_repeat**2)
    uncert_prop = mean_mean * sigma_prop

    return uncert_prop

# %% ORIGINAL 

h2o_line = np.array([0, 7])
co2_line = np.array([0, 3])
sz_sm = 50
sz = 90

fig, ax = plt.subplots(1, 2, figsize = (14, 7))

ax = ax.flatten()
ax[0].plot(h2o_line, h2o_line, 'k', lw=0.5, zorder=0)
ax[0].scatter(H2O_expmean(Fiege63), H2O_mean(Fiege63), s=sz, c='#fff7bc', ec='#171008', lw=0.5, zorder=15, label='ABWCl-F0x')
ax[0].scatter(H2O_expmean(Fiege63)+0.01, H2O_mean(Fiege63), s=sz_sm-10, marker='>', c='#FFFFFF', ec='#171008', lw=0.5, zorder=20)
ax[0].errorbar(H2O_expmean(Fiege63), H2O_mean(Fiege63), xerr=H2O_expstd(Fiege63), yerr=Error_H2O(Fiege63), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(Fiege73), H2O_mean(Fiege73), s=sz-15, marker='D', c='#fee392', ec='#171008', lw=0.5, zorder=15, label='ABWB-0x')
ax[0].scatter(H2O_expmean(Fiege73)+0.01, H2O_mean(Fiege73), s=sz_sm-10, marker='>', c='#FFFFFF', ec='#171008', lw=0.5, zorder=20)
ax[0].errorbar(H2O_expmean(Fiege73), H2O_mean(Fiege73), xerr=H2O_expstd(Fiege73), yerr=Error_H2O(Fiege73), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(BF73), H2O_mean(BF73), s=sz-15, marker='D', c='#fec44f', ec='#171008', lw=0.5, zorder=20, label='BF73')
ax[0].errorbar(H2O_expmean(BF73), H2O_mean(BF73), xerr=H2O_expstd(BF73), yerr=Error_H2O(BF73), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(BF76), H2O_mean(BF76), s=sz-15, marker='D', c='#fb9a29', ec='#171008', lw=0.5, zorder=20, label='BF76')
ax[0].errorbar(H2O_expmean(BF76), H2O_mean(BF76), xerr=H2O_expstd(BF76), yerr=Error_H2O(BF76), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(BF77), H2O_mean(BF77), s=sz-15, marker='D', c='#ec7014', ec='#171008', lw=0.5, zorder=20, label='BF77')
ax[0].errorbar(H2O_expmean(BF77), H2O_mean(BF77), xerr=H2O_expstd(BF77), yerr=Error_H2O(BF77), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(NS1), H2O_mean(NS1), s=sz, c='#cc4c02', ec='#171008', lw=0.5, zorder=20, label='NS-1')
ax[0].errorbar(H2O_expmean(NS1), H2O_mean(NS1), xerr=H2O_expstd(NS1), yerr=Error_H2O(NS1), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(M35), H2O_mean(M35), s=sz-15, marker='D', c='#983404', ec='#171008', lw=0.5, zorder=20, label='M35')
ax[0].errorbar(H2O_expmean(M35), H2O_mean(M35), xerr=H2O_expstd(M35), yerr=Error_H2O(M35), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(M43), H2O_mean(M43), s=sz-15, marker='D', c='#662506', ec='#171008', lw=0.5, zorder=20, label='M43')
ax[0].errorbar(H2O_expmean(M43), H2O_mean(M43), xerr=H2O_expstd(M43), yerr=Error_H2O(M43), lw=0.5, c='k', zorder=10)


ax[0].scatter(H2O_expmean(ETFSR_Ol8), H2O_mean(ETFSR_Ol8), s=sz, marker='s', facecolors='white', ec='#FEE391', lw=2.0, zorder=20, label='ETFSR-OL8') 
ax[0].errorbar(H2O_expmean(ETFSR_Ol8), H2O_mean(ETFSR_Ol8), xerr=H2O_expstd(ETFSR_Ol8), yerr=Error_H2O(ETFSR_Ol8), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(CD33_12_2_2), H2O_mean(CD33_12_2_2), s=sz, marker='s', facecolors='white',  ec='#FEC44F', lw=2.0, zorder=20, label='CD33-12-2-2')
ax[0].errorbar(H2O_expmean(CD33_12_2_2), H2O_mean(CD33_12_2_2), xerr=H2O_expstd(CD33_12_2_2), yerr=Error_H2O(CD33_12_2_2), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(CD33_22_1_1), H2O_mean(CD33_22_1_1), s=sz, marker='s',facecolors='white', ec='#FB9A29', lw=2.0, zorder=20, label='CD33-22-1-1')
ax[0].errorbar(H2O_expmean(CD33_22_1_1), H2O_mean(CD33_22_1_1), xerr=H2O_expstd(CD33_22_1_1), yerr=Error_H2O(CD33_22_1_1), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(STD_D1010), H2O_mean(STD_D1010), s=sz, facecolors='white', ec='#EC7014', lw=2.0, zorder=20, label='D1010')
ax[0].errorbar(H2O_expmean(STD_D1010), H2O_mean(STD_D1010), xerr=H2O_expstd(STD_D1010), yerr=Error_H2O(STD_D1010), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ALV1833_11), H2O_mean(ALV1833_11), s=sz, facecolors='white', ec='#CC4C02', lw=2.0, zorder=20, label='ALV1833-11')
ax[0].errorbar(H2O_expmean(ALV1833_11), H2O_mean(ALV1833_11), xerr=H2O_expstd(ALV1833_11), yerr=Error_H2O(ALV1833_11), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(WOK5_4), H2O_mean(WOK5_4), s=sz, facecolors='white', ec='#993404', lw=2.0, zorder=20, label='23WOK5-4')
ax[0].errorbar(H2O_expmean(WOK5_4), H2O_mean(WOK5_4), xerr=H2O_expstd(WOK5_4), yerr=Error_H2O(WOK5_4), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ALV1846), H2O_mean(ALV1846), s=sz, facecolors='white', ec='#662506', lw=2.0, zorder=20, label='21ALV1846-9')
ax[0].errorbar(H2O_expmean(ALV1846), H2O_mean(ALV1846), xerr=H2O_expstd(ALV1846), yerr=Error_H2O(ALV1846), lw=0.5, c='k', zorder=10)


ax[0].scatter(H2O_expmean(INSOL), H2O_mean(INSOL), s=sz+15, marker='o', facecolors='#fff7bc', ec='#171008', lw=0.5, zorder=20, label='INSOL-MX1-BA4') 
ax[0].errorbar(H2O_expmean(INSOL), H2O_mean(INSOL), xerr=H2O_expstd(INSOL), yerr=Error_H2O(INSOL), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ND70_2), H2O_mean(ND70_2), s=sz+15, marker='h', facecolors='#FEE391', ec='#171008', lw=0.5, zorder=20, label='ND70-02') 
ax[0].errorbar(H2O_expmean(ND70_2), H2O_mean(ND70_2), xerr=H2O_expstd(ND70_2), yerr=Error_H2O(ND70_2), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ND70_3), H2O_mean(ND70_3), s=sz+15, marker='h',facecolors='#FB9A29', ec='#171008', lw=0.5, zorder=20, label='ND70-03') 
ax[0].errorbar(H2O_expmean(ND70_3), H2O_mean(ND70_3), xerr=H2O_expstd(ND70_3), yerr=Error_H2O(ND70_3), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ND70_4), H2O_mean(ND70_4), s=sz+15, marker='h', facecolors='#CC4C02', ec='#171008', lw=0.5, zorder=20, label='ND70-04') 
ax[0].errorbar(H2O_expmean(ND70_4), H2O_mean(ND70_4), xerr=H2O_expstd(ND70_4), yerr=Error_H2O(ND70_4), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ND70_5), H2O_mean(ND70_5), s=sz+15, marker='h', facecolors='#993404', ec='#171008', lw=0.5, zorder=20, label='ND70-05') 
ax[0].errorbar(H2O_expmean(ND70_5), H2O_mean(ND70_5), xerr=H2O_expstd(ND70_5), yerr=Error_H2O(ND70_5), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ND70_6), H2O_mean(ND70_6), s=sz+15, marker='h', facecolors='#662506', ec='#171008', lw=0.5, zorder=20, label='ND70-06') 
ax[0].errorbar(H2O_expmean(ND70_6), H2O_mean(ND70_6), xerr=H2O_expstd(ND70_6), yerr=Error_H2O(ND70_6), lw=0.5, c='k', zorder=10)

ax[0].set_xlim([0, 7])
ax[0].set_ylim([0, 7])
ax[0].set_xlabel('$\mathregular{H_2O}$ Expected (wt.%)')
ax[0].set_ylabel('LDEO FTIR $\mathregular{H_2O_t}$ with PyIRoGlass (wt.%)')
l1 = ax[0].legend(loc='lower right', labelspacing=0.3, handletextpad=0.25, handlelength=1.00, prop={'size': 10}, frameon=False)
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

ftir_sym = ax[0].scatter(np.nan, np.nan, s=sz, ec='#171008', facecolors='white', lw=0.5, zorder=20, label='FTIR')
sims_sym = ax[0].scatter(np.nan, np.nan, s=sz, marker='s', ec='#171008', facecolors='white', lw=0.5, zorder=20, label='SIMS')
kft_sym = ax[0].scatter(np.nan, np.nan, s=sz-15, marker='D', ec='#171008', facecolors='white', lw=0.5, zorder=20, label='KFT')
ea_sym = ax[0].scatter(np.nan, np.nan, s=sz, marker='p', ec='#171008', facecolors='white', lw=0.5, zorder=20, label='KFT')
erda_sym = ax[0].scatter(np.nan, np.nan, s=sz, marker='h', ec='#171008', facecolors='white', lw=0.5, zorder=20, label='KFT')
nra_sym = ax[0].scatter(np.nan, np.nan, s=sz, marker='8', ec='#171008', facecolors='white', lw=0.5, zorder=20, label='KFT')
sat_symb = ax[0].scatter(np.nan, np.nan, s=sz_sm, marker='>', ec='#171008', facecolors='white', lw=0.5, zorder=20, label='$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax[0].legend([ftir_sym, sims_sym, kft_sym, ea_sym, erda_sym, nra_sym, sat_symb], ['FTIR', 'SIMS', 'KFT', 'EA', 'ERDA', 'NRA', '$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc=(0.0125, 0.49), labelspacing=0.3, handletextpad=0.25, handlelength=1.00, prop={'size': 10})
ax[0].add_artist(l1)
ax[0].annotate("A.", xy=(0.0175, 0.95), xycoords="axes fraction", fontsize=20, weight='bold')

h2o_exp = np.array([H2O_expmean(Fiege63), H2O_expmean(Fiege73), H2O_expmean(BF73), H2O_expmean(BF76), H2O_expmean(BF77), H2O_expmean(M35), H2O_expmean(M43), H2O_expmean(NS1), H2O_expmean(ETFSR_Ol8), H2O_expmean(CD33_12_2_2), H2O_expmean(CD33_22_1_1), H2O_expmean(STD_D1010), H2O_expmean(ALV1833_11), H2O_expmean(WOK5_4), H2O_expmean(ALV1846), H2O_expmean(ND70_2), H2O_expmean(ND70_3), H2O_expmean(ND70_4), H2O_expmean(ND70_5), H2O_expmean(ND70_6)])
h2o_py = np.array([H2O_mean(Fiege63), H2O_mean(Fiege73), H2O_mean(BF73), H2O_mean(BF76), H2O_mean(BF77), H2O_mean(M35), H2O_mean(M43), H2O_mean(NS1), H2O_mean(ETFSR_Ol8), H2O_mean(CD33_12_2_2), H2O_mean(CD33_22_1_1), H2O_mean(STD_D1010), H2O_mean(ALV1833_11), H2O_mean(WOK5_4), H2O_mean(ALV1846), H2O_mean(ND70_2), H2O_mean(ND70_3), H2O_mean(ND70_4), H2O_mean(ND70_5), H2O_mean(ND70_6)]) 
slope0, intercept0, r_value0, p_value0, std_err0 = scipy.stats.linregress(h2o_exp, h2o_py)
ccc0 = pig.calculate_CCC(h2o_exp, h2o_py)
rmse0 = pig.calculate_RMSE(h2o_exp-h2o_py)
rrmse0 = pig.calculate_RRMSE(h2o_exp, h2o_py)

ax[0].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value0**2, 3)), xy=(0.02, 0.8275), xycoords="axes fraction", fontsize=16)
ax[0].annotate("CCC="+str(np.round(ccc0, 3)), xy=(0.02, 0.91), xycoords="axes fraction", fontsize=16)
ax[0].annotate("RMSE="+str(np.round(rmse0, 3))+"; RRMSE="+str(np.round(rrmse0*100, 3))+'%', xy=(0.02, 0.87), xycoords="axes fraction", fontsize=16)
ax[0].annotate("m="+str(np.round(slope0, 3)), xy=(0.02, 0.79), xycoords="axes fraction", fontsize=16)
ax[0].annotate("b="+str(np.round(intercept0, 3)), xy=(0.02, 0.75), xycoords="axes fraction", fontsize=16)


ax[1].plot(co2_line, co2_line, 'k', lw=0.5, zorder=0)
ax[1].scatter(CO2_expmean(BF73), CO2_mean(BF73)*eps_mean(BF73)/265, s=sz+10, marker='p', c='#fec44f', ec='#171008', lw=0.5, zorder=20, label='BF73')
ax[1].errorbar(CO2_expmean(BF73), CO2_mean(BF73)*eps_mean(BF73)/265, xerr=CO2_expstd(BF73), yerr=Error_CO2(BF73)*eps_mean(BF73)/265, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(BF76), CO2_mean(BF76)*eps_mean(BF76)/265, s=sz+10, marker='p', c='#fb9a29', ec='#171008', lw=0.5, zorder=20, label='BF76')
ax[1].errorbar(CO2_expmean(BF76), CO2_mean(BF76)*eps_mean(BF76)/265, xerr=CO2_expstd(BF76), yerr=Error_CO2(BF76)*eps_mean(BF76)/265, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(BF77), CO2_mean(BF77)*eps_mean(BF77)/265, s=sz+10, marker='p', c='#ec7014', ec='#171008', lw=0.5, zorder=20, label='BF77')
ax[1].errorbar(CO2_expmean(BF77), CO2_mean(BF77)*eps_mean(BF77)/265, xerr=CO2_expstd(BF77), yerr=Error_CO2(BF77)*eps_mean(BF77)/265, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(NS1), CO2_mean(NS1)*eps_mean(NS1)/375, s=sz, marker='s', c='#cc4c02', ec='#171008', lw=0.5, zorder=20, label='NS-1')
ax[1].errorbar(CO2_expmean(NS1), CO2_mean(NS1)*eps_mean(NS1)/375, xerr=CO2_expstd(NS1), yerr=Error_CO2(NS1)*eps_mean(NS1)/375, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(M35), CO2_mean(M35)*eps_mean(M35)/317, s=sz, c='#983404', ec='#171008', lw=0.5, zorder=20, label='M35')
ax[1].errorbar(CO2_expmean(M35), CO2_mean(M35)*eps_mean(M35)/317, xerr=CO2_expstd(M35), yerr=Error_CO2(M35)*eps_mean(M35)/317, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(M43), CO2_mean(M43)*eps_mean(M43)/317, s=sz, c='#662506', ec='#171008', lw=0.5, zorder=20, label='M43')
ax[1].errorbar(CO2_expmean(M43), CO2_mean(M43)*eps_mean(M43)/317, xerr=CO2_expstd(M43), yerr=Error_CO2(M43)*eps_mean(M43)/317, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), s=sz, marker='s', facecolors='white', ec='#FEC44F', lw=2.0, zorder=20, label='CD33-12-2-2')
ax[1].errorbar(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), xerr=CO2_expstd(CD33_12_2_2), yerr=Error_CO2(CD33_12_2_2), lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), s=sz, marker='s', facecolors='white', ec='#FB9A29', lw=2.0, zorder=20, label='CD33-22-1-1')
ax[1].errorbar(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), xerr=CO2_expstd(CD33_22_1_1), yerr=Error_CO2(CD33_22_1_1), lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(STD_D1010), CO2_mean(STD_D1010)*eps_mean(STD_D1010)/375, s=sz, facecolors='white', ec='#EC7014', lw=2.0, zorder=20, label='D1010')
ax[1].errorbar(CO2_expmean(STD_D1010), CO2_mean(STD_D1010)*eps_mean(STD_D1010)/375, xerr=CO2_expstd(STD_D1010), yerr=Error_CO2(STD_D1010)*eps_mean(STD_D1010)/375, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11)*eps_mean(ALV1833_11)/375, s=sz, facecolors='white', ec='#CC4C02', lw=2.0, zorder=20, label='ALV1833-11')
ax[1].errorbar(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11)*eps_mean(ALV1833_11)/375, xerr=CO2_expstd(ALV1833_11), yerr=Error_CO2(ALV1833_11)*eps_mean(ALV1833_11)/375, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(WOK5_4), CO2_mean(WOK5_4)*eps_mean(WOK5_4)/375, s=sz, facecolors='white', ec='#993404', lw=2.0, zorder=20, label='23WOK5-4')
ax[1].errorbar(CO2_expmean(WOK5_4), CO2_mean(WOK5_4)*eps_mean(WOK5_4)/375, xerr=CO2_expstd(WOK5_4), yerr=Error_CO2(WOK5_4)*eps_mean(WOK5_4)/375, lw=0.5, c='k', zorder=10)


ax[1].scatter(CO2_expmean(INSOL), CO2_mean(INSOL), s=sz+15, marker='o', facecolors='#fff7bc', ec='#171008', lw=0.5, zorder=20, label='INSOL-MX1-BA4') 
ax[1].errorbar(CO2_expmean(INSOL), CO2_mean(INSOL), xerr=CO2_expstd(INSOL), yerr=Error_CO2(INSOL), lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(ND70_2), CO2_mean(ND70_2), s=sz+15, marker='8', facecolors='#FEE391', ec='#171008', lw=0.5, zorder=20, label='ND70-02') 
ax[1].errorbar(CO2_expmean(ND70_2), CO2_mean(ND70_2), xerr=CO2_expstd(ND70_2), yerr=Error_CO2(ND70_2), lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(ND70_3), CO2_mean(ND70_3), s=sz+15, marker='8',facecolors='#FB9A29', ec='#171008', lw=0.5, zorder=20, label='ND70-03') 
ax[1].errorbar(CO2_expmean(ND70_3), CO2_mean(ND70_3), xerr=CO2_expstd(ND70_3), yerr=Error_CO2(ND70_3), lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(ND70_4), CO2_mean(ND70_4), s=sz+15, marker='8', facecolors='#CC4C02', ec='#171008', lw=0.5, zorder=20, label='ND70-04') 
ax[1].errorbar(CO2_expmean(ND70_4), CO2_mean(ND70_4), xerr=CO2_expstd(ND70_4), yerr=Error_CO2(ND70_4), lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(ND70_5), CO2_mean(ND70_5), s=sz+15, marker='8', facecolors='#993404', ec='#171008', lw=0.5, zorder=20, label='ND70-05') 
ax[1].errorbar(CO2_expmean(ND70_5), CO2_mean(ND70_5), xerr=CO2_expstd(ND70_5), yerr=Error_CO2(ND70_5), lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(ND70_6), CO2_mean(ND70_6), s=sz+15, marker='8', facecolors='#662506', ec='#171008', lw=0.5, zorder=20, label='ND70-06') 
ax[1].errorbar(CO2_expmean(ND70_6), CO2_mean(ND70_6), xerr=CO2_expstd(ND70_6), yerr=Error_CO2(ND70_6), lw=0.5, c='k', zorder=10)


inset_ax = inset_axes(ax[1], width="25%", height="25%", loc='center left')
inset_ax.plot(co2_line, co2_line, 'k', lw=0.5, zorder=0)
inset_ax.scatter(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), s=sz, marker='s', facecolors='white', ec='#FEC44F', lw=2.0, zorder=20, label='CD33-12-2-2')
inset_ax.errorbar(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), xerr=CO2_expstd(CD33_12_2_2), yerr=Error_CO2(CD33_12_2_2), lw=0.5, c='k', zorder=10)
inset_ax.scatter(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), s=sz, marker='s', facecolors='white', ec='#FB9A29', lw=2.0, zorder=20, label='CD33-22-1-1')
inset_ax.errorbar(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), xerr=CO2_expstd(CD33_22_1_1), yerr=Error_CO2(CD33_22_1_1), lw=0.5, c='k', zorder=10)
inset_ax.scatter(CO2_expmean(STD_D1010), CO2_mean(STD_D1010)*eps_mean(STD_D1010)/375, s=sz, facecolors='white', ec='#EC7014', lw=2.0, zorder=20, label='D1010')
inset_ax.errorbar(CO2_expmean(STD_D1010), CO2_mean(STD_D1010)*eps_mean(STD_D1010)/375, xerr=CO2_expstd(STD_D1010), yerr=Error_CO2(STD_D1010)*eps_mean(STD_D1010)/375, lw=0.5, c='k', zorder=10)
inset_ax.scatter(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11)*eps_mean(ALV1833_11)/375, s=sz, facecolors='white', ec='#CC4C02', lw=2.0, zorder=20, label='ALV1833-11')
inset_ax.errorbar(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11)*eps_mean(ALV1833_11)/375, xerr=CO2_expstd(ALV1833_11), yerr=Error_CO2(ALV1833_11)*eps_mean(ALV1833_11)/375, lw=0.5, c='k', zorder=10)
inset_ax.scatter(CO2_expmean(WOK5_4), CO2_mean(WOK5_4)*eps_mean(WOK5_4)/375, s=sz, facecolors='white', ec='#993404', lw=2.0, zorder=20, label='23WOK5-4')
inset_ax.errorbar(CO2_expmean(WOK5_4), CO2_mean(WOK5_4)*eps_mean(WOK5_4)/375, xerr=CO2_expstd(WOK5_4), yerr=Error_CO2(WOK5_4)*eps_mean(WOK5_4)/375, lw=0.5, c='k', zorder=10)
inset_ax.set_xlim([0, 0.02])
inset_ax.set_ylim([0, 0.02])
inset_ax.tick_params(axis='both', which='major', labelsize=12)
inset_ax.xaxis.set_major_locator(ticker.MultipleLocator(0.01))
inset_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
inset_ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
inset_ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)

ax[1].set_xlim([0, 1.8])
ax[1].set_ylim([0, 1.8])
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax[1].set_xlabel('$\mathregular{CO_2}$ Expected (wt.%)')
ax[1].set_ylabel('LDEO FTIR $\mathregular{CO_2}$ with PyIRoGlass (wt.%)')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].legend(loc='lower right', labelspacing=0.3, handletextpad=0.25, handlelength=1.00, prop={'size': 10}, frameon=False)
ax[1].annotate("B.", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='bold')

co2_exp = np.array([CO2_expmean(BF73), CO2_expmean(BF76), CO2_expmean(BF77), CO2_expmean(M35), CO2_expmean(M43), CO2_expmean(NS1), CO2_expmean(CD33_12_2_2), CO2_expmean(CD33_22_1_1), CO2_expmean(STD_D1010), CO2_expmean(ALV1833_11), CO2_expmean(WOK5_4), CO2_expmean(ND70_2), CO2_expmean(ND70_3), CO2_expmean(ND70_4), CO2_expmean(ND70_5), CO2_expmean(ND70_6)])
co2_py = np.array([CO2_mean(BF73)*eps_mean(BF73)/265, CO2_mean(BF76)*eps_mean(BF73)/265, CO2_mean(BF77)*eps_mean(BF73)/265, CO2_mean(M35)*eps_mean(M35)/317, CO2_mean(M43)*eps_mean(M43)/317, CO2_mean(NS1)*eps_mean(NS1)/375, CO2_mean(CD33_12_2_2), CO2_mean(CD33_22_1_1), CO2_mean(STD_D1010)*eps_mean(STD_D1010)/375, CO2_mean(ALV1833_11)*eps_mean(ALV1833_11)/375, CO2_mean(WOK5_4)*eps_mean(WOK5_4)/375, CO2_mean(ND70_2), CO2_mean(ND70_3), CO2_mean(ND70_4), CO2_mean(ND70_5), CO2_mean(ND70_6)])
slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(co2_exp, co2_py)
ccc1 = pig.calculate_CCC(co2_exp, co2_py)
rmse1 = pig.calculate_RMSE(co2_exp-co2_py)
rrmse1 = pig.calculate_RRMSE(co2_exp, co2_py)

ax[1].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value1**2, 3)), xy=(0.02, 0.8275), xycoords="axes fraction", fontsize=16)
ax[1].annotate("CCC="+str(np.round(ccc1, 3)), xy=(0.02, 0.91), xycoords="axes fraction", fontsize=16)
ax[1].annotate("RMSE="+str(np.round(rmse1, 3))+"; RRMSE="+str(np.round(rrmse1*100, 3))+'%', xy=(0.02, 0.87), xycoords="axes fraction", fontsize=16)
ax[1].annotate("m="+str(np.round(slope1, 3)), xy=(0.02, 0.79), xycoords="axes fraction", fontsize=16)
ax[1].annotate("b="+str(np.round(intercept1, 3)), xy=(0.02, 0.75), xycoords="axes fraction", fontsize=16)

plt.tight_layout()
# plt.savefig('FTIRSIMS_Comparison_ND70.pdf', bbox_inches='tight', pad_inches=0.025)
plt.show()

# %% 


h2o_line = np.array([0, 7])
co2_line = np.array([0, 3])
sz_sm = 50
sz = 90

fig, ax = plt.subplots(1, 2, figsize = (14, 7))

ax = ax.flatten()
ax[0].plot(h2o_line, h2o_line, 'k', lw=0.5, zorder=0)
ax[0].scatter(H2O_expmean(Fiege63), H2O_mean(Fiege63), s=sz, c='#fff7bc', ec='#171008', lw=0.5, zorder=15, label='ABWCl-F0x')
ax[0].scatter(H2O_expmean(Fiege63)+0.01, H2O_mean(Fiege63), s=sz_sm-10, marker='>', c='#FFFFFF', ec='#171008', lw=0.5, zorder=20)
ax[0].errorbar(H2O_expmean(Fiege63), H2O_mean(Fiege63), xerr=H2O_expstd(Fiege63), yerr=H2O_mean(Fiege63)*H2O_rsd(Fiege63), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(Fiege73), H2O_mean(Fiege73), s=sz-15, marker='D', c='#fee392', ec='#171008', lw=0.5, zorder=15, label='ABWB-0x')
ax[0].scatter(H2O_expmean(Fiege73)+0.01, H2O_mean(Fiege73), s=sz_sm-10, marker='>', c='#FFFFFF', ec='#171008', lw=0.5, zorder=20)
ax[0].errorbar(H2O_expmean(Fiege73), H2O_mean(Fiege73), xerr=H2O_expstd(Fiege73), yerr=H2O_mean(Fiege73)*H2O_rsd(Fiege73), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(BF73), H2O_mean(BF73), s=sz-15, marker='D', c='#fec44f', ec='#171008', lw=0.5, zorder=20, label='BF73')
ax[0].errorbar(H2O_expmean(BF73), H2O_mean(BF73), xerr=H2O_expstd(BF73), yerr=H2O_mean(BF73)*H2O_rsd(BF73), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(BF76), H2O_mean(BF76), s=sz-15, marker='D', c='#fb9a29', ec='#171008', lw=0.5, zorder=20, label='BF76')
ax[0].errorbar(H2O_expmean(BF76), H2O_mean(BF76), xerr=H2O_expstd(BF76), yerr=H2O_mean(BF76)*H2O_rsd(BF76), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(BF77), H2O_mean(BF77), s=sz-15, marker='D', c='#ec7014', ec='#171008', lw=0.5, zorder=20, label='BF77')
ax[0].errorbar(H2O_expmean(BF77), H2O_mean(BF77), xerr=H2O_expstd(BF77), yerr=H2O_mean(BF77)*H2O_rsd(BF77), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(NS1), H2O_mean(NS1), s=sz, c='#cc4c02', ec='#171008', lw=0.5, zorder=20, label='NS-1')
ax[0].errorbar(H2O_expmean(NS1), H2O_mean(NS1), xerr=H2O_expstd(NS1), yerr=H2O_mean(NS1)*H2O_rsd(NS1), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(M35), H2O_mean(M35), s=sz-15, marker='D', c='#983404', ec='#171008', lw=0.5, zorder=20, label='M35')
ax[0].errorbar(H2O_expmean(M35), H2O_mean(M35), xerr=H2O_expstd(M35), yerr=H2O_mean(M35)*H2O_rsd(M35), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(M43), H2O_mean(M43), s=sz-15, marker='D', c='#662506', ec='#171008', lw=0.5, zorder=20, label='M43')
ax[0].errorbar(H2O_expmean(M43), H2O_mean(M43), xerr=H2O_expstd(M43), yerr=H2O_mean(M43)*H2O_rsd(M43), lw=0.5, c='k', zorder=10)


ax[0].scatter(H2O_expmean(ETFSR_Ol8), H2O_mean(ETFSR_Ol8), s=sz, marker='s', facecolors='white', ec='#FEE391', lw=2.0, zorder=20, label='ETFSR-OL8') 
ax[0].errorbar(H2O_expmean(ETFSR_Ol8), H2O_mean(ETFSR_Ol8), xerr=H2O_expstd(ETFSR_Ol8), yerr=H2O_mean(ETFSR_Ol8)*H2O_rsd(ETFSR_Ol8), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(CD33_12_2_2), H2O_mean(CD33_12_2_2), s=sz, marker='s', facecolors='white',  ec='#FEC44F', lw=2.0, zorder=20, label='CD33-12-2-2')
ax[0].errorbar(H2O_expmean(CD33_12_2_2), H2O_mean(CD33_12_2_2), xerr=H2O_expstd(CD33_12_2_2), yerr=H2O_mean(CD33_12_2_2)*H2O_rsd(CD33_12_2_2), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(CD33_22_1_1), H2O_mean(CD33_22_1_1), s=sz, marker='s',facecolors='white', ec='#FB9A29', lw=2.0, zorder=20, label='CD33-22-1-1')
ax[0].errorbar(H2O_expmean(CD33_22_1_1), H2O_mean(CD33_22_1_1), xerr=H2O_expstd(CD33_22_1_1), yerr=H2O_mean(CD33_22_1_1)*H2O_rsd(CD33_22_1_1), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(STD_D1010), H2O_mean(STD_D1010), s=sz, facecolors='white', ec='#EC7014', lw=2.0, zorder=20, label='D1010')
ax[0].errorbar(H2O_expmean(STD_D1010), H2O_mean(STD_D1010), xerr=H2O_expstd(STD_D1010), yerr=H2O_mean(STD_D1010)*H2O_rsd(STD_D1010), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ALV1833_11), H2O_mean(ALV1833_11), s=sz, facecolors='white', ec='#CC4C02', lw=2.0, zorder=20, label='ALV1833-11')
ax[0].errorbar(H2O_expmean(ALV1833_11), H2O_mean(ALV1833_11), xerr=H2O_expstd(ALV1833_11), yerr=H2O_mean(ALV1833_11)*H2O_rsd(ALV1833_11), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(WOK5_4), H2O_mean(WOK5_4), s=sz, facecolors='white', ec='#993404', lw=2.0, zorder=20, label='23WOK5-4')
ax[0].errorbar(H2O_expmean(WOK5_4), H2O_mean(WOK5_4), xerr=H2O_expstd(WOK5_4), yerr=H2O_mean(WOK5_4)*H2O_rsd(WOK5_4), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ALV1846), H2O_mean(ALV1846), s=sz, facecolors='white', ec='#662506', lw=2.0, zorder=20, label='21ALV1846-9')
ax[0].errorbar(H2O_expmean(ALV1846), H2O_mean(ALV1846), xerr=H2O_expstd(ALV1846), yerr=H2O_mean(ALV1846)*H2O_rsd(ALV1846), lw=0.5, c='k', zorder=10)


ax[0].scatter(H2O_expmean(INSOL), H2O_mean(INSOL), s=sz+15, marker='o', facecolors='#fff7bc', ec='#171008', lw=0.5, zorder=20, label='INSOL-MX1-BA4') 
ax[0].errorbar(H2O_expmean(INSOL), H2O_mean(INSOL), xerr=H2O_expstd(INSOL), yerr=H2O_mean(INSOL)*H2O_rsd(INSOL), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ND70_2), H2O_mean(ND70_2), s=sz+15, marker='h', facecolors='#FEE391', ec='#171008', lw=0.5, zorder=20, label='ND70-02') 
ax[0].errorbar(H2O_expmean(ND70_2), H2O_mean(ND70_2), xerr=H2O_expstd(ND70_2), yerr=H2O_mean(ND70_2)*H2O_rsd(ND70_2), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ND70_3), H2O_mean(ND70_3), s=sz+15, marker='h',facecolors='#FB9A29', ec='#171008', lw=0.5, zorder=20, label='ND70-03') 
ax[0].errorbar(H2O_expmean(ND70_3), H2O_mean(ND70_3), xerr=H2O_expstd(ND70_3), yerr=H2O_mean(ND70_3)*H2O_rsd(ND70_3), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ND70_4), H2O_mean(ND70_4), s=sz+15, marker='h', facecolors='#CC4C02', ec='#171008', lw=0.5, zorder=20, label='ND70-04') 
ax[0].errorbar(H2O_expmean(ND70_4), H2O_mean(ND70_4), xerr=H2O_expstd(ND70_4), yerr=H2O_mean(ND70_4)*H2O_rsd(ND70_4), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ND70_5), H2O_mean(ND70_5), s=sz+15, marker='h', facecolors='#993404', ec='#171008', lw=0.5, zorder=20, label='ND70-05') 
ax[0].errorbar(H2O_expmean(ND70_5), H2O_mean(ND70_5), xerr=H2O_expstd(ND70_5), yerr=H2O_mean(ND70_5)*H2O_rsd(ND70_5), lw=0.5, c='k', zorder=10)

ax[0].scatter(H2O_expmean(ND70_6), H2O_mean(ND70_6), s=sz+15, marker='h', facecolors='#662506', ec='#171008', lw=0.5, zorder=20, label='ND70-06') 
ax[0].errorbar(H2O_expmean(ND70_6), H2O_mean(ND70_6), xerr=H2O_expstd(ND70_6), yerr=H2O_mean(ND70_6)*H2O_rsd(ND70_6), lw=0.5, c='k', zorder=10)

ax[0].set_xlim([0, 7])
ax[0].set_ylim([0, 7])
ax[0].set_xlabel('$\mathregular{H_2O}$ Expected (wt.%)')
ax[0].set_ylabel('LDEO FTIR $\mathregular{H_2O_t}$ with PyIRoGlass (wt.%)')
l1 = ax[0].legend(loc='lower right', labelspacing=0.3, handletextpad=0.25, handlelength=1.00, prop={'size': 10}, frameon=False)
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

ftir_sym = ax[0].scatter(np.nan, np.nan, s=sz, ec='#171008', facecolors='white', lw=0.5, zorder=20, label='FTIR')
sims_sym = ax[0].scatter(np.nan, np.nan, s=sz, marker='s', ec='#171008', facecolors='white', lw=0.5, zorder=20, label='SIMS')
kft_sym = ax[0].scatter(np.nan, np.nan, s=sz-15, marker='D', ec='#171008', facecolors='white', lw=0.5, zorder=20, label='KFT')
ea_sym = ax[0].scatter(np.nan, np.nan, s=sz, marker='p', ec='#171008', facecolors='white', lw=0.5, zorder=20, label='KFT')
erda_sym = ax[0].scatter(np.nan, np.nan, s=sz, marker='h', ec='#171008', facecolors='white', lw=0.5, zorder=20, label='KFT')
nra_sym = ax[0].scatter(np.nan, np.nan, s=sz, marker='8', ec='#171008', facecolors='white', lw=0.5, zorder=20, label='KFT')
sat_symb = ax[0].scatter(np.nan, np.nan, s=sz_sm, marker='>', ec='#171008', facecolors='white', lw=0.5, zorder=20, label='$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax[0].legend([ftir_sym, sims_sym, kft_sym, ea_sym, erda_sym, nra_sym, sat_symb], ['FTIR', 'SIMS', 'KFT', 'EA', 'ERDA', 'NRA', '$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc=(0.0125, 0.49), labelspacing=0.3, handletextpad=0.25, handlelength=1.00, prop={'size': 10})
ax[0].add_artist(l1)
ax[0].annotate("A.", xy=(0.0175, 0.95), xycoords="axes fraction", fontsize=20, weight='bold')

h2o_exp = np.array([H2O_expmean(Fiege63), H2O_expmean(Fiege73), H2O_expmean(BF73), H2O_expmean(BF76), H2O_expmean(BF77), H2O_expmean(M35), H2O_expmean(M43), H2O_expmean(NS1), H2O_expmean(ETFSR_Ol8), H2O_expmean(CD33_12_2_2), H2O_expmean(CD33_22_1_1), H2O_expmean(STD_D1010), H2O_expmean(ALV1833_11), H2O_expmean(WOK5_4), H2O_expmean(ALV1846), H2O_expmean(ND70_2), H2O_expmean(ND70_3), H2O_expmean(ND70_4), H2O_expmean(ND70_5), H2O_expmean(ND70_6)])
h2o_py = np.array([H2O_mean(Fiege63), H2O_mean(Fiege73), H2O_mean(BF73), H2O_mean(BF76), H2O_mean(BF77), H2O_mean(M35), H2O_mean(M43), H2O_mean(NS1), H2O_mean(ETFSR_Ol8), H2O_mean(CD33_12_2_2), H2O_mean(CD33_22_1_1), H2O_mean(STD_D1010), H2O_mean(ALV1833_11), H2O_mean(WOK5_4), H2O_mean(ALV1846), H2O_mean(ND70_2), H2O_mean(ND70_3), H2O_mean(ND70_4), H2O_mean(ND70_5), H2O_mean(ND70_6)]) 
slope0, intercept0, r_value0, p_value0, std_err0 = scipy.stats.linregress(h2o_exp, h2o_py)
ccc0 = pig.calculate_CCC(h2o_exp, h2o_py)
rmse0 = pig.calculate_RMSE(h2o_exp-h2o_py)
rrmse0 = pig.calculate_RRMSE(h2o_exp, h2o_py)

ax[0].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value0**2, 3)), xy=(0.02, 0.8275), xycoords="axes fraction", fontsize=16)
ax[0].annotate("CCC="+str(np.round(ccc0, 3)), xy=(0.02, 0.91), xycoords="axes fraction", fontsize=16)
ax[0].annotate("RMSE="+str(np.round(rmse0, 3))+"; RRMSE="+str(np.round(rrmse0*100, 3))+'%', xy=(0.02, 0.87), xycoords="axes fraction", fontsize=16)
ax[0].annotate("m="+str(np.round(slope0, 3)), xy=(0.02, 0.79), xycoords="axes fraction", fontsize=16)
ax[0].annotate("b="+str(np.round(intercept0, 3)), xy=(0.02, 0.75), xycoords="axes fraction", fontsize=16)


ax[1].plot(co2_line, co2_line, 'k', lw=0.5, zorder=0)
ax[1].scatter(CO2_expmean(BF73), CO2_mean(BF73)*338.6880953/265, s=sz+10, marker='p', c='#fec44f', ec='#171008', lw=0.5, zorder=20, label='BF73')
ax[1].errorbar(CO2_expmean(BF73), CO2_mean(BF73)*338.6880953/265, xerr=CO2_expstd(BF73)*2, yerr=CO2_mean(BF73)*CO2_rsd(BF73)*2*338.6880953/265, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(BF76), CO2_mean(BF76)*338.6880953/265, s=sz+10, marker='p', c='#fb9a29', ec='#171008', lw=0.5, zorder=20, label='BF76')
ax[1].errorbar(CO2_expmean(BF76), CO2_mean(BF76)*338.6880953/265, xerr=CO2_expstd(BF76)*2, yerr=CO2_mean(BF76)*CO2_rsd(BF76)*2*338.6880953/265, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(BF77), CO2_mean(BF77)*338.6880953/265, s=sz+10, marker='p', c='#ec7014', ec='#171008', lw=0.5, zorder=20, label='BF77')
ax[1].errorbar(CO2_expmean(BF77), CO2_mean(BF77)*338.6880953/265, xerr=CO2_expstd(BF77)*2, yerr=CO2_mean(BF77)*CO2_rsd(BF77)*2*338.6880953/265, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(NS1), CO2_mean(NS1)*315.2724799/375, s=sz, marker='s', c='#cc4c02', ec='#171008', lw=0.5, zorder=20, label='NS-1')
ax[1].errorbar(CO2_expmean(NS1), CO2_mean(NS1)*315.2724799/375, xerr=CO2_expstd(NS1)*2, yerr=CO2_mean(NS1)*CO2_rsd(NS1)*2/315.2724799*375, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(M35), CO2_mean(M35)*329.7316656/317, s=sz, c='#983404', ec='#171008', lw=0.5, zorder=20, label='M35')
ax[1].errorbar(CO2_expmean(M35), CO2_mean(M35)*329.7316656/317, xerr=CO2_expstd(M35)*2, yerr=CO2_mean(M35)*CO2_rsd(M35)* 2*329.7316656/317, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(M43), CO2_mean(M43)*336.1936113/317, s=sz, c='#662506', ec='#171008', lw=0.5, zorder=20, label='M43')
ax[1].errorbar(CO2_expmean(M43), CO2_mean(M43)*336.1936113/317, xerr=CO2_expstd(M43)*2, yerr=CO2_mean(M43)*CO2_rsd(M43)*2*336.1936113/317, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), s=sz, marker='s', facecolors='white', ec='#FEC44F', lw=2.0, zorder=20, label='CD33-12-2-2')
ax[1].errorbar(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), xerr=CO2_expstd(CD33_12_2_2)*2, yerr=CO2_mean(CD33_12_2_2)*CO2_rsd(CD33_12_2_2)*2, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), s=sz, marker='s', facecolors='white', ec='#FB9A29', lw=2.0, zorder=20, label='CD33-22-1-1')
ax[1].errorbar(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), xerr=CO2_expstd(CD33_22_1_1)*2, yerr=CO2_mean(CD33_22_1_1)*CO2_rsd(CD33_22_1_1)*2, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(STD_D1010), CO2_mean(STD_D1010)*315.7646212/375, s=sz, facecolors='white', ec='#EC7014', lw=2.0, zorder=20, label='D1010')
ax[1].errorbar(CO2_expmean(STD_D1010), CO2_mean(STD_D1010)*315.7646212/375, xerr=CO2_expstd(STD_D1010)*2, yerr=CO2_mean(STD_D1010)*CO2_rsd(STD_D1010)*2*315.7646212/375, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11)*353.8503071/375, s=sz, facecolors='white', ec='#CC4C02', lw=2.0, zorder=20, label='ALV1833-11')
ax[1].errorbar(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11)*353.8503071/375, xerr=CO2_expstd(ALV1833_11)*2, yerr=CO2_mean(ALV1833_11)*CO2_rsd(ALV1833_11)*2*353.8503071/375, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(WOK5_4), CO2_mean(WOK5_4)*310.7007261/375, s=sz, facecolors='white', ec='#993404', lw=2.0, zorder=20, label='23WOK5-4')
ax[1].errorbar(CO2_expmean(WOK5_4), CO2_mean(WOK5_4)*310.7007261/375, xerr=CO2_expstd(WOK5_4)*2, yerr=CO2_mean(WOK5_4)*CO2_rsd(WOK5_4)*2*310.7007261/375, lw=0.5, c='k', zorder=10)


ax[1].scatter(CO2_expmean(INSOL), CO2_mean(INSOL), s=sz+15, marker='o', facecolors='#fff7bc', ec='#171008', lw=0.5, zorder=20, label='INSOL-MX1-BA4') 
ax[1].errorbar(CO2_expmean(INSOL), CO2_mean(INSOL), xerr=CO2_expstd(INSOL)*2, yerr=CO2_mean(INSOL)*CO2_rsd(INSOL)*2, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(ND70_2), CO2_mean(ND70_2), s=sz+15, marker='8', facecolors='#FEE391', ec='#171008', lw=0.5, zorder=20, label='ND70-02') 
ax[1].errorbar(CO2_expmean(ND70_2), CO2_mean(ND70_2), xerr=CO2_expstd(ND70_2)*2, yerr=CO2_mean(ND70_2)*CO2_rsd(ND70_2)*2, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(ND70_3), CO2_mean(ND70_3), s=sz+15, marker='8',facecolors='#FB9A29', ec='#171008', lw=0.5, zorder=20, label='ND70-03') 
ax[1].errorbar(CO2_expmean(ND70_3), CO2_mean(ND70_3), xerr=CO2_expstd(ND70_3)*2, yerr=CO2_mean(ND70_3)*CO2_rsd(ND70_3)*2, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(ND70_4), CO2_mean(ND70_4), s=sz+15, marker='8', facecolors='#CC4C02', ec='#171008', lw=0.5, zorder=20, label='ND70-04') 
ax[1].errorbar(CO2_expmean(ND70_4), CO2_mean(ND70_4), xerr=CO2_expstd(ND70_4)*2, yerr=CO2_mean(ND70_4)*CO2_rsd(ND70_4)*2, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(ND70_5), CO2_mean(ND70_5), s=sz+15, marker='8', facecolors='#993404', ec='#171008', lw=0.5, zorder=20, label='ND70-05') 
ax[1].errorbar(CO2_expmean(ND70_5), CO2_mean(ND70_5), xerr=CO2_expstd(ND70_5)*2, yerr=CO2_mean(ND70_5)*CO2_rsd(ND70_5)*2, lw=0.5, c='k', zorder=10)

ax[1].scatter(CO2_expmean(ND70_6), CO2_mean(ND70_6), s=sz+15, marker='8', facecolors='#662506', ec='#171008', lw=0.5, zorder=20, label='ND70-06') 
ax[1].errorbar(CO2_expmean(ND70_6), CO2_mean(ND70_6), xerr=CO2_expstd(ND70_6)*2, yerr=CO2_mean(ND70_6)*CO2_rsd(ND70_6)*2, lw=0.5, c='k', zorder=10)


inset_ax = inset_axes(ax[1], width="25%", height="25%", loc='center left')
inset_ax.plot(co2_line, co2_line, 'k', lw=0.5, zorder=0)
inset_ax.scatter(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), s=sz, marker='s', facecolors='white', ec='#FEC44F', lw=2.0, zorder=20, label='CD33-12-2-2')
inset_ax.errorbar(CO2_expmean(CD33_12_2_2), CO2_mean(CD33_12_2_2), xerr=CO2_expstd(CD33_12_2_2)*2, yerr=CO2_mean(CD33_12_2_2)*CO2_rsd(CD33_12_2_2)*2, lw=0.5, c='k', zorder=10)
inset_ax.scatter(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), s=sz, marker='s', facecolors='white', ec='#FB9A29', lw=2.0, zorder=20, label='CD33-22-1-1')
inset_ax.errorbar(CO2_expmean(CD33_22_1_1), CO2_mean(CD33_22_1_1), xerr=CO2_expstd(CD33_22_1_1)*2, yerr=CO2_mean(CD33_22_1_1)*CO2_rsd(CD33_22_1_1)*2, lw=0.5, c='k', zorder=10)
inset_ax.scatter(CO2_expmean(STD_D1010), CO2_mean(STD_D1010)*315.7646212/375, s=sz, facecolors='white', ec='#EC7014', lw=2.0, zorder=20, label='D1010')
inset_ax.errorbar(CO2_expmean(STD_D1010), CO2_mean(STD_D1010)*315.7646212/375, xerr=CO2_expstd(STD_D1010)*2, yerr=CO2_mean(STD_D1010)*CO2_rsd(STD_D1010)*2*315.7646212/375, lw=0.5, c='k', zorder=10)
inset_ax.scatter(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11)*353.8503071/375, s=sz, facecolors='white', ec='#CC4C02', lw=2.0, zorder=20, label='ALV1833-11')
inset_ax.errorbar(CO2_expmean(ALV1833_11), CO2_mean(ALV1833_11)*353.8503071/375, xerr=CO2_expstd(ALV1833_11)*2, yerr=CO2_mean(ALV1833_11)*CO2_rsd(ALV1833_11)*2*353.8503071/375, lw=0.5, c='k', zorder=10)
inset_ax.scatter(CO2_expmean(WOK5_4), CO2_mean(WOK5_4)*310.7007261/375, s=sz, facecolors='white', ec='#993404', lw=2.0, zorder=20, label='23WOK5-4')
inset_ax.errorbar(CO2_expmean(WOK5_4), CO2_mean(WOK5_4)*310.7007261/375, xerr=CO2_expstd(WOK5_4)*2, yerr=CO2_mean(WOK5_4)*CO2_rsd(WOK5_4)*2*310.7007261/375, lw=0.5, c='k', zorder=10)
inset_ax.set_xlim([0, 0.02])
inset_ax.set_ylim([0, 0.02])
inset_ax.tick_params(axis='both', which='major', labelsize=12)
inset_ax.xaxis.set_major_locator(ticker.MultipleLocator(0.01))
inset_ax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
inset_ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
inset_ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)

ax[1].set_xlim([0, 1.8])
ax[1].set_ylim([0, 1.8])
ax[1].xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax[1].yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax[1].set_xlabel('$\mathregular{CO_2}$ Expected (wt.%)')
ax[1].set_ylabel('LDEO FTIR $\mathregular{CO_2}$ with PyIRoGlass (wt.%)')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].legend(loc='lower right', labelspacing=0.3, handletextpad=0.25, handlelength=1.00, prop={'size': 10}, frameon=False)
ax[1].annotate("B.", xy=(0.02, 0.95), xycoords="axes fraction", fontsize=20, weight='bold')

co2_exp = np.array([CO2_expmean(BF73), CO2_expmean(BF76), CO2_expmean(BF77), CO2_expmean(M35), CO2_expmean(M43), CO2_expmean(NS1), CO2_expmean(CD33_12_2_2), CO2_expmean(CD33_22_1_1), CO2_expmean(STD_D1010), CO2_expmean(ALV1833_11), CO2_expmean(WOK5_4), CO2_expmean(ND70_2), CO2_expmean(ND70_3), CO2_expmean(ND70_4), CO2_expmean(ND70_5), CO2_expmean(ND70_6)])
co2_py = np.array([CO2_mean(BF73)*338.6880953/265, CO2_mean(BF76)*338.6880953/265, CO2_mean(BF77)*338.6880953/265, CO2_mean(M35)*329.7316656/317, CO2_mean(M43)*336.1936113/317, CO2_mean(NS1)*315.2724799/375, CO2_mean(CD33_12_2_2), CO2_mean(CD33_22_1_1), CO2_mean(STD_D1010)*315.7646212/375, CO2_mean(ALV1833_11)*353.8503071/375, CO2_mean(WOK5_4)*310.7007261/375, CO2_mean(ND70_2), CO2_mean(ND70_3), CO2_mean(ND70_4), CO2_mean(ND70_5), CO2_mean(ND70_6)])
slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(co2_exp, co2_py)
ccc1 = pig.calculate_CCC(co2_exp, co2_py)
rmse1 = pig.calculate_RMSE(co2_exp-co2_py)
rrmse1 = pig.calculate_RRMSE(co2_exp, co2_py)

ax[1].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value1**2, 3)), xy=(0.02, 0.8275), xycoords="axes fraction", fontsize=16)
ax[1].annotate("CCC="+str(np.round(ccc1, 3)), xy=(0.02, 0.91), xycoords="axes fraction", fontsize=16)
ax[1].annotate("RMSE="+str(np.round(rmse1, 3))+"; RRMSE="+str(np.round(rrmse1*100, 3))+'%', xy=(0.02, 0.87), xycoords="axes fraction", fontsize=16)
ax[1].annotate("m="+str(np.round(slope1, 3)), xy=(0.02, 0.79), xycoords="axes fraction", fontsize=16)
ax[1].annotate("b="+str(np.round(intercept1, 3)), xy=(0.02, 0.75), xycoords="axes fraction", fontsize=16)

plt.tight_layout()
# plt.savefig('FTIRSIMS_Comparison_ND70_new1.pdf', bbox_inches='tight', pad_inches=0.025)
plt.show()

# %%
# %%

c_si_si_calib = np.array([0.00352726, 0.00364888, 0.39991483, 0.35405633, 0.25057402, 0.25527224, 0.77261763, 0.69592738, 0.01366744, 0.01466238, 15.21391320, 15.75483819])
co2_calib = np.array([0, 0, 165, 165, 90, 90, 243, 243, 0, 0, 7754, 7754])

bf73_csisi = np.array([4.98502480, 4.83861929])
bf76_csisi = np.array([5.33802805, 4.88439129])
bf77_csisi = np.array([1.77990001, 1.66122648])
ns1_csisi = np.array([10.30641893, 9.29042104, 9.59020493])

bf73_co2 = np.array([2321, 2321])
bf76_co2 = np.array([1769, 1769])
bf77_co2 = np.array([679, 679])
ns1_co2 = np.array([4125, 4125, 4125])

arr = np.array([0, 10, 17])
slope0, intercept0, r_value0, p_value0, std_err0 = scipy.stats.linregress(c_si_si_calib, co2_calib)
line = slope0*arr+intercept0

sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot(arr, line, 'k--')
ax.scatter(c_si_si_calib, co2_calib, s=sz-25, c='#9D9D9D', edgecolors='black', linewidth = 0.5, zorder=15, label='Calibration Standards')
ax.scatter(bf73_csisi, bf73_co2, marker='s', s=sz, c='#F4A582', edgecolors='black', linewidth = 0.5, zorder=15, label='BF73')
ax.scatter(bf76_csisi, bf76_co2, marker='s', s=sz, c='#FDDBC7', edgecolors='black', linewidth = 0.5, zorder=15, label='BF76')
ax.scatter(bf77_csisi, bf77_co2, marker='s', s=sz, c='#F7F7F7', edgecolors='black', linewidth = 0.5, zorder=15, label='BF77')
ax.scatter(ns1_csisi, ns1_co2, marker='s', s=sz, c='#4393C3', edgecolors='black', linewidth = 0.5, zorder=15, label='NS-1')
ax.annotate('y=502.8225x-38.1266', xy=(0.55, 0.8), xycoords="axes fraction", horizontalalignment='center', fontsize=16)
ax.annotate('$\mathregular{R^2}$=0.9994', xy=(0.55, 0.765), xycoords="axes fraction", horizontalalignment='center', fontsize=16)
ax.set_xlabel(r'$\mathregular{^{12}C/^{30}Si \cdot SiO_2}$')
ax.set_ylabel('$\mathregular{CO_2}$ (ppm)')
ax.legend(labelspacing=0.3, handletextpad=0.25, handlelength=1.00, prop={'size': 13}, frameon=False)
ax.set_xlim([-0.5, 18])
ax.set_ylim([-250, 8250])
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('SIMS_Calibration.pdf', bbox_inches='tight', pad_inches=0.025)

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

names=np.array(['D1010', 'C1', 'CN92C_OL2', 'VF74_127_7', 'VF74_132_2',
    'Fiege63', 'ETFSR_Ol8', 'Fiege73', 'CD33_12_2_2', 'CD33_22_1_1', 'ALV1833_11', 'WOK5_4', 'ALV1846'])

h2o_vmin, h2o_vmax = min(H2O_stdexpmean), max(H2O_stdmean)

fig, ax = plt.subplots(1, 2, figsize = (18, 8))
ax = ax.flatten()
sc1 = ax[0].plot(h2o_line, h2o_line, 'k', lw=0.5, zorder=0)

for i in range(len(names)): 
    if names[i] in ('C1', 'CN92C_OL2', 'VF74_127_7', 'VF74_132_2', 'ETFSR_Ol8'):
        scatter1 = ax[0].scatter(H2O_stdexpmean[i], H2O_stdmean[i], s=sz, marker='s', c=H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec='#171008', lw=0.5, zorder=20)
        ax[0].errorbar(H2O_stdexpmean[i], H2O_stdmean[i], marker='s', xerr=H2O_stdexpstd[i], yerr=H2O_stdmean[i] * H2O_stdrsd[i], lw=0.5, ls='none', c='k', zorder=10)
        ax[1].scatter(CO2_stdexpmean[i], CO2_stdmean[i], s=sz, marker='s', c=H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec='#171008', lw=0.5, zorder=20)
        ax[1].errorbar(CO2_stdexpmean[i], CO2_stdmean[i], xerr=CO2_stdexpstd[i], yerr=CO2_stdmean[i] * CO2_stdrsd[i], lw=0.5, ls='none', c='k', zorder=10)
    elif names[i] in ('Fiege73'):
        scatter1 = ax[0].scatter(H2O_stdexpmean[i], H2O_stdmean[i], s=sz, marker='D', c=H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec='#171008', lw=0.5, zorder=20)
        ax[0].errorbar(H2O_stdexpmean[i], H2O_stdmean[i], marker='s', xerr=H2O_stdexpstd[i], yerr=H2O_stdmean[i] * H2O_stdrsd[i], lw=0.5, ls='none', c='k', zorder=10)
        ax[1].scatter(CO2_stdexpmean[i], CO2_stdmean[i], s=sz, marker='s', c=H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec='#171008', lw=0.5, zorder=20)
        ax[1].errorbar(CO2_stdexpmean[i], CO2_stdmean[i], xerr=CO2_stdexpstd[i], yerr=CO2_stdmean[i] * CO2_stdrsd[i], lw=0.5, ls='none', c='k', zorder=10)
    else: 
        ax[0].scatter(H2O_stdexpmean[i], H2O_stdmean[i], s=sz, c=H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec='#171008', lw=0.5, zorder=20)
        ax[0].errorbar(H2O_stdexpmean[i], H2O_stdmean[i], xerr=H2O_stdexpstd[i], yerr=H2O_stdmean[i] * H2O_stdrsd[i], lw=0.5, ls='none', c='k', zorder=10)
        ax[1].scatter(CO2_stdexpmean[i], CO2_stdmean[i], s=sz, c=H2O_stdmean[i], vmin = 0, vmax = h2o_vmax, cmap = 'Blues', ec='#171008', lw=0.5, zorder=20)
        ax[1].errorbar(CO2_stdexpmean[i], CO2_stdmean[i], xerr=CO2_stdexpstd[i], yerr=CO2_stdmean[i] * CO2_stdrsd[i], lw=0.5, ls='none', c='k', zorder=10)
ax[0].scatter(H2O_expmean(Fiege63)+0.01, H2O_mean(Fiege63), s=sz_sm, marker='>', c='#FFFFFF', ec='#171008', lw=0.5, zorder=20)
ax[0].scatter(H2O_expmean(Fiege73)+0.01, H2O_mean(Fiege73), s=sz_sm, marker='>', c='#FFFFFF', ec='#171008', lw=0.5, zorder=20)

ax[0].set_xlim([0, 6])
ax[0].set_ylim([0, 6])
ax[0].set_xlabel('$\mathregular{H_2O}$ Expected (wt.%)')
ax[0].set_ylabel('$\mathregular{H_2O_t}$ Measured by FTIR (wt.%)')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

ax[1].plot(co2_line, co2_line, 'k', lw=0.5, zorder=0)
ax[1].set_xlim([0, 1400])
ax[1].set_ylim([0, 1400])
ax[1].set_xlabel('$\mathregular{CO_2}$ Expected (ppm)')
ax[1].set_ylabel('$\mathregular{CO_2}$ Measured by FTIR (ppm)')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)

cbaxes=inset_axes(ax[1], width="15%", height="5%", loc='lower right') 
cbar = fig.colorbar(scatter1, cax=cbaxes, orientation='horizontal')
cbaxes.xaxis.set_ticks_position("top")
cbaxes.tick_params(labelsize=12)

ax[1].text(0.905, 0.13, '$\mathregular{H_2O}$ (wt.%)', fontsize = 12, horizontalalignment='center', verticalalignment='center', transform=ax[1].transAxes)
plt.tight_layout()
# plt.savefig('FTIRSIMS_Comparison_H2O.pdf')
plt.show()

# %%