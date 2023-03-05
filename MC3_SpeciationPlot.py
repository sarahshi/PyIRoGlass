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

# %% 

path_parent = os.path.dirname(os.getcwd())
path_beg = path_parent + '/BASELINES/'
path_input = path_parent + '/BASELINES/Inputs/'
output_dir = ["FIGURES", "PLOTFILES", "NPZFILES", "LOGFILES", "FINALDATA"] 
OUTPUT_PATH = ['F18', 'STD', 'FRH', 'SIMSSTD']

for ii in range(len(output_dir)):
    if not os.path.exists(path_beg + output_dir[ii]):
       os.makedirs(path_beg + output_dir[ii], exist_ok=True)

df_f18 = pd.read_csv(output_dir[-1] + '/' + 'F18_H2OCO2.csv', index_col = 0)
df_f74 = pd.read_csv(output_dir[-1] + '/' + 'STD_H2OCO2.csv', index_col = 0)
df_f74 = df_f74[df_f74.index.str.contains('VF74')]
df_sims1 = pd.read_csv(output_dir[-1] + '/' + 'SIMSSTD_H2OCO2.csv', index_col = 0)
df_sims2 = pd.read_csv(output_dir[-1] + '/' + 'SIMSSTD2_H2OCO2.csv', index_col = 0)
df_sims = pd.concat([df_sims1, df_sims2])

# %%

df = df_sims
df_sat = df[df['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df[df['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.scatter(df_sat['H2Om_1635_BP'], df_sat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.scatter(df_sat['H2Om_1635_BP']+0.005, df_sat['H2Om_5200_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2Om_1635_BP'], df_sat['H2Om_5200_M'], yerr = df_sat['H2Om_5200_STD'], xerr = df_sat['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2Om_1635_BP'], df_unsat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2Om_1635_BP'], df_unsat['H2Om_5200_M'], yerr = df_unsat['H2Om_5200_STD']/2, xerr = df_unsat['H2Om_1635_STD']/2, ls = 'none', elinewidth = 0.5, ecolor = 'k')

sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 7])
ax.set_ylim([0, 7])
ax.set_xlabel('$\mathregular{H_2O_{m, 1635}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/5200v1635_SIMS.pdf')

# %%

df = df_sims
df_sat = df[df['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df[df['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_1635_BP'], df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_1635_BP']+0.005, df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_3550_M']-df_sat['H2Om_1635_BP'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = (df_sat['H2Om_1635_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M']-df_unsat['H2Om_1635_BP'], df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_3550_M']-df_unsat['H2Om_1635_BP'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = (df_unsat['H2Om_1635_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 7])
ax.set_ylim([0, 7])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500} - H_2O_{m, 1635}}$ (wt.%)')
ax.set_ylabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/4500v3550-1635_SIMS.pdf')

# %%

df = df_sims
df_sat = df[df['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df[df['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_5200_M'], df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_5200_M']+0.005, df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_3550_M']-df_sat['H2Om_5200_M'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = (df_sat['H2Om_5200_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M']-df_unsat['H2Om_5200_M'], df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_3550_M']-df_unsat['H2Om_5200_M'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = (df_unsat['H2Om_5200_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 7])
ax.set_ylim([0, 7])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500} - H_2O_{m, 5200}}$ (wt.%)')
ax.set_ylabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/4500v3550-5200_SIMS.pdf')


# %%

df = df_sims
df_sat = df[df['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df[df['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.scatter(df_sat['H2OT_3550_M'], df_sat['H2Om_1635_BP']+df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.scatter(df_sat['H2OT_3550_M']+0.005, df_sat['H2Om_1635_BP']+df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_3550_M'], df_sat['H2Om_1635_BP']+df_sat['OH_4500_M'], yerr = (df_sat['H2Om_1635_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), xerr = df_sat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M'], df_unsat['H2Om_1635_BP']+df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_3550_M'], df_unsat['H2Om_1635_BP']+df_unsat['OH_4500_M'], yerr = (df_unsat['H2Om_1635_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), xerr = df_unsat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 7])
ax.set_ylim([0, 7])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500} - H_2O_{m, 5200}}$ (wt.%)')
ax.set_ylabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/NIR1635vH2O3550_SIMS.pdf')


# %%

df = df_sims
df_sat = df[df['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df[df['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.scatter(df_sat['H2OT_3550_M'], df_sat['H2Om_5200_M']+df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.scatter(df_sat['H2OT_3550_M']+0.005, df_sat['H2Om_5200_M']+df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_3550_M'], df_sat['H2Om_5200_M']+df_sat['OH_4500_M'], yerr = (df_sat['H2Om_5200_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), xerr = df_sat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M'], df_unsat['H2Om_5200_M']+df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_3550_M'], df_unsat['H2Om_5200_M']+df_unsat['OH_4500_M'], yerr = (df_unsat['H2Om_5200_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), xerr = df_unsat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 7])
ax.set_ylim([0, 7])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500} - H_2O_{m, 5200}}$ (wt.%)')
ax.set_ylabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/NIRvH2O3550_SIMS.pdf')

# %%

df = df_sims
df_sat = df[df['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df[df['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]

h2o_tot_wt = np.array([0.00, 0.11, 0.25, 0.35, 0.52, 0.73, 1.02, 1.42, 1.73, 2.00, 2.24, 2.46, 2.66, 2.86, 3.04, 3.22, 4.76, 6.08, 7.32, 8.55, 9.05])
h2o_mol_wt = np.array([0.00, 0.00, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.48, 0.63, 0.78, 0.93, 1.08, 1.22, 1.36, 1.50, 2.83, 4.08, 5.31, 6.56, 7.05])
oh_wt = h2o_tot_wt - h2o_mol_wt

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
# ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.plot(h2o_tot_wt, h2o_mol_wt, '-.', c = '#171008', linewidth = 1.0, label = '$\mathregular{H_2O_{m, 1635}}}$')
ax.plot(h2o_tot_wt, oh_wt, '--', c = '#171008', linewidth = 1.0, label = '$\mathregular{OH^-_{4500}}$')

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Saturated $\mathregular{H_2O_{m, 5200}}$')
ax.scatter(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], s = sz, marker = 's', c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'Saturated $\mathregular{OH^-_{4500}}$')
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M'], yerr = df_sat['H2Om_5200_STD'], xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M'], s = sz, marker = 'o', c = '#9dcaf1', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Unsaturated $\mathregular{H_2O_{m, 5200}}$')
ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['OH_4500_M'], s = sz, marker = 's', c = '#9dcaf1', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Unsaturated $\mathregular{OH^-_{4500}}$')
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M'], yerr = df_unsat['H2Om_5200_STD'], xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

# sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
# ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 9])
ax.set_ylim([0, 9])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200}}$' +' $\mathregular{or}$ '+ '$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/Speciation_5200_SIMS.pdf')


# %% 


df = df_sims
df_sat = df[df['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df[df['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]

h2o_tot_wt = np.array([0.00, 0.11, 0.25, 0.35, 0.52, 0.73, 1.02, 1.42, 1.73, 2.00, 2.24, 2.46, 2.66, 2.86, 3.04, 3.22, 4.76, 6.08, 7.32, 8.55, 9.05])
h2o_mol_wt = np.array([0.00, 0.00, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.48, 0.63, 0.78, 0.93, 1.08, 1.22, 1.36, 1.50, 2.83, 4.08, 5.31, 6.56, 7.05])
oh_wt = h2o_tot_wt - h2o_mol_wt

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
# ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.plot(h2o_tot_wt, h2o_mol_wt, '-.', c = '#171008', linewidth = 1.0, label = '$\mathregular{H_2O_{m, 1635}}}$')
ax.plot(h2o_tot_wt, oh_wt, '--', c = '#171008', linewidth = 1.0, label = '$\mathregular{OH^-_{4500}}$')

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Saturated $\mathregular{H_2O_{m, 5200}}$')
ax.scatter(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], s = sz, marker = 's', c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'Saturated $\mathregular{OH^-_{4500}}$')
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP'], yerr = df_sat['H2Om_1635_STD'], xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP'], s = sz, marker = 'o', c = '#9dcaf1', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Unsaturated $\mathregular{H_2O_{m, 5200}}$')
ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['OH_4500_M'], s = sz, marker = 's', c = '#9dcaf1', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Unsaturated $\mathregular{OH^-_{4500}}$')
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP'], yerr = df_unsat['H2Om_1635_STD'], xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 9])
ax.set_ylim([0, 9])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 1635}}$' +' $\mathregular{or}$ '+ '$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/Speciation_1635_SIMS.pdf')


# %%

df = df_sims
df_sat = df[df['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df[df['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M']/df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Saturated $\mathregular{H_2O_{m, 5200}}$')
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M']/df_sat['OH_4500_M'], yerr = ((df_sat['H2Om_5200_STD']/df_sat['H2Om_5200_M'])**2 + (df_sat['OH_4500_STD']/df_sat['OH_4500_M'])**2)**(1/2), xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M']/df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#9dcaf1', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Unsaturated $\mathregular{H_2O_{m, 5200}}$')
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M']/df_unsat['OH_4500_M'], yerr = ((df_unsat['H2Om_5200_STD']/df_unsat['H2Om_5200_M'])**2 + (df_unsat['OH_4500_STD']/df_unsat['OH_4500_M'])**2)**(1/2), xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 9])
ax.set_ylim([0, 2])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200}}/\mathregular{OH^-_{4500}}$')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/SpeciationRatio_5200_SIMS.pdf')

# %%

df = df_sims
df_sat = df[df['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df[df['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP']/df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Saturated $\mathregular{H_2O_{m, 5200}}$')
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP']/df_sat['OH_4500_M'], yerr = ((df_sat['H2Om_1635_STD']/df_sat['H2Om_1635_BP'])**2 + (df_sat['OH_4500_STD']/df_sat['OH_4500_M'])**2)**(1/2), xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP']/df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#9dcaf1', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Unsaturated $\mathregular{H_2O_{m, 5200}}$')
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP']/df_unsat['OH_4500_M'], yerr = ((df_unsat['H2Om_1635_STD']/df_unsat['H2Om_1635_BP'])**2 + (df_unsat['OH_4500_STD']/df_unsat['OH_4500_M'])**2)**(1/2), xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 9])
ax.set_ylim([0, 2])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 1635}}/\mathregular{OH^-_{4500}}$')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/SpeciationRatio_1635_SIMS.pdf')

# %%
# %% 
# %% 
# %% 
# %% 
# %% 

df = df_f18
prefix = df.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2 = df.loc[df.groupby(prefix)['CO2_MEAN'].idxmax()]
df1 = df_f74
df1 = df1[~df1.index.str.startswith('VF74_134D')]
prefix = df1.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2_1 = df1.loc[df1.groupby(prefix)['CO2_MEAN'].idxmax()]

df_sat = df_co2[df_co2['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df_co2[df_co2['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]
df_unsat1 = df_co2_1[df_co2_1['H2OT_3550_SAT'] == '-']
df_unsat1 = df_unsat1[(df_unsat1['ERR_5200']=='-') & (df_unsat1['ERR_4500']=='-')]
df_unsat1 = df_unsat1[df_unsat1['H2Om_5200_M']/df_unsat1['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.scatter(df_sat['H2Om_1635_BP'], df_sat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 2018')
ax.scatter(df_sat['H2Om_1635_BP']+0.005, df_sat['H2Om_5200_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2Om_1635_BP'], df_sat['H2Om_5200_M'], yerr = df_sat['H2Om_5200_STD'], xerr = df_sat['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2Om_1635_BP'], df_unsat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2Om_1635_BP'], df_unsat['H2Om_5200_M'], yerr = df_unsat['H2Om_5200_STD'], xerr = df_unsat['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2Om_1635_BP'], df_unsat1['H2Om_5200_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2Om_1635_BP'], df_unsat1['H2Om_5200_M'], yerr = df_unsat1['H2Om_5200_STD'], xerr = df_unsat1['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
ax.set_xlabel('$\mathregular{H_2O_{m, 1635}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/5200v1635_VF.pdf')

# %% 
# %%

df = df_f18
prefix = df.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2 = df.loc[df.groupby(prefix)['CO2_MEAN'].idxmax()]
df1 = df_f74
df1 = df1[~df1.index.str.startswith('VF74_134D')]
prefix = df1.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2_1 = df1.loc[df1.groupby(prefix)['CO2_MEAN'].idxmax()]

df_sat = df_co2[df_co2['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df_co2[df_co2['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]
df_unsat1 = df_co2_1[df_co2_1['H2OT_3550_SAT'] == '-']
df_unsat1 = df_unsat1[(df_unsat1['ERR_5200']=='-') & (df_unsat1['ERR_4500']=='-')]
df_unsat1 = df_unsat1[df_unsat1['H2Om_5200_M']/df_unsat1['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_1635_BP'], df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 2018')
ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_1635_BP']+0.005, df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_3550_M']-df_sat['H2Om_1635_BP'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = (df_sat['H2Om_1635_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')


ax.scatter(df_unsat['H2OT_3550_M']-df_unsat['H2Om_1635_BP'], df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_3550_M']-df_unsat['H2Om_1635_BP'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = (df_unsat['H2Om_1635_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2OT_3550_M']-df_unsat1['H2Om_1635_BP'], df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2OT_3550_M']-df_unsat1['H2Om_1635_BP'], df_unsat1['OH_4500_M'], yerr = df_unsat1['OH_4500_STD'], xerr = (df_unsat1['H2Om_1635_STD']**2 + df_unsat1['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500} - H_2O_{m, 1635}}$ (wt.%)')
ax.set_ylabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/4500v3550-1635_VF.pdf')

# %%
# %%

df = df_f18
prefix = df.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2 = df.loc[df.groupby(prefix)['CO2_MEAN'].idxmax()]
df1 = df_f74
df1 = df1[~df1.index.str.startswith('VF74_134D')]
prefix = df1.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2_1 = df1.loc[df1.groupby(prefix)['CO2_MEAN'].idxmax()]

df_sat = df_co2[df_co2['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df_co2[df_co2['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]
df_unsat1 = df_co2_1[df_co2_1['H2OT_3550_SAT'] == '-']
df_unsat1 = df_unsat1[(df_unsat1['ERR_5200']=='-') & (df_unsat1['ERR_4500']=='-')]
df_unsat1 = df_unsat1[df_unsat1['H2Om_5200_M']/df_unsat1['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_5200_M'], df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 2018')
ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_5200_M']+0.005, df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_3550_M']-df_sat['H2Om_5200_M'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = (df_sat['H2Om_5200_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M']-df_unsat['H2Om_5200_M'], df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_3550_M']-df_unsat['H2Om_5200_M'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = (df_unsat['H2Om_5200_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.scatter(df_unsat1['H2OT_3550_M']-df_unsat1['H2Om_5200_M'], df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2OT_3550_M']-df_unsat1['H2Om_5200_M'], df_unsat1['OH_4500_M'], yerr = df_unsat1['OH_4500_STD'], xerr = (df_unsat1['H2Om_5200_STD']**2 + df_unsat1['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500} - H_2O_{m, 5200}}$ (wt.%)')
ax.set_ylabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/4500v3550-5200_VF.pdf')

# %%
# %%

df = df_f18
prefix = df.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2 = df.loc[df.groupby(prefix)['CO2_MEAN'].idxmax()]
df1 = df_f74
df1 = df1[~df1.index.str.startswith('VF74_134D')]
prefix = df1.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2_1 = df1.loc[df1.groupby(prefix)['CO2_MEAN'].idxmax()]

df_sat = df_co2[df_co2['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df_co2[df_co2['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]
df_unsat1 = df_co2_1[df_co2_1['H2OT_3550_SAT'] == '-']
df_unsat1 = df_unsat1[(df_unsat1['ERR_5200']=='-') & (df_unsat1['ERR_4500']=='-')]
df_unsat1 = df_unsat1[df_unsat1['H2Om_5200_M']/df_unsat1['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.scatter(df_sat['H2OT_3550_M'], df_sat['H2Om_1635_BP']+df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 2018')
ax.scatter(df_sat['H2OT_3550_M']+0.005, df_sat['H2Om_1635_BP']+df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_3550_M'], df_sat['H2Om_1635_BP']+df_sat['OH_4500_M'], yerr = (df_sat['H2Om_1635_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), xerr = df_sat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M'], df_unsat['H2Om_1635_BP']+df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_3550_M'], df_unsat['H2Om_1635_BP']+df_unsat['OH_4500_M'], yerr = (df_unsat['H2Om_1635_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), xerr = df_unsat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.scatter(df_unsat1['H2OT_3550_M'], df_unsat1['H2Om_1635_BP']+df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2OT_3550_M'], df_unsat1['H2Om_1635_BP']+df_unsat1['OH_4500_M'], yerr = (df_unsat1['H2Om_1635_STD']**2 + df_unsat1['OH_4500_STD']**2)**(1/2), xerr = df_unsat1['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500} - H_2O_{m, 1635}}$ (wt.%)')
ax.set_ylabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/NIR1635vH2O3550_VF.pdf')

# %%
# %%

df = df_f18
prefix = df.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2 = df.loc[df.groupby(prefix)['CO2_MEAN'].idxmax()]
df1 = df_f74
df1 = df1[~df1.index.str.startswith('VF74_134D')]
prefix = df1.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2_1 = df1.loc[df1.groupby(prefix)['CO2_MEAN'].idxmax()]

df_sat = df_co2[df_co2['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df_co2[df_co2['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]
df_unsat1 = df_co2_1[df_co2_1['H2OT_3550_SAT'] == '-']
df_unsat1 = df_unsat1[(df_unsat1['ERR_5200']=='-') & (df_unsat1['ERR_4500']=='-')]
df_unsat1 = df_unsat1[df_unsat1['H2Om_5200_M']/df_unsat1['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.scatter(df_sat['H2OT_3550_M'], df_sat['H2Om_5200_M']+df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 2018')
ax.scatter(df_sat['H2OT_3550_M']+0.005, df_sat['H2Om_5200_M']+df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_3550_M'], df_sat['H2Om_5200_M']+df_sat['OH_4500_M'], yerr = (df_sat['H2Om_5200_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), xerr = df_sat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M'], df_unsat['H2Om_5200_M']+df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_3550_M'], df_unsat['H2Om_5200_M']+df_unsat['OH_4500_M'], yerr = (df_unsat['H2Om_5200_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), xerr = df_unsat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.scatter(df_unsat1['H2OT_3550_M'], df_unsat1['H2Om_5200_M']+df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2OT_3550_M'], df_unsat1['H2Om_5200_M']+df_unsat1['OH_4500_M'], yerr = (df_unsat1['H2Om_5200_STD']**2 + df_unsat1['OH_4500_STD']**2)**(1/2), xerr = df_unsat1['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)

sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500} - H_2O_{m, 5200}}$ (wt.%)')
ax.set_ylabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/NIRvH2O3550_VF.pdf')

# %%
# %%

df = df_f18
prefix = df.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2 = df.loc[df.groupby(prefix)['CO2_MEAN'].idxmax()]
df1 = df_f74
df1 = df1[~df1.index.str.startswith('VF74_134D')]
prefix = df1.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2_1 = df1.loc[df1.groupby(prefix)['CO2_MEAN'].idxmax()]

df_sat = df_co2[df_co2['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df_co2[df_co2['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]
df_unsat1 = df_co2_1[df_co2_1['H2OT_3550_SAT'] == '-']
df_unsat1 = df_unsat1[(df_unsat1['ERR_5200']=='-') & (df_unsat1['ERR_4500']=='-')]
df_unsat1 = df_unsat1[df_unsat1['H2Om_5200_M']/df_unsat1['H2Om_1635_BP'] < 1.5]

h2o_tot_wt = np.array([0.00, 0.11, 0.25, 0.35, 0.52, 0.73, 1.02, 1.42, 1.73, 2.00, 2.24, 2.46, 2.66, 2.86, 3.04, 3.22, 4.76, 6.08, 7.32, 8.55, 9.05])
h2o_mol_wt = np.array([0.00, 0.00, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.48, 0.63, 0.78, 0.93, 1.08, 1.22, 1.36, 1.50, 2.83, 4.08, 5.31, 6.56, 7.05])
oh_wt = h2o_tot_wt - h2o_mol_wt

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
# ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.plot(h2o_tot_wt, h2o_mol_wt, '-.', c = '#171008', linewidth = 1.0, label = '$\mathregular{H_2O_{m, 1635}}}$')
ax.plot(h2o_tot_wt, oh_wt, '--', c = '#171008', linewidth = 1.0, label = '$\mathregular{OH^-_{4500}}$')

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 2018 $\mathregular{H_2O_{m, 5200}}$')
ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.scatter(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], s = sz, marker = 's', c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'Fuego 2018 $\mathregular{OH^-_{4500}}$')
ax.scatter(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M'], yerr = df_sat['H2Om_5200_STD'], xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['OH_4500_M'], s = sz, marker = 's', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M'], yerr = df_unsat['H2Om_5200_STD'], xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_5200_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 1974 $\mathregular{H_2O_{m, 5200}}$')
ax.scatter(df_unsat1['H2OT_MEAN'], df_unsat1['OH_4500_M'], s = sz, marker = 's', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 1974 $\mathregular{OH^-_{4500}}$')
ax.errorbar(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_5200_M'], yerr = df_unsat1['H2Om_5200_STD'], xerr = df_unsat1['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_unsat1['H2OT_MEAN'], df_unsat1['OH_4500_M'], yerr = df_unsat1['OH_4500_STD'], xerr = df_unsat1['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200}}$' +' $\mathregular{or}$ '+ '$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/Speciation_5200_VF.pdf')


# %% 
# %% 

df = df_f18
prefix = df.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2 = df.loc[df.groupby(prefix)['CO2_MEAN'].idxmax()]
df1 = df_f74
df1 = df1[~df1.index.str.startswith('VF74_134D')]
prefix = df1.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2_1 = df1.loc[df1.groupby(prefix)['CO2_MEAN'].idxmax()]

df_sat = df_co2[df_co2['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df_co2[df_co2['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]
df_unsat1 = df_co2_1[df_co2_1['H2OT_3550_SAT'] == '-']
df_unsat1 = df_unsat1[(df_unsat1['ERR_5200']=='-') & (df_unsat1['ERR_4500']=='-')]
df_unsat1 = df_unsat1[df_unsat1['H2Om_5200_M']/df_unsat1['H2Om_1635_BP'] < 1.5]


h2o_tot_wt = np.array([0.00, 0.11, 0.25, 0.35, 0.52, 0.73, 1.02, 1.42, 1.73, 2.00, 2.24, 2.46, 2.66, 2.86, 3.04, 3.22, 4.76, 6.08, 7.32, 8.55, 9.05])
h2o_mol_wt = np.array([0.00, 0.00, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.48, 0.63, 0.78, 0.93, 1.08, 1.22, 1.36, 1.50, 2.83, 4.08, 5.31, 6.56, 7.05])
oh_wt = h2o_tot_wt - h2o_mol_wt

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
# ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.plot(h2o_tot_wt, h2o_mol_wt, '-.', c = '#171008', linewidth = 1.0, label = '$\mathregular{H_2O_{m, 1635}}}$')
ax.plot(h2o_tot_wt, oh_wt, '--', c = '#171008', linewidth = 1.0, label = '$\mathregular{OH^-_{4500}}$')

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 2018 $\mathregular{H_2O_{m, 5200}}$')
ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.scatter(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], s = sz, marker = 's', c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'Fuego 2018 $\mathregular{OH^-_{4500}}$')
ax.scatter(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP'], yerr = df_sat['H2Om_1635_STD'], xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['OH_4500_M'], s = sz, marker = 's', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP'], yerr = df_unsat['H2Om_1635_STD'], xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_1635_BP'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 1974 $\mathregular{H_2O_{m, 5200}}$')
ax.scatter(df_unsat1['H2OT_MEAN'], df_unsat1['OH_4500_M'], s = sz, marker = 's', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 1974 $\mathregular{OH^-_{4500}}$')
ax.errorbar(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_1635_BP'], yerr = df_unsat1['H2Om_1635_STD'], xerr = df_unsat1['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_unsat1['H2OT_MEAN'], df_unsat1['OH_4500_M'], yerr = df_unsat1['OH_4500_STD'], xerr = df_unsat1['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 1635}}$' +' $\mathregular{or}$ '+ '$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/Speciation_1635_VF.pdf')


# %% 
# %%

df = df_f18
prefix = df.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2 = df.loc[df.groupby(prefix)['CO2_MEAN'].idxmax()]
df1 = df_f74
df1 = df1[~df1.index.str.startswith('VF74_134D')]
prefix = df1.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2_1 = df1.loc[df1.groupby(prefix)['CO2_MEAN'].idxmax()]

df_sat = df_co2[df_co2['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df_co2[df_co2['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]
df_unsat1 = df_co2_1[df_co2_1['H2OT_3550_SAT'] == '-']
df_unsat1 = df_unsat1[(df_unsat1['ERR_5200']=='-') & (df_unsat1['ERR_4500']=='-')]
df_unsat1 = df_unsat1[df_unsat1['H2Om_5200_M']/df_unsat1['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M']/df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 2018')
ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M']/df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M']/df_sat['OH_4500_M'], yerr = ((df_sat['H2Om_5200_STD']/df_sat['H2Om_5200_M'])**2 + (df_sat['OH_4500_STD']/df_sat['OH_4500_M'])**2)**(1/2), xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M']/df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M']/df_unsat['OH_4500_M'], yerr = ((df_unsat['H2Om_5200_STD']/df_unsat['H2Om_5200_M'])**2 + (df_unsat['OH_4500_STD']/df_unsat['OH_4500_M'])**2)**(1/2), xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_5200_M']/df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 5, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_5200_M']/df_unsat1['OH_4500_M'], yerr = ((df_unsat1['H2Om_5200_STD']/df_unsat1['H2Om_5200_M'])**2 + (df_unsat1['OH_4500_STD']/df_unsat1['OH_4500_M'])**2)**(1/2), xerr = df_unsat1['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([2, 5])
ax.set_ylim([0, 2])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200}}/\mathregular{OH^-_{4500}}$')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/SpeciationRatio_5200_VF.pdf')

# %%
# %%

df = df_f18
prefix = df.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2 = df.loc[df.groupby(prefix)['CO2_MEAN'].idxmax()]
df1 = df_f74
df1 = df1[~df1.index.str.startswith('VF74_134D')]
prefix = df1.index.str.split('_').str[:-4]
prefix = prefix.str.join('_')
df_co2_1 = df1.loc[df1.groupby(prefix)['CO2_MEAN'].idxmax()]

df_sat = df_co2[df_co2['H2OT_3550_SAT'] == '*']
df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
df_unsat = df_co2[df_co2['H2OT_3550_SAT'] == '-']
df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]
df_unsat1 = df_co2_1[df_co2_1['H2OT_3550_SAT'] == '-']
df_unsat1 = df_unsat1[(df_unsat1['ERR_5200']=='-') & (df_unsat1['ERR_4500']=='-')]
df_unsat1 = df_unsat1[df_unsat1['H2Om_5200_M']/df_unsat1['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP']/df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 2018')
ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP']/df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP']/df_sat['OH_4500_M'], yerr = ((df_sat['H2Om_1635_STD']/df_sat['H2Om_1635_BP'])**2 + (df_sat['OH_4500_STD']/df_sat['OH_4500_M'])**2)**(1/2), xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP']/df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP']/df_unsat['OH_4500_M'], yerr = ((df_unsat['H2Om_1635_STD']/df_unsat['H2Om_1635_BP'])**2 + (df_unsat['OH_4500_STD']/df_unsat['OH_4500_M'])**2)**(1/2), xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_1635_BP']/df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 5, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_1635_BP']/df_unsat1['OH_4500_M'], yerr = ((df_unsat1['H2Om_1635_STD']/df_unsat1['H2Om_1635_BP'])**2 + (df_unsat1['OH_4500_STD']/df_unsat1['OH_4500_M'])**2)**(1/2), xerr = df_unsat1['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.50, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([2, 5])
ax.set_ylim([0, 2])
ax.set_xlabel('$\mathregular{H_2O_{t, 3500}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 1635}}/\mathregular{OH^-_{4500}}$')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('VOLATILESANDSPECIATION_FINAL/SpeciationRatio_1635_VF.pdf')

# %%
