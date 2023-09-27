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


# %% 

path_parent = os.path.dirname(os.getcwd())


df_f18 = pd.read_csv(path_parent + '/FINALDATA/FUEGO_H2OCO2.csv', index_col = 0)
df_std = pd.read_csv(path_parent + '/FINALDATA/STD_H2OCO2.csv', index_col = 0)
df_f74 = df_std[df_std.index.str.contains('VF74')]
df_std = df_std[~df_std.index.str.contains('VF74')]
df_sims = pd.read_csv(path_parent + '/FINALDATA/STD_H2OCO2.csv', index_col = 0)

def df_filt(df): 

    df_sat = df[df['H2OT_3550_SAT'] == '*']
    df_sat = df_sat[(df_sat['ERR_5200']=='-') & (df_sat['ERR_4500']=='-')]
    df_unsat = df[df['H2OT_3550_SAT'] == '-']
    df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
    df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]

    return df_sat, df_unsat

def std_filt(df): 

    prefix = df.index.str.split('_').str[:-3]
    prefix = prefix.str.join('_')
    df = df.loc[df.groupby(prefix)['CO2_MEAN'].idxmax()]
    
    return df

def fuego_filt(df): 

    prefix = df.index.str.split('_').str[:-4]
    prefix = prefix.str.join('_')
    df = df.loc[df.groupby(prefix)['CO2_MEAN'].idxmax()]
    
    return df

def fuego74_filt(df): 

    df = df[~df.index.str.startswith('VF74_134D')]
    prefix = df.index.str.split('_').str[:-4]
    prefix = prefix.str.join('_')
    df = df.loc[df.groupby(prefix)['CO2_MEAN'].idxmax()]
    
    return df


# %%

df_sat, df_unsat = df_filt(df_sims)
df_sat_std, df_unsat_std = df_filt(df_std)
df_sat_std = std_filt(df_sat_std)
df_unsat_std = std_filt(df_unsat_std)

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.scatter(df_sat['H2Om_1635_BP'], df_sat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Standards')
ax.scatter(df_sat['H2Om_1635_BP']+0.005, df_sat['H2Om_5200_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2Om_1635_BP'], df_sat['H2Om_5200_M'], yerr = df_sat['H2Om_5200_STD'], xerr = df_sat['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2Om_1635_BP'], df_unsat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2Om_1635_BP'], df_unsat['H2Om_5200_M'], yerr = df_unsat['H2Om_5200_STD'], xerr = df_unsat['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_sat_std['H2Om_1635_BP'], df_sat_std['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.scatter(df_sat_std['H2Om_1635_BP']+0.005, df_sat_std['H2Om_5200_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat_std['H2Om_1635_BP'], df_sat_std['H2Om_5200_M'], yerr = df_sat_std['H2Om_5200_STD'], xerr = df_sat_std['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat_std['H2Om_1635_BP'], df_unsat_std['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat_std['H2Om_1635_BP'], df_unsat_std['H2Om_5200_M'], yerr = df_unsat_std['H2Om_5200_STD'], xerr = df_unsat_std['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 6])
ax.set_ylim([0, 6])
ax.set_xlabel('$\mathregular{H_2O_{m, 1635}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/5200v1635_STD.pdf')

# %%

# %%

df_sat, df_unsat = df_filt(df_sims)
df_sat_std, df_unsat_std = df_filt(df_std)
df_sat_std = std_filt(df_sat_std)
df_unsat_std = std_filt(df_unsat_std)

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
# ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_1635_BP'], df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
# ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_1635_BP'], df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax.errorbar(df_sat['H2OT_3550_M']-df_sat['H2Om_1635_BP'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = (df_sat['H2Om_1635_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M']-df_unsat['H2Om_1635_BP'], df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Standards')
ax.errorbar(df_unsat['H2OT_3550_M']-df_unsat['H2Om_1635_BP'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = (df_unsat['H2Om_1635_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

# ax.scatter(df_sat_std['H2OT_3550_M']-df_sat_std['H2Om_1635_BP'], df_sat_std['OH_4500_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15)
# ax.scatter(df_sat_std['H2OT_3550_M']-df_sat_std['H2Om_1635_BP'], df_sat_std['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax.errorbar(df_sat_std['H2OT_3550_M']-df_sat_std['H2Om_1635_BP'], df_sat_std['OH_4500_M'], yerr = df_sat_std['OH_4500_STD'], xerr = (df_sat_std['H2Om_1635_STD']**2 + df_sat_std['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat_std['H2OT_3550_M']-df_unsat_std['H2Om_1635_BP'], df_unsat_std['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat_std['H2OT_3550_M']-df_unsat_std['H2Om_1635_BP'], df_unsat_std['OH_4500_M'], yerr = df_unsat_std['OH_4500_STD'], xerr = (df_unsat_std['H2Om_1635_STD']**2 + df_unsat_std['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.legend(loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
# sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
# ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel('$\mathregular{H_2O_{t, 3550} - H_2O_{m, 1635}}$ (wt.%)')
ax.set_ylabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/4500v3550-1635_STD.pdf')

# %%

df_sat, df_unsat = df_filt(df_sims)
df_sat_std, df_unsat_std = df_filt(df_std)
df_sat_std = std_filt(df_sat_std)
df_unsat_std = std_filt(df_unsat_std)

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
# ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_5200_M'], df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
# ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_5200_M']+0.005, df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax.errorbar(df_sat['H2OT_3550_M']-df_sat['H2Om_5200_M'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = (df_sat['H2Om_5200_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M']-df_unsat['H2Om_5200_M'], df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Standards')
ax.errorbar(df_unsat['H2OT_3550_M']-df_unsat['H2Om_5200_M'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = (df_unsat['H2Om_5200_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

# ax.scatter(df_sat_std['H2OT_3550_M']-df_sat_std['H2Om_5200_M'], df_sat_std['OH_4500_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15)
# ax.scatter(df_sat_std['H2OT_3550_M']-df_sat_std['H2Om_5200_M']+0.005, df_sat_std['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax.errorbar(df_sat_std['H2OT_3550_M']-df_sat_std['H2Om_5200_M'], df_sat_std['OH_4500_M'], yerr = df_sat_std['OH_4500_STD'], xerr = (df_sat_std['H2Om_5200_STD']**2 + df_sat_std['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat_std['H2OT_3550_M']-df_unsat_std['H2Om_5200_M'], df_unsat_std['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat_std['H2OT_3550_M']-df_unsat_std['H2Om_5200_M'], df_unsat_std['OH_4500_M'], yerr = df_unsat_std['OH_4500_STD'], xerr = (df_unsat_std['H2Om_5200_STD']**2 + df_unsat_std['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.legend(loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

# sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
# ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel('$\mathregular{H_2O_{t, 3550} - H_2O_{m, 5200}}$ (wt.%)')
ax.set_ylabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/4500v3550-5200_STD.pdf')


# %%
# %%

df_sat, df_unsat = df_filt(df_sims)
df_sat_std, df_unsat_std = df_filt(df_std)
df_sat_std = std_filt(df_sat_std)
df_unsat_std = std_filt(df_unsat_std)

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
# ax.scatter(df_sat['H2OT_3550_M'], df_sat['H2Om_1635_BP']+df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
# ax.scatter(df_sat['H2OT_3550_M']+0.005, df_sat['H2Om_1635_BP']+df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax.errorbar(df_sat['H2OT_3550_M'], df_sat['H2Om_1635_BP']+df_sat['OH_4500_M'], yerr = (df_sat['H2Om_1635_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), xerr = df_sat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M'], df_unsat['H2Om_1635_BP']+df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_3550_M'], df_unsat['H2Om_1635_BP']+df_unsat['OH_4500_M'], yerr = (df_unsat['H2Om_1635_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), xerr = df_unsat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat_std['H2OT_3550_M'], df_unsat_std['H2Om_1635_BP']+df_unsat_std['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat_std['H2OT_3550_M'], df_unsat_std['H2Om_1635_BP']+df_unsat_std['OH_4500_M'], yerr = (df_unsat_std['H2Om_1635_STD']**2 + df_unsat_std['OH_4500_STD']**2)**(1/2), xerr = df_unsat_std['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')


sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 7])
ax.set_ylim([0, 7])
ax.set_xlabel('$\mathregular{H_2O_{t, 3550}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 1635}+OH^-_{4500}}$ (wt.%)')

ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/NIR1635vH2O3550_STD.pdf')


# %%

df_sat, df_unsat = df_filt(df_sims)
df_sat_std, df_unsat_std = df_filt(df_std)
df_sat_std = std_filt(df_sat_std)
df_unsat_std = std_filt(df_unsat_std)

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
# ax.scatter(df_sat['H2OT_3550_M'], df_sat['H2Om_5200_M']+df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
# ax.scatter(df_sat['H2OT_3550_M']+0.005, df_sat['H2Om_5200_M']+df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax.errorbar(df_sat['H2OT_3550_M'], df_sat['H2Om_5200_M']+df_sat['OH_4500_M'], yerr = (df_sat['H2Om_5200_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), xerr = df_sat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M'], df_unsat['H2Om_5200_M']+df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Standards')
ax.errorbar(df_unsat['H2OT_3550_M'], df_unsat['H2Om_5200_M']+df_unsat['OH_4500_M'], yerr = (df_unsat['H2Om_5200_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), xerr = df_unsat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat_std['H2OT_3550_M'], df_unsat_std['H2Om_5200_M']+df_unsat_std['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat_std['H2OT_3550_M'], df_unsat_std['H2Om_5200_M']+df_unsat_std['OH_4500_M'], yerr = (df_unsat_std['H2Om_5200_STD']**2 + df_unsat_std['OH_4500_STD']**2)**(1/2), xerr = df_unsat_std['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
# sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
# ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 7])
ax.set_ylim([0, 7])
ax.set_xlabel('$\mathregular{H_2O_{t, 3550}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200}+OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/NIRvH2O3550_STD.pdf')

# %%
# %%

df_sat, df_unsat = df_filt(df_sims)
df_sat_std, df_unsat_std = df_filt(df_std)
df_sat_std = std_filt(df_sat_std)
df_unsat_std = std_filt(df_unsat_std)


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
# ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 9])
ax.set_ylim([0, 9])
ax.set_xlabel('$\mathregular{H_2O_{t}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200}}$' +' $\mathregular{or}$ '+ '$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/Speciation_5200_STD.pdf')


# %% 

df_sat, df_unsat = df_filt(df_sims)
df_sat_std, df_unsat_std = df_filt(df_std)
df_sat_std = std_filt(df_sat_std)
df_unsat_std = std_filt(df_unsat_std)

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

ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 9])
ax.set_ylim([0, 9])
ax.set_xlabel('$\mathregular{H_2O_{t}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 1635}}$' +' $\mathregular{or}$ '+ '$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/Speciation_1635_STD.pdf')


# %%

df_sat, df_unsat = df_filt(df_sims)
df_sat_std, df_unsat_std = df_filt(df_std)
df_sat_std = std_filt(df_sat_std)
df_unsat_std = std_filt(df_unsat_std)

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M']/df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Saturated $\mathregular{H_2O_{m, 5200}}$')
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M']/df_sat['OH_4500_M'], yerr = ((df_sat['H2Om_5200_STD']/df_sat['H2Om_5200_M'])**2 + (df_sat['OH_4500_STD']/df_sat['OH_4500_M'])**2)**(1/2), xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M']/df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#9dcaf1', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Unsaturated $\mathregular{H_2O_{m, 5200}}$')
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M']/df_unsat['OH_4500_M'], yerr = ((df_unsat['H2Om_5200_STD']/df_unsat['H2Om_5200_M'])**2 + (df_unsat['OH_4500_STD']/df_unsat['OH_4500_M'])**2)**(1/2), xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 9])
ax.set_ylim([0, 2])
ax.set_xlabel('$\mathregular{H_2O_{t, 3550}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200}}/\mathregular{OH^-_{4500}}$')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/SpeciationRatio_5200_STD.pdf')

# %%

df_sat, df_unsat = df_filt(df_sims)
df_sat_std, df_unsat_std = df_filt(df_std)
df_sat_std = std_filt(df_sat_std)
df_unsat_std = std_filt(df_unsat_std)

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP']/df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Saturated $\mathregular{H_2O_{m, 5200}}$')
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP']/df_sat['OH_4500_M'], yerr = ((df_sat['H2Om_1635_STD']/df_sat['H2Om_1635_BP'])**2 + (df_sat['OH_4500_STD']/df_sat['OH_4500_M'])**2)**(1/2), xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP']/df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#9dcaf1', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Unsaturated $\mathregular{H_2O_{m, 5200}}$')
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP']/df_unsat['OH_4500_M'], yerr = ((df_unsat['H2Om_1635_STD']/df_unsat['H2Om_1635_BP'])**2 + (df_unsat['OH_4500_STD']/df_unsat['OH_4500_M'])**2)**(1/2), xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

ax.set_xlim([0, 9])
ax.set_ylim([0, 2])
ax.set_xlabel('$\mathregular{H_2O_{t, 3550}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 1635}}/\mathregular{OH^-_{4500}}$')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/SpeciationRatio_1635_STD.pdf')

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
df_unsat1 = df_unsat1[(df_unsat1['ERR_5200']=='-') & (df_unsat1['ERR_4500']=='-') ]
df_unsat1 = df_unsat1[df_unsat1['H2Om_5200_M']/df_unsat1['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.scatter(df_sat['H2Om_1635_BP'], df_sat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego')
ax.scatter(df_sat['H2Om_1635_BP']+0.005, df_sat['H2Om_5200_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2Om_1635_BP'], df_sat['H2Om_5200_M'], yerr = df_sat['H2Om_5200_STD'], xerr = df_sat['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2Om_1635_BP'], df_unsat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2Om_1635_BP'], df_unsat['H2Om_5200_M'], yerr = df_unsat['H2Om_5200_STD'], xerr = df_unsat['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2Om_1635_BP'], df_unsat1['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15) #, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2Om_1635_BP'], df_unsat1['H2Om_5200_M'], yerr = df_unsat1['H2Om_5200_STD'], xerr = df_unsat1['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
ax.set_xlabel('$\mathregular{H_2O_{m, 1635}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/5200v1635_VF.pdf')

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
df_unsat = df_unsat[(df_unsat['ERR_4500']=='-') & (df_unsat['ERR_4500']=='-')] # df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]
df_unsat1 = df_co2_1[df_co2_1['H2OT_3550_SAT'] == '-']
df_unsat1 = df_unsat1[(df_unsat1['ERR_5200']=='-') & (df_unsat1['ERR_4500']=='-')]
df_unsat1 = df_unsat1[df_unsat1['H2Om_5200_M']/df_unsat1['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
# ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_1635_BP'], df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego')
# ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_1635_BP']+0.005, df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax.errorbar(df_sat['H2OT_3550_M']-df_sat['H2Om_1635_BP'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = (df_sat['H2Om_1635_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M']-df_unsat['H2Om_1635_BP'], df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego')
ax.errorbar(df_unsat['H2OT_3550_M']-df_unsat['H2Om_1635_BP'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = (df_unsat['H2Om_1635_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2OT_3550_M']-df_unsat1['H2Om_1635_BP'], df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15) #, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2OT_3550_M']-df_unsat1['H2Om_1635_BP'], df_unsat1['OH_4500_M'], yerr = df_unsat1['OH_4500_STD'], xerr = (df_unsat1['H2Om_1635_STD']**2 + df_unsat1['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

# sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
# ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 4])
ax.set_ylim([0, 4])
ax.set_xlabel('$\mathregular{H_2O_{t, 3550} - H_2O_{m, 1635}}$ (wt.%)')
ax.set_ylabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/4500v3550-1635_VF.pdf')

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
df_unsat = df_unsat[(df_unsat['ERR_4500']=='-') & (df_unsat['ERR_4500']=='-')] # df_unsat = df_unsat[(df_unsat['ERR_5200']=='-') & (df_unsat['ERR_4500']=='-')]
df_unsat = df_unsat[df_unsat['H2Om_5200_M']/df_unsat['H2Om_1635_BP'] < 1.5]
df_unsat1 = df_co2_1[df_co2_1['H2OT_3550_SAT'] == '-']
df_unsat1 = df_unsat1[(df_unsat1['ERR_4500']=='-')]
df_unsat1 = df_unsat1[df_unsat1['H2Om_5200_M']/df_unsat1['H2Om_1635_BP'] < 1.5]

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
# ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_5200_M'], df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego')
# ax.scatter(df_sat['H2OT_3550_M']-df_sat['H2Om_5200_M']+0.005, df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax.errorbar(df_sat['H2OT_3550_M']-df_sat['H2Om_5200_M'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = (df_sat['H2Om_5200_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M']-df_unsat['H2Om_5200_M'], df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego')
ax.errorbar(df_unsat['H2OT_3550_M']-df_unsat['H2Om_5200_M'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = (df_unsat['H2Om_5200_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2OT_3550_M']-df_unsat1['H2Om_5200_M'], df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15) #, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2OT_3550_M']-df_unsat1['H2Om_5200_M'], df_unsat1['OH_4500_M'], yerr = df_unsat1['OH_4500_STD'], xerr = (df_unsat1['H2Om_5200_STD']**2 + df_unsat1['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

# sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
# ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 4])
ax.set_ylim([0, 4])
ax.set_xlabel('$\mathregular{H_2O_{t, 3550} - H_2O_{m, 5200}}$ (wt.%)')
ax.set_ylabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/4500v3550-5200_VF.pdf')

# %%
# %%

df_f18 = fuego_filt(df_f18)
df_f74 = fuego74_filt(df_f74)
df_sat, df_unsat = df_filt(df_f18)
df_sat1, df_unsat1 = df_filt(df_f74)

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
# ax.scatter(df_sat['H2OT_3550_M'], df_sat['H2Om_1635_BP']+df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego')
# ax.scatter(df_sat['H2OT_3550_M']+0.005, df_sat['H2Om_1635_BP']+df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax.errorbar(df_sat['H2OT_3550_M'], df_sat['H2Om_1635_BP']+df_sat['OH_4500_M'], yerr = (df_sat['H2Om_1635_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), xerr = df_sat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M'], df_unsat['H2Om_1635_BP']+df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego')
ax.errorbar(df_unsat['H2OT_3550_M'], df_unsat['H2Om_1635_BP']+df_unsat['OH_4500_M'], yerr = (df_unsat['H2Om_1635_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), xerr = df_unsat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2OT_3550_M'], df_unsat1['H2Om_1635_BP']+df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15) #, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2OT_3550_M'], df_unsat1['H2Om_1635_BP']+df_unsat1['OH_4500_M'], yerr = (df_unsat1['H2Om_1635_STD']**2 + df_unsat1['OH_4500_STD']**2)**(1/2), xerr = df_unsat1['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

# sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
# ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel('$\mathregular{H_2O_{t, 3550}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 1635} + OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/NIR1635vH2O3550_VF.pdf')

# %%
# %%

df_f18 = fuego_filt(df_f18)
df_f74 = fuego74_filt(df_f74)
df_sat, df_unsat = df_filt(df_f18)
df_sat1, df_unsat1 = df_filt(df_f74)

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
# ax.scatter(df_sat['H2OT_3550_M'], df_sat['H2Om_5200_M']+df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 2018')
# ax.scatter(df_sat['H2OT_3550_M']+0.005, df_sat['H2Om_5200_M']+df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
# ax.errorbar(df_sat['H2OT_3550_M'], df_sat['H2Om_5200_M']+df_sat['OH_4500_M'], yerr = (df_sat['H2Om_5200_STD']**2 + df_sat['OH_4500_STD']**2)**(1/2), xerr = df_sat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_3550_M'], df_unsat['H2Om_5200_M']+df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_3550_M'], df_unsat['H2Om_5200_M']+df_unsat['OH_4500_M'], yerr = (df_unsat['H2Om_5200_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), xerr = df_unsat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2OT_3550_M'], df_unsat1['H2Om_5200_M']+df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2OT_3550_M'], df_unsat1['H2Om_5200_M']+df_unsat1['OH_4500_M'], yerr = (df_unsat1['H2Om_5200_STD']**2 + df_unsat1['OH_4500_STD']**2)**(1/2), xerr = df_unsat1['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)

sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel('$\mathregular{H_2O_{t, 3550}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200} + OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/NIRvH2O3550_VF.pdf')

# %%
# %%

df_f18 = fuego_filt(df_f18)
df_f74 = fuego74_filt(df_f74)
df_sat, df_unsat = df_filt(df_f18)
df_sat1, df_unsat1 = df_filt(df_f74)

h2o_tot_wt = np.array([0.00, 0.11, 0.25, 0.35, 0.52, 0.73, 1.02, 1.42, 1.73, 2.00, 2.24, 2.46, 2.66, 2.86, 3.04, 3.22, 4.76, 6.08, 7.32, 8.55, 9.05])
h2o_mol_wt = np.array([0.00, 0.00, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.48, 0.63, 0.78, 0.93, 1.08, 1.22, 1.36, 1.50, 2.83, 4.08, 5.31, 6.56, 7.05])
oh_wt = h2o_tot_wt - h2o_mol_wt

sz_sm = 80
sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
# ax.plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax.plot(h2o_tot_wt, h2o_mol_wt, '-.', c = '#171008', linewidth = 1.0, label = '$\mathregular{H_2O_{m, 1635}}}$')
ax.plot(h2o_tot_wt, oh_wt, '--', c = '#171008', linewidth = 1.0, label = '$\mathregular{OH^-_{4500}}$')

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego $\mathregular{H_2O_{m, 5200}}$')
ax.scatter(df_sat['H2OT_MEAN']+0.005, df_sat['H2Om_5200_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.scatter(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], s = sz, marker = 's', c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'Fuego $\mathregular{OH^-_{4500}}$')
ax.scatter(df_sat['H2OT_MEAN']+0.005, df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M'], yerr = df_sat['H2Om_5200_STD'], xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['OH_4500_M'], s = sz, marker = 's', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M'], yerr = df_unsat['H2Om_5200_STD'], xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15) #, label = 'Fuego 1974 $\mathregular{H_2O_{m, 5200}}$')
ax.scatter(df_unsat1['H2OT_MEAN'], df_unsat1['OH_4500_M'], s = sz, marker = 's', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15) #, label = 'Fuego 1974 $\mathregular{OH^-_{4500}}$')
ax.errorbar(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_5200_M'], yerr = df_unsat1['H2Om_5200_STD'], xerr = df_unsat1['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_unsat1['H2OT_MEAN'], df_unsat1['OH_4500_M'], yerr = df_unsat1['OH_4500_STD'], xerr = df_unsat1['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel('$\mathregular{H_2O_{t, 3550}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200}}$' +' $\mathregular{or}$ '+ '$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/Speciation_5200_VF.pdf')


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

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego $\mathregular{H_2O_{m, 5200}}$')
ax.scatter(df_sat['H2OT_MEAN']+0.005, df_sat['H2Om_1635_BP'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.scatter(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], s = sz, marker = 's', c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20, label = 'Fuego $\mathregular{OH^-_{4500}}$')
ax.scatter(df_sat['H2OT_MEAN']+0.005, df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP'], yerr = df_sat['H2Om_1635_STD'], xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['OH_4500_M'], yerr = df_sat['OH_4500_STD'], xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['OH_4500_M'], s = sz, marker = 's', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP'], yerr = df_unsat['H2Om_1635_STD'], xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['OH_4500_M'], yerr = df_unsat['OH_4500_STD'], xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_1635_BP'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15) #, label = 'Fuego 1974 $\mathregular{H_2O_{m, 5200}}$')
ax.scatter(df_unsat1['H2OT_MEAN'], df_unsat1['OH_4500_M'], s = sz, marker = 's', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15) #, label = 'Fuego 1974 $\mathregular{OH^-_{4500}}$')
ax.errorbar(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_1635_BP'], yerr = df_unsat1['H2Om_1635_STD'], xerr = df_unsat1['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax.errorbar(df_unsat1['H2OT_MEAN'], df_unsat1['OH_4500_M'], yerr = df_unsat1['OH_4500_STD'], xerr = df_unsat1['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([0, 5])
ax.set_ylim([0, 5])
ax.set_xlabel('$\mathregular{H_2O_{t, 3550}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 1635}}$' +' $\mathregular{or}$ '+ '$\mathregular{OH^-_{4500}}$ (wt.%)')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/Speciation_1635_VF.pdf')


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

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M']/df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego')
ax.scatter(df_sat['H2OT_MEAN']+0.005, df_sat['H2Om_5200_M']/df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_5200_M']/df_sat['OH_4500_M'], yerr = ((df_sat['H2Om_5200_STD']/df_sat['H2Om_5200_M'])**2 + (df_sat['OH_4500_STD']/df_sat['OH_4500_M'])**2)**(1/2), xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M']/df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_5200_M']/df_unsat['OH_4500_M'], yerr = ((df_unsat['H2Om_5200_STD']/df_unsat['H2Om_5200_M'])**2 + (df_unsat['OH_4500_STD']/df_unsat['OH_4500_M'])**2)**(1/2), xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_5200_M']/df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 5) #, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_5200_M']/df_unsat1['OH_4500_M'], yerr = ((df_unsat1['H2Om_5200_STD']/df_unsat1['H2Om_5200_M'])**2 + (df_unsat1['OH_4500_STD']/df_unsat1['OH_4500_M'])**2)**(1/2), xerr = df_unsat1['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([2, 5])
ax.set_ylim([0, 2])
ax.set_xlabel('$\mathregular{H_2O_{t}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 5200}}/\mathregular{OH^-_{4500}}$')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/SpeciationRatio_5200_VF.pdf')

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

ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP']/df_sat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego')
ax.scatter(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP']/df_sat['OH_4500_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax.errorbar(df_sat['H2OT_MEAN'], df_sat['H2Om_1635_BP']/df_sat['OH_4500_M'], yerr = ((df_sat['H2Om_1635_STD']/df_sat['H2Om_1635_BP'])**2 + (df_sat['OH_4500_STD']/df_sat['OH_4500_M'])**2)**(1/2), xerr = df_sat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP']/df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax.errorbar(df_unsat['H2OT_MEAN'], df_unsat['H2Om_1635_BP']/df_unsat['OH_4500_M'], yerr = ((df_unsat['H2Om_1635_STD']/df_unsat['H2Om_1635_BP'])**2 + (df_unsat['OH_4500_STD']/df_unsat['OH_4500_M'])**2)**(1/2), xerr = df_unsat['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

ax.scatter(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_1635_BP']/df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 5) #, label = 'Fuego 1974')
ax.errorbar(df_unsat1['H2OT_MEAN'], df_unsat1['H2Om_1635_BP']/df_unsat1['OH_4500_M'], yerr = ((df_unsat1['H2Om_1635_STD']/df_unsat1['H2Om_1635_BP'])**2 + (df_unsat1['OH_4500_STD']/df_unsat1['OH_4500_M'])**2)**(1/2), xerr = df_unsat1['H2OT_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')

leg1 = ax.legend(loc = 'upper left', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
sat_symb = ax.scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax.legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax.add_artist(leg1)

ax.set_xlim([2, 5])
ax.set_ylim([0, 2])
ax.set_xlabel('$\mathregular{H_2O_{t, 3550}}$ (wt.%)')
ax.set_ylabel('$\mathregular{H_2O_{m, 1635}}/\mathregular{OH^-_{4500}}$')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('VOLATILESANDSPECIATION_FINAL/SpeciationRatio_1635_VF.pdf')

# %%

def concordance_correlation_coefficient(y_true, y_pred):
    """Concordance correlation coefficient."""
    # Remove NaNs
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred
    })
    df = df.dropna()
    y_true = df['y_true']
    y_pred = df['y_pred']
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(y_true, y_pred)[0][1]
    # Mean
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    # Variance
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    # Standard deviation
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    return numerator / denominator



# %% SYNTH FUEGO FIGURE 

from sklearn.metrics import mean_squared_error


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
df_unsat1 = df_unsat1[(df_unsat1['ERR_5200']=='-') & (df_unsat1['ERR_4500']=='-') ]
df_unsat1 = df_unsat1[df_unsat1['H2Om_5200_M']/df_unsat1['H2Om_1635_BP'] < 1.5]

df_all = pd.concat([df_sat, df_unsat, df_unsat1])
df_all = df_all[df_all.OH_4500_M<4]


sz = 150

fig, ax = plt.subplots(3, 2, figsize = (14, 17.5))
ax = ax.flatten()

import scipy
slope0, intercept0, r_value0, p_value0, std_err0 = scipy.stats.linregress(df_all['H2Om_1635_BP'], df_all['H2Om_5200_M'])
ccc0 = concordance_correlation_coefficient(df_all['H2Om_1635_BP'], df_all['H2Om_5200_M'])
rmse0 = mean_squared_error(df_all['H2Om_1635_BP'], df_all['H2Om_5200_M'], squared=False)

ax[0].plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax[0].scatter(df_sat['H2Om_1635_BP'], df_sat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego')
ax[0].scatter(df_sat['H2Om_1635_BP']+0.005, df_sat['H2Om_5200_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax[0].errorbar(df_sat['H2Om_1635_BP'], df_sat['H2Om_5200_M'], yerr = df_sat['H2Om_5200_STD'], xerr = df_sat['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax[0].scatter(df_unsat['H2Om_1635_BP'], df_unsat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax[0].errorbar(df_unsat['H2Om_1635_BP'], df_unsat['H2Om_5200_M'], yerr = df_unsat['H2Om_5200_STD'], xerr = df_unsat['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax[0].scatter(df_unsat1['H2Om_1635_BP'], df_unsat1['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15) #, label = 'Fuego 1974')
ax[0].errorbar(df_unsat1['H2Om_1635_BP'], df_unsat1['H2Om_5200_M'], yerr = df_unsat1['H2Om_5200_STD'], xerr = df_unsat1['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
leg1 = ax[0].legend(loc = (0.025, 0.87), labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
sat_symb = ax[0].scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax[0].annotate("A.", xy=(0.035, 0.935), xycoords="axes fraction", fontsize=20, weight='bold')
ax[0].legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax[0].add_artist(leg1)
ax[0].set_xlim([0, 3])
ax[0].set_ylim([0, 3])
ax[0].set_xticks(np.arange(0, 4, 1))
ax[0].set_yticks(np.arange(0, 4, 1))
ax[0].set_xlabel('$\mathregular{H_2O_{m, 1635}}$ (wt.%)')
ax[0].set_ylabel('$\mathregular{H_2O_{m, 5200}}$ (wt.%)')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[0].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value0**2, 2)), xy=(0.035, 0.7675), xycoords="axes fraction", fontsize=12)
ax[0].annotate("RMSE="+str(np.round(rmse0, 2)), xy=(0.035, 0.81), xycoords="axes fraction", fontsize=12)
ax[0].annotate("CCC="+str(np.round(ccc0, 2)), xy=(0.035, 0.85), xycoords="axes fraction", fontsize=12)
ax[0].annotate("m="+str(np.round(slope0, 2)), xy=(0.035, 0.73), xycoords="axes fraction", fontsize=12)
ax[0].annotate("b="+str(np.round(intercept0, 2)), xy=(0.035, 0.69), xycoords="axes fraction", fontsize=12)

slope2, intercept2, r_value2, p_value2, std_err2 = scipy.stats.linregress(df_all['H2OT_3550_M'], df_all['H2Om_1635_BP']+df_all['OH_4500_M'])
ccc2 = concordance_correlation_coefficient(df_all['H2OT_3550_M'], df_all['H2Om_1635_BP']+df_all['OH_4500_M'])
rmse2 = mean_squared_error(df_all['H2OT_3550_M'], df_all['H2Om_1635_BP']+df_all['OH_4500_M'], squared=False)

ax[2].plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax[2].scatter(df_unsat['H2OT_3550_M'], df_unsat['H2Om_1635_BP']+df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego')
ax[2].errorbar(df_unsat['H2OT_3550_M'], df_unsat['H2Om_1635_BP']+df_unsat['OH_4500_M'], yerr = (df_unsat['H2Om_1635_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), xerr = df_unsat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax[2].scatter(df_unsat1['H2OT_3550_M'], df_unsat1['H2Om_1635_BP']+df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15) #, label = 'Fuego 1974')
ax[2].errorbar(df_unsat1['H2OT_3550_M'], df_unsat1['H2Om_1635_BP']+df_unsat1['OH_4500_M'], yerr = (df_unsat1['H2Om_1635_STD']**2 + df_unsat1['OH_4500_STD']**2)**(1/2), xerr = df_unsat1['H2OT_3550_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
leg1 = ax[2].legend(loc = (0.025, 0.87), labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax[2].annotate("C.", xy=(0.035, 0.935), xycoords="axes fraction", fontsize=20, weight='bold')
ax[2].add_artist(leg1)
ax[2].set_xlim([0, 5])
ax[2].set_ylim([0, 5])
ax[2].set_xlabel('$\mathregular{H_2O_{t, 3550}}$ (wt.%)')
ax[2].set_ylabel('$\mathregular{H_2O_{m, 1635} + OH^-_{4500}}$ (wt.%)')
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[2].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value2**2, 2)), xy=(0.035, 0.7675), xycoords="axes fraction", fontsize=12)
ax[2].annotate("CCC="+str(np.round(ccc2, 2)), xy=(0.035, 0.81), xycoords="axes fraction", fontsize=12)
ax[2].annotate("RMSE=0.80", xy=(0.035, 0.85), xycoords="axes fraction", fontsize=12)
ax[2].annotate("m="+str(np.round(slope2, 2)), xy=(0.035, 0.73), xycoords="axes fraction", fontsize=12)
ax[2].annotate("b="+str(np.round(intercept2, 2)), xy=(0.035, 0.69), xycoords="axes fraction", fontsize=12)




slope3, intercept3, r_value3, p_value3, std_err3 = scipy.stats.linregress(df_all['H2OT_3550_M'], df_all['H2Om_5200_M']+df_all['OH_4500_M'])
ccc3 = concordance_correlation_coefficient(df_all['H2OT_3550_M'], df_all['H2Om_5200_M']+df_all['OH_4500_M'])
rmse3 = mean_squared_error(df_all['H2OT_3550_M'], df_all['H2Om_5200_M']+df_all['OH_4500_M'], squared=False)
ax[3].plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax[3].scatter(df_unsat['H2OT_3550_M'], df_unsat['H2Om_5200_M']+df_unsat['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.3, zorder = 15)
ax[3].errorbar(df_unsat['H2OT_3550_M'], df_unsat['H2Om_5200_M']+df_unsat['OH_4500_M'], yerr = (df_unsat['H2Om_5200_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), xerr = df_unsat['H2OT_3550_STD'], ls = 'none', elinewidth = 0.3, ecolor = 'k')
ax[3].scatter(df_unsat1['H2OT_3550_M'], df_unsat1['H2Om_5200_M']+df_unsat1['OH_4500_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.3, zorder = 13, label = 'Fuego')
ax[3].errorbar(df_unsat1['H2OT_3550_M'], df_unsat1['H2Om_5200_M']+df_unsat1['OH_4500_M'], yerr = (df_unsat1['H2Om_5200_STD']**2 + df_unsat1['OH_4500_STD']**2)**(1/2), xerr = df_unsat1['H2OT_3550_STD'], ls = 'none', elinewidth = 0.3, ecolor = 'k')
leg1 = ax[3].legend(loc = (0.025, 0.87), labelspacing = 0.4, handletextpad = 0.3, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax[3].annotate("D.", xy=(0.035, 0.935), xycoords="axes fraction", fontsize=20, weight='bold')
ax[3].add_artist(leg1)
ax[3].set_xlim([0, 5])
ax[3].set_ylim([0, 5])
ax[3].set_xlabel('$\mathregular{H_2O_{t, 3550}}$ (wt.%)')
ax[3].set_ylabel('$\mathregular{H_2O_{m, 5200} + OH^-_{4500}}$ (wt.%)')
ax[3].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[3].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[3].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value3**2, 2)), xy=(0.035, 0.7675), xycoords="axes fraction", fontsize=12)
ax[3].annotate("CCC="+str(np.round(ccc3, 2)), xy=(0.035, 0.85), xycoords="axes fraction", fontsize=12)
ax[3].annotate("RMSE="+str(np.round(rmse3, 2)), xy=(0.035, 0.81), xycoords="axes fraction", fontsize=12)
ax[3].annotate("m="+str(np.round(slope3, 2)), xy=(0.035, 0.73), xycoords="axes fraction", fontsize=12)
ax[3].annotate("b="+str(np.round(intercept3, 2)), xy=(0.035, 0.69), xycoords="axes fraction", fontsize=12)




slope4, intercept4, r_value4, p_value4, std_err4 = scipy.stats.linregress(df_all['OH_4500_M'], df_all['H2OT_3550_M']-df_all['H2Om_1635_BP'])
ccc4 = concordance_correlation_coefficient(df_all['OH_4500_M'], df_all['H2OT_3550_M']-df_all['H2Om_1635_BP'])
rmse4 = mean_squared_error(df_all['OH_4500_M'], df_all['H2OT_3550_M']-df_all['H2Om_1635_BP'], squared=False)
ax[4].plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax[4].scatter(df_unsat['OH_4500_M'], df_unsat['H2OT_3550_M']-df_unsat['H2Om_1635_BP'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego')
ax[4].errorbar(df_unsat['OH_4500_M'], df_unsat['H2OT_3550_M']-df_unsat['H2Om_1635_BP'], xerr = df_unsat['OH_4500_STD'], yerr = (df_unsat['H2Om_1635_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax[4].scatter(df_unsat1['OH_4500_M'], df_unsat1['H2OT_3550_M']-df_unsat1['H2Om_1635_BP'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15) #, label = 'Fuego 1974')
ax[4].errorbar(df_unsat1['OH_4500_M'], df_unsat1['H2OT_3550_M']-df_unsat1['H2Om_1635_BP'], xerr = df_unsat1['OH_4500_STD'], yerr = (df_unsat1['H2Om_1635_STD']**2 + df_unsat1['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')
leg1 = ax[4].legend(loc = (0.025, 0.87), labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax[4].annotate("E.", xy=(0.035, 0.935), xycoords="axes fraction", fontsize=20, weight='bold')
ax[4].add_artist(leg1)
ax[4].set_xlim([0, 4])
ax[4].set_ylim([0, 4])
ax[4].set_xticks(np.arange(0, 5, 1))
ax[4].set_yticks(np.arange(0, 5, 1))
ax[4].set_ylabel('$\mathregular{H_2O_{t, 3550} - H_2O_{m, 1635}}$ (wt.%)')
ax[4].set_xlabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax[4].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[4].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[4].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value4**2, 2)), xy=(0.035, 0.7675), xycoords="axes fraction", fontsize=12)
ax[4].annotate("CCC="+str(np.round(ccc4, 2)), xy=(0.035, 0.85), xycoords="axes fraction", fontsize=12)
ax[4].annotate("RMSE="+str(np.round(rmse4, 2)), xy=(0.035, 0.81), xycoords="axes fraction", fontsize=12)
ax[4].annotate("m="+str(np.round(slope4, 2)), xy=(0.035, 0.73), xycoords="axes fraction", fontsize=12)
ax[4].annotate("b="+str(np.round(intercept4, 2)), xy=(0.035, 0.69), xycoords="axes fraction", fontsize=12)






slope5, intercept5, r_value5, p_value5, std_err5 = scipy.stats.linregress(df_all['OH_4500_M'], df_all['H2OT_3550_M']-df_all['H2Om_5200_M'])
ccc5 = concordance_correlation_coefficient(df_all['OH_4500_M'], df_all['H2OT_3550_M']-df_all['H2Om_5200_M'])
rmse5 = mean_squared_error(df_all['OH_4500_M'], df_all['H2OT_3550_M']-df_all['H2Om_5200_M'], squared=False)
ax[5].plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax[5].scatter(df_unsat['OH_4500_M'], df_unsat['H2OT_3550_M']-df_unsat['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Fuego')
ax[5].errorbar(df_unsat['OH_4500_M'], df_unsat['H2OT_3550_M']-df_unsat['H2Om_5200_M'], xerr = df_unsat['OH_4500_STD'], yerr = (df_unsat['H2Om_5200_STD']**2 + df_unsat['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax[5].scatter(df_unsat1['OH_4500_M'], df_unsat1['H2OT_3550_M']-df_unsat1['H2Om_5200_M'], s = sz, marker = 'o', c = '#0C7BDC', edgecolors='#171008', linewidth = 0.5, zorder = 15) #, label = 'Fuego 1974')
ax[5].errorbar(df_unsat1['OH_4500_M'], df_unsat1['H2OT_3550_M']-df_unsat1['H2Om_5200_M'], xerr = df_unsat1['OH_4500_STD'], yerr = (df_unsat1['H2Om_5200_STD']**2 + df_unsat1['OH_4500_STD']**2)**(1/2), ls = 'none', elinewidth = 0.5, ecolor = 'k')
leg1 = ax[5].legend(loc = (0.025, 0.87), labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax[5].annotate("F.", xy=(0.035, 0.935), xycoords="axes fraction", fontsize=20, weight='bold')
ax[5].add_artist(leg1)
ax[5].set_xlim([0, 4])
ax[5].set_ylim([0, 4])
ax[5].set_xticks(np.arange(0, 5, 1))
ax[5].set_yticks(np.arange(0, 5, 1))
ax[5].set_ylabel('$\mathregular{H_2O_{t, 3550} - H_2O_{m, 5200}}$ (wt.%)')
ax[5].set_xlabel('$\mathregular{OH^-_{4500}}$ (wt.%)')
ax[5].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[5].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[5].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value5**2, 2)), xy=(0.035, 0.7675), xycoords="axes fraction", fontsize=12)
ax[5].annotate("CCC="+str(np.round(ccc5, 2)), xy=(0.035, 0.85), xycoords="axes fraction", fontsize=12)
ax[5].annotate("RMSE="+str(np.round(rmse5, 2)), xy=(0.035, 0.81), xycoords="axes fraction", fontsize=12)
ax[5].annotate("m="+str(np.round(slope5, 2)), xy=(0.035, 0.73), xycoords="axes fraction", fontsize=12)
ax[5].annotate("b="+str(np.round(intercept5, 2)), xy=(0.035, 0.69), xycoords="axes fraction", fontsize=12)



df_sat, df_unsat = df_filt(df_sims)
df_sat_std, df_unsat_std = df_filt(df_std)
df_sat_std = std_filt(df_sat_std)
df_unsat_std = std_filt(df_unsat_std)

df_unsat_std = df_unsat_std[~df_unsat_std.index.str.startswith('STD_ETF46')]


df_net = pd.concat([df_sat, df_unsat, df_sat_std, df_unsat_std])
df_net = df_net.dropna()

slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(df_net['H2Om_1635_BP'], df_net['H2Om_5200_M'])
ccc1 = concordance_correlation_coefficient(df_net['H2Om_1635_BP'], df_net['H2Om_5200_M'])
rmse1 = mean_squared_error(df_net['H2Om_1635_BP'], df_net['H2Om_5200_M'], squared=False)


sz_sm = 80
sz = 150
ax[1].plot([0,7], [0,7], c = '#171008', linewidth = 1.0)
ax[1].scatter(df_sat['H2Om_1635_BP'], df_sat['H2Om_5200_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15, label = 'Standards')
ax[1].scatter(df_sat['H2Om_1635_BP']+0.005, df_sat['H2Om_5200_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax[1].errorbar(df_sat['H2Om_1635_BP'], df_sat['H2Om_5200_M'], yerr = df_sat['H2Om_5200_STD'], xerr = df_sat['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax[1].scatter(df_unsat['H2Om_1635_BP'], df_unsat['H2Om_5200_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax[1].errorbar(df_unsat['H2Om_1635_BP'], df_unsat['H2Om_5200_M'], yerr = df_unsat['H2Om_5200_STD'], xerr = df_unsat['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax[1].scatter(df_sat_std['H2Om_1635_BP'], df_sat_std['H2Om_5200_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax[1].scatter(df_sat_std['H2Om_1635_BP']+0.005, df_sat_std['H2Om_5200_M'], s = sz_sm, marker = '>', c = '#FFFFFF', ec = '#171008', lw = 0.5, zorder = 20)
ax[1].errorbar(df_sat_std['H2Om_1635_BP'], df_sat_std['H2Om_5200_M'], yerr = df_sat_std['H2Om_5200_STD'], xerr = df_sat_std['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
ax[1].scatter(df_unsat_std['H2Om_1635_BP'], df_unsat_std['H2Om_5200_M'], s = sz, marker = 'o', c = '#E42211', edgecolors='#171008', linewidth = 0.5, zorder = 15)
ax[1].errorbar(df_unsat_std['H2Om_1635_BP'], df_unsat_std['H2Om_5200_M'], yerr = df_unsat_std['H2Om_5200_STD'], xerr = df_unsat_std['H2Om_1635_STD'], ls = 'none', elinewidth = 0.5, ecolor = 'k')
leg1 = ax[1].legend(loc = (0.025, 0.87), labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
sat_symb = ax[1].scatter(np.nan, np.nan, s = sz_sm, marker = '>', ec = '#171008', facecolors='none', lw = 0.5, zorder = 20, label = '$\mathregular{H_2O_{t, 3550}}$ Saturated')
ax[1].annotate("B.", xy=(0.035, 0.935), xycoords="axes fraction", fontsize=20, weight='bold')
ax[1].legend([sat_symb], ['$\mathregular{H_2O_{t, 3550}}$ Saturated'], loc = 'lower right', labelspacing = 0.4, handletextpad = 0.5, handlelength = 1.0, prop={'size': 12}, frameon=False)
ax[1].add_artist(leg1)
ax[1].set_xlim([0, 6])
ax[1].set_ylim([0, 6])
ax[1].set_xlabel('$\mathregular{H_2O_{m, 1635}}$ (wt.%)')
ax[1].set_ylabel('$\mathregular{H_2O_{m, 5200}}$ (wt.%)')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value1**2, 2)), xy=(0.035, 0.7675), xycoords="axes fraction", fontsize=12)
ax[1].annotate("CCC="+str(np.round(ccc1, 2)), xy=(0.035, 0.85), xycoords="axes fraction", fontsize=12)
ax[1].annotate("RMSE="+str(np.round(rmse1, 2)), xy=(0.035, 0.81), xycoords="axes fraction", fontsize=12)
ax[1].annotate("m="+str(np.round(slope1, 2)), xy=(0.035, 0.73), xycoords="axes fraction", fontsize=12)
ax[1].annotate("b="+str(np.round(intercept1, 2)), xy=(0.035, 0.69), xycoords="axes fraction", fontsize=12)

plt.tight_layout()

plt.savefig('NetSpeciation.pdf', bbox_inches='tight', pad_inches = 0.025)


# %%
