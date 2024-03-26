# %% -*- coding: utf-8 -*-

""" Created on January 20, 2023 // @author: Sarah Shi for figures"""
import numpy as np
import pandas as pd 

import os
import glob 
from pathlib import Path
from scipy import signal

from matplotlib import pyplot as plt
from matplotlib import rc, cm

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

# %% 

wn_low = 1250
wn_high = 2400

parent_dir = os.path.split(os.getcwd())[0]

bl_i = pd.read_csv('BLi.csv', index_col = 'Wavenumber')
bl_i = bl_i.loc[wn_low:wn_high]

h2o_co2_bdl = pd.read_csv('H2O_CO2_BDL.csv', index_col = 'Wavenumber')
h2o_co2_bdl = h2o_co2_bdl.loc[wn_low:wn_high]

co2_bdl = pd.read_csv('All_BDL.csv', index_col = 'Wavenumber')
co2_bdl = co2_bdl.loc[wn_low:wn_high]

peak_strip = pd.read_csv('Peak_Stripped.csv', index_col = 'Wavenumber')
peak_strip = peak_strip.loc[wn_low:wn_high]

peak1635 = pd.read_csv('H2Om1635_BL_Removed.csv', index_col = 'Wavenumber')
peak1635 = peak1635.loc[wn_low:wn_high]

BaselinePCA = pd.read_csv(parent_dir + '/Peak_Fit/InputData/Baseline_Avg+PCA.csv', index_col = 'Wavenumber')
BaselinePCA = BaselinePCA.loc[wn_low:wn_high]

H2OPCA = pd.read_csv(parent_dir + '/Peak_Fit/InputData/H2Om1635_PC.csv', index_col = 'Wavenumber')
H2OPCA = H2OPCA.loc[wn_low:wn_high]

H2OPCA_Plot = pd.read_csv(parent_dir + '/Peak_Fit/InputData/H2Om1635_PC_Plotting.csv', index_col = 'Wavenumber')

def rescale(abs, range): 
    abs_m = abs * (range / np.abs(abs.iloc[0] - abs.iloc[-1]))
    abs_mb = abs_m - abs_m.iloc[-1]
    return abs_mb

def rescale_peak(abs, range): 
    abs_m = abs * (range / np.max(abs))
    abs_mb = abs_m - abs_m.iloc[-1]
    return abs_mb

# %% 

BaselinePCA = pd.read_csv(parent_dir + '/Peak_Fit/InputData/Baseline_Avg+PCA.csv', index_col = 'Wavenumber')
BaselinePCA = BaselinePCA.loc[wn_low:wn_high]

h2o_free_scale = h2o_co2_bdl.apply(lambda x: rescale(x, 2))

fig, ax = plt.subplots(3, 2, figsize=(13, 16)) 
ax = ax.flatten()
ax[0].plot(np.nan, np.nan, lw = 0, c = None, label = '')
ax[0].plot(h2o_co2_bdl.index, rescale(bl_i, 2), c = '#171008', lw = 3, label = '$\mathregular{\overline{Baseline_i}}$')
ax[0].plot(h2o_co2_bdl.index, h2o_free_scale, alpha = 0.5)
ax[0].plot(h2o_co2_bdl.index, h2o_free_scale.iloc[:, -1], alpha = 0.5, label = 'Volatiles Below Detection Spectra, n='+str(np.shape(h2o_co2_bdl)[1]))
ax[0].plot(h2o_co2_bdl.index, rescale(bl_i, 2), c = '#171008', lw = 3)
ax[0].annotate("A.", xy=(0.0425, 0.925), xycoords="axes fraction", fontsize=20, weight='bold')
ax[0].legend(loc = (0.03, 0.77), labelspacing=0.05, handletextpad=0.5, handlelength=0.6, prop={'size': 15.5}, frameon=False)
ax[0].set_xlim([1200, 2450])
ax[0].set_ylim([-0.25, 2.25])
ax[0].tick_params(axis="x", direction='in', length=5, pad=6.5, labelbottom=False)
ax[0].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[0].invert_xaxis()

dan_scale = co2_bdl.apply(lambda x: rescale(x, 2))
common_columns = h2o_co2_bdl.columns.intersection(dan_scale.columns)
dan_scale_lim = dan_scale.drop(columns=common_columns)

ax[1].plot(np.nan, np.nan, lw = 0, c = None, label = '')
ax[1].plot(bl_i.index, rescale(bl_i, 2), c = '#171008', lw = 3, label = '$\mathregular{\overline{Baseline_i}}$')
ax[1].plot(co2_bdl.index, dan_scale_lim, alpha = 0.5)
ax[1].plot(co2_bdl.index, dan_scale_lim.iloc[:, -1], alpha = 0.5, label = '$\mathregular{CO_{3}^{2-}}$ Below Detection Spectra, n='+str(np.shape(dan_scale_lim)[1]-1))
ax[1].plot(h2o_co2_bdl.index, rescale(bl_i, 2), c = '#171008', lw = 3)
ax[1].annotate("B.", xy=(0.0425, 0.925), xycoords="axes fraction", fontsize=20, weight='bold')
ax[1].legend(loc = (0.03, 0.77), labelspacing=0.05, handletextpad=0.5, handlelength=0.6, prop={'size': 15.5}, frameon=False)
ax[1].set_xlim([1200, 2450])
ax[1].set_ylim([-0.25, 2.25])
ax[1].tick_params(axis="x", direction='in', length=5, pad=6.5, labelbottom=False)
ax[1].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[1].invert_xaxis()

peak1635_scale = peak1635.apply(lambda x: rescale_peak(x, 2))
ax[2].plot(np.nan, np.nan, lw = 0, c = None, label = '')
ax[2].plot(H2OPCA.index, rescale_peak(H2OPCA['1635_Peak_Mean'], 2), c = '#171008', lw = 3, label = '$\overline{\mathregular{H_2O_{m, 1635}}}$')
ax[2].plot(peak1635_scale.index, peak1635_scale, alpha = 0.5)
ax[2].plot(peak1635_scale.index, peak1635_scale.iloc[:, -1], alpha = 0.5, label = '$\mathregular{H_2O_{m, 1635}}$ Spectra, n='+str(np.shape(peak1635)[1]+1))
ax[2].plot(H2OPCA.index, rescale_peak(H2OPCA['1635_Peak_Mean'], 2), c = '#171008', lw = 2)
ax[2].annotate("C.", xy=(0.0425, 0.925), xycoords="axes fraction", fontsize=20, weight='bold')
ax[2].legend(loc = (0.03, 0.77), labelspacing=0.05, handletextpad=0.5, handlelength=0.6, prop={'size': 15.5}, frameon=False)
ax[2].set_xlim([1200, 2450])
ax[2].set_ylim([-0.25, 2.25])
ax[2].tick_params(axis="x", direction='in', length=5, pad=6.5, labelbottom=False)
ax[2].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[2].invert_xaxis()

devol_scale = peak_strip.apply(lambda x: rescale(x, 2))
ax[3].plot(np.nan, np.nan, lw = 0, c = None, label = '')
ax[3].plot(BaselinePCA.index, rescale(BaselinePCA.Average_Baseline, 2), c = '#171008', lw = 3, label = '$\mathregular{\overline{Baseline}}$')
ax[3].plot(devol_scale.index, devol_scale, alpha = 0.5)
ax[3].plot(devol_scale.index, devol_scale.iloc[:, -1], alpha = 0.5, label = 'Peak Stripped Spectra, n='+str(np.shape(peak_strip)[1]))
ax[3].plot(BaselinePCA.index, rescale(BaselinePCA.Average_Baseline, 2), c = '#171008', lw = 2)
ax[3].annotate("D.", xy=(0.0425, 0.925), xycoords="axes fraction", fontsize=20, weight='bold')
ax[3].legend(loc = (0.03, 0.77), labelspacing=0.05, handletextpad=0.5, handlelength=0.6, prop={'size': 15.5}, frameon=False)
ax[3].set_xlim([1200, 2450])
ax[3].set_ylim([-0.25, 2.25])
ax[3].tick_params(axis="x", direction='in', length=5, pad=6.5, labelbottom = False)
ax[3].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[3].invert_xaxis()

BaselinePCA = pd.read_csv(parent_dir + '/Peak_Fit/InputData/Baseline_Avg+PCA.csv', index_col = 'Wavenumber')
ax[4].plot(np.nan, np.nan, lw = 0, c = None, label = '')
ax[4].plot(BaselinePCA.index, BaselinePCA.Average_Baseline, c = '#171008', lw = 3, label = '$\mathregular{\overline{Baseline}}$')
ax[4].plot(BaselinePCA.index, BaselinePCA.PCA_1, c = '#0C7BDC', lw = 2, label = '$\mathregular{\overline{Baseline}_{PC1,}}$79% exp var')
ax[4].plot(BaselinePCA.index, BaselinePCA.PCA_2, c = '#E42211', lw = 2, label = '$\mathregular{\overline{Baseline}_{PC2,}}$15% exp var')
ax[4].plot(BaselinePCA.index, BaselinePCA.PCA_3, c = '#5DB147', lw = 2, label = '$\mathregular{\overline{Baseline}_{PC3,}}$ 4% exp var')
ax[4].plot(BaselinePCA.index, BaselinePCA.PCA_4, c = '#F9C300', lw = 2, label = '$\mathregular{\overline{Baseline}_{PC4,}}$ 1% exp var')
ax[4].annotate("E.", xy=(0.0425, 0.925), xycoords="axes fraction", fontsize=20, weight='bold')
ax[4].legend(loc = (0.03, 0.55), labelspacing=0.05, handletextpad=0.5, handlelength=0.6, prop={'size': 15.5}, frameon=False)
ax[4].set_xlim([1200, 2450])
ax[4].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[4].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[4].invert_xaxis()

ax[5].plot(np.nan, np.nan, lw = 0, c = None, label = '')
ax[5].plot(H2OPCA.index, H2OPCA['1635_Peak_Mean'], c = '#171008', lw = 3, label = '$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[5].plot(H2OPCA.index, H2OPCA['1635_Peak_PC1'], c = '#0C7BDC', lw = 2, label = '$\mathregular{\overline{H_2O_{m, 1635}}_{PC1,}}$65% exp var')
ax[5].plot(H2OPCA.index, H2OPCA['1635_Peak_PC2'], c = '#E42211', lw = 2, label = '$\mathregular{\overline{H_2O_{m, 1635}}_{PC2,}}$16% exp var')
ax[5].annotate("F.", xy=(0.0425, 0.925), xycoords="axes fraction", fontsize=20, weight='bold')
ax[5].legend(loc = (0.03, 0.7), labelspacing=0.05, handletextpad=0.5, handlelength=0.6, prop={'size': 15.5}, frameon=False)
ax[5].set_xlim([1200, 2450])
ax[5].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[5].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[5].invert_xaxis()

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y=0.03)
fig.supylabel('Absorbance', x = 0.05)
plt.tight_layout()
plt.savefig('AllBaselines1.pdf', bbox_inches='tight', pad_inches = 0.025)

# %%

# %%



n1 = 15
sz = 150
fig, ax=plt.subplots(2, 2, figsize=(13, 13))
ax=ax.flatten()
ax[0].plot(BaselinePCA.index, BaselinePCA.Average_Baseline, c='#171008', lw=2, label='$\mathregular{\overline{Baseline}}$')
ax[0].plot(BaselinePCA.loc[0::n1].index, BaselinePCA.loc[0::n1].Average_Baseline + BaselinePCA.loc[0::n1].PCA_1, c='#0C7BDC', marker='+', markersize=7.5, linestyle='None', label='$\mathregular{\overline{Baseline}}$ + $\mathregular{\overline{Baseline}_{PC1}}$', zorder=20)
ax[0].plot(BaselinePCA.index, BaselinePCA.Average_Baseline - BaselinePCA.PCA_1, c='#0C7BDC', lw=2, ls='--', label='$\mathregular{\overline{Baseline}}$ \N{MINUS SIGN} $\mathregular{\overline{Baseline}_{PC1}}$',)

ax[1].plot(BaselinePCA.index, BaselinePCA.Average_Baseline, c='#171008', lw=2, label='$\mathregular{\overline{Baseline}}$')
ax[1].plot(BaselinePCA.loc[0::n1].index, BaselinePCA.loc[0::n1].Average_Baseline + BaselinePCA.loc[0::n1].PCA_2, c='#E42211', marker='+', markersize=7.5, linestyle='None', label='$\mathregular{\overline{Baseline}}$ + $\mathregular{\overline{Baseline}_{PC2}}$', zorder=15)
ax[1].plot(BaselinePCA.index, BaselinePCA.Average_Baseline - BaselinePCA.PCA_2, c='#E42211', lw=2, ls='--', label='$\mathregular{\overline{Baseline}}$ \N{MINUS SIGN} $\mathregular{\overline{Baseline}_{PC2}}$',)

ax[2].plot(BaselinePCA.index, BaselinePCA.Average_Baseline, c='#171008', lw=2, label='$\mathregular{\overline{Baseline}}$')
ax[2].plot(BaselinePCA.loc[0::n1].index, BaselinePCA.loc[0::n1].Average_Baseline + BaselinePCA.loc[0::n1].PCA_3, c='#5DB147', marker='+', markersize=7.5, linestyle='None', label='$\mathregular{\overline{Baseline}}$ + $\mathregular{\overline{Baseline}_{PC3}}$', zorder=10)
ax[2].plot(BaselinePCA.index, BaselinePCA.Average_Baseline - BaselinePCA.PCA_3, c='#5DB147', lw=2, ls='--', label='$\mathregular{\overline{Baseline}}$ \N{MINUS SIGN} $\mathregular{\overline{Baseline}_{PC3}}$',)

ax[3].plot(BaselinePCA.index, BaselinePCA.Average_Baseline, c='#171008', lw=2, label='$\mathregular{\overline{Baseline}}$')
ax[3].plot(BaselinePCA.loc[0::n1].index, BaselinePCA.loc[0::n1].Average_Baseline + BaselinePCA.loc[0::n1].PCA_4, c='#F9C300', marker='+', markersize=7.5, linestyle='None', label='$\mathregular{\overline{Baseline}}$ + $\mathregular{\overline{Baseline}_{PC4}}$', zorder=5)
ax[3].plot(BaselinePCA.index, BaselinePCA.Average_Baseline - BaselinePCA.PCA_4, c='#F9C300', lw=2, ls='--', label='$\mathregular{\overline{Baseline}}$ \N{MINUS SIGN} $\mathregular{\overline{Baseline}_{PC4}}$',)

ax[0].set_xlim([1125, 2475])
ax[1].set_xlim([1125, 2475])
ax[2].set_xlim([1125, 2475])
ax[3].set_xlim([1125, 2475])

ax[0].set_ylim([-0.6, 0.6])
ax[1].set_ylim([-0.6, 0.6])
ax[2].set_ylim([-0.6, 0.6])
ax[3].set_ylim([-0.6, 0.6])

desired_ticks = [0, 0.2, 0.4]
for axis in ax:
    axis.set_yticks(desired_ticks)

ax[0].tick_params(axis="x", direction='in', length=5, pad=6.5, labelbottom=False)
ax[1].tick_params(axis="x", direction='in', length=5, pad=6.5, labelbottom=False)
ax[2].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[3].tick_params(axis="x", direction='in', length=5, pad=6.5)

ax[0].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad=6.5, labelleft=False)
ax[2].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[3].tick_params(axis="y", direction='in', length=5, pad=6.5, labelleft=False)

ax[0].invert_xaxis()
ax[1].invert_xaxis()
ax[2].invert_xaxis()
ax[3].invert_xaxis()

ax[0].annotate("A.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[0].legend(loc=(0.025, 0.7), labelspacing=0.3, handletextpad=0.5, handlelength=1.0, prop={'size': 18}, frameon=False)
ax[1].annotate("B.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[1].legend(loc=(0.025, 0.7), labelspacing=0.3, handletextpad=0.5, handlelength=1.0, prop={'size': 18}, frameon=False)
ax[2].annotate("C.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[2].legend(loc=(0.025, 0.7), labelspacing=0.3, handletextpad=0.5, handlelength=1.0, prop={'size': 18}, frameon=False)
ax[3].annotate("D.", xy=(0.0375, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[3].legend(loc=(0.025, 0.7), labelspacing=0.3, handletextpad=0.5, handlelength=1.0, prop={'size': 18}, frameon=False)

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y=0.04)
fig.supylabel('Absorbance', x=0.05)

plt.tight_layout()
# plt.savefig('BL+PCVectors_Subplot.pdf', bbox_inches='tight', pad_inches = 0.025)

# %% 


fig, ax=plt.subplots(2, 2, figsize=(13, 13))
ax=ax.flatten()
ax[0].plot(H2OPCA.index, H2OPCA['1635_Peak_Mean'], c='#171008', lw=2, label='$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[0].plot(H2OPCA_Plot.index, H2OPCA_Plot['1635_Peak_Mean'] + H2OPCA_Plot['1635_Peak_PC1'], c='#0C7BDC', marker='+', markersize=7.5, linestyle='None', label='$\mathregular{\overline{H_2O_{m, 1635}}}$ + $\mathregular{\overline{H_2O_{m, 1635}}_{PC1}}$', zorder=20)

ax[1].plot(H2OPCA.index, H2OPCA['1635_Peak_Mean'], c='#171008', lw=2, label='$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[1].plot(H2OPCA.index, H2OPCA['1635_Peak_Mean'] - H2OPCA['1635_Peak_PC1'], c='#0C7BDC', lw=2.0, ls='--', label='$\mathregular{\overline{H_2O_{m, 1635}}}$ - $\mathregular{\overline{H_2O_{m, 1635}}_{PC1}}$', zorder=20)

ax[2].plot(H2OPCA.index, H2OPCA['1635_Peak_Mean'], c='#171008', lw=2, label='$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[2].plot(H2OPCA_Plot.index, H2OPCA_Plot['1635_Peak_Mean'] + H2OPCA_Plot['1635_Peak_PC2'], c='#E42211', marker='+', markersize=7.5, linestyle='None', label='$\mathregular{\overline{H_2O_{m, 1635}}}$ + $\mathregular{\overline{H_2O_{m, 1635}}_{PC2}}$', zorder=20)

ax[3].plot(H2OPCA.index, H2OPCA['1635_Peak_Mean'], c='#171008', lw=2, label='$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[3].plot(H2OPCA.index, H2OPCA['1635_Peak_Mean'] - H2OPCA['1635_Peak_PC2'], c='#E42211', lw=2.0, ls='--', label='$\mathregular{\overline{H_2O_{m, 1635}}}$ - $\mathregular{\overline{H_2O_{m, 1635}}_{PC2}}$', zorder=20)

ax[0].annotate("A.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[0].legend(loc=(0.025, 0.78), labelspacing=0.2, handletextpad=0.4, handlelength=0.4, prop={'size': 18}, frameon=False)
ax[1].annotate("B.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[1].legend(loc=(0.025, 0.78), labelspacing=0.2, handletextpad=0.4, handlelength=0.4, prop={'size': 18}, frameon=False)
ax[2].annotate("C.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[2].legend(loc=(0.025, 0.78), labelspacing=0.2, handletextpad=0.4, handlelength=0.4, prop={'size': 18}, frameon=False)
ax[3].annotate("D.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[3].legend(loc=(0.025, 0.78), labelspacing=0.2, handletextpad=0.4, handlelength=0.4, prop={'size': 18}, frameon=False)

ax[0].set_xlim([1125, 2475])
ax[1].set_xlim([1125, 2475])
ax[2].set_xlim([1125, 2475])
ax[3].set_xlim([1125, 2475])
ax[0].set_ylim([-0.15, 1.15])
ax[1].set_ylim([-0.15, 1.15])
ax[2].set_ylim([-0.15, 1.15])
ax[3].set_ylim([-0.15, 1.15])

ax[0].tick_params(axis="x", direction='in', length=5, pad=6.5, labelbottom=False)
ax[1].tick_params(axis="x", direction='in', length=5, pad=6.5, labelbottom=False)
ax[2].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[3].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad=6.5, labelleft=False)
ax[2].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[3].tick_params(axis="y", direction='in', length=5, pad=6.5, labelleft=False)
ax[0].invert_xaxis()
ax[1].invert_xaxis()
ax[2].invert_xaxis()
ax[3].invert_xaxis()

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y=0.04)
fig.supylabel('Absorbance', x=0.05)
plt.tight_layout()
# plt.savefig('H2Om1635+PCVectors_Subplot+1.pdf', bbox_inches='tight', pad_inches = 0.025)

# %%
