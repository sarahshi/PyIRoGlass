# %% -*- coding: utf-8 -*-
""" Created on August 1, 2021 // @author: Sarah Shi """

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
import matplotlib.gridspec as gridspec

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


# %% PCA Component Plotting

BaselinePCA = pd.read_csv('./InputData/Baseline_Avg+PCA.csv')
H2OPCA = pd.read_csv('./InputData/Water_Peak_1635_All.csv')
H2OPCA_Plot = pd.read_csv('./InputData/Water_Peak_1635_Plotting.csv')

sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline, c = '#171008', lw = 2, label = 'Baseline')
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.PCA_1, c = '#0C7BDC', lw = 2, label = 'PC1')
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.PCA_2, c = '#E42211', lw = 2, label = 'PC2')
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.PCA_3, c = '#5DB147', lw = 2, label = 'PC3')
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.PCA_4, c = '#F9C300', lw = 2, label = 'PC4')
ax.legend(loc = 'upper left', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax.set_xlim([1125, 2475])
ax.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax.set_ylabel('Absorbance')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax.invert_xaxis()
plt.tight_layout()
plt.savefig('BL_PCVectors.pdf')

fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak, c = '#171008', lw = 2, label = '$\mathregular{H_2O_{m, 1635}}$')
ax.plot(H2OPCA.Wavenumber, H2OPCA['1630_Peak_PCA_1'], c = '#0C7BDC', lw = 2, label = 'PC1')
ax.plot(H2OPCA.Wavenumber, H2OPCA['1630_Peak_PCA_2'], c = '#E42211', lw = 2, label = 'PC2')
ax.legend(loc = 'upper left', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax.set_xlim([1125, 2475])
ax.set_ylim([-0.4, 1.2])
ax.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax.set_ylabel('Absorbance')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax.invert_xaxis()
plt.tight_layout()
# plt.savefig('H2Om1635_PCVectors.pdf')

# %% 

BaselinePCA = pd.read_csv('./InputData/Baseline_Avg+PCA.csv')
H2OPCA = pd.read_csv('./InputData/Water_Peak_1635_All.csv')
H2OPCA_Plot = pd.read_csv('./InputData/Water_Peak_1635_Plotting.csv')

sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline, c = '#171008', lw = 2, label = 'Baseline')
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline + BaselinePCA.PCA_1, c = '#0C7BDC', lw = 2, label = 'Baseline \N{PLUS-MINUS SIGN} PC1', zorder = 20)
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline - BaselinePCA.PCA_1, c = '#0C7BDC', lw = 2)
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline + BaselinePCA.PCA_2, c = '#E42211', lw = 2, label = 'Baseline \N{PLUS-MINUS SIGN} PC2', zorder = 15)
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline - BaselinePCA.PCA_2, c = '#E42211', lw = 2)
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline + BaselinePCA.PCA_3, c = '#5DB147', lw = 2, label = 'Baseline \N{PLUS-MINUS SIGN} PC3', zorder = 10)
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline - BaselinePCA.PCA_3, c = '#5DB147', lw = 2)
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline + BaselinePCA.PCA_4, c = '#F9C300', lw = 2, label = 'Baseline \N{PLUS-MINUS SIGN} PC4', zorder = 5)
ax.plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline - BaselinePCA.PCA_4, c = '#F9C300', lw = 2)
ax.legend(loc = 'upper left', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax.set_xlim([1125, 2475])
ax.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax.set_ylabel('Absorbance')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax.invert_xaxis()
plt.tight_layout()
# plt.savefig('BL+PCVectors.pdf')

# %%


BaselinePCA = pd.read_csv('./InputData/Baseline_Avg+PCA.csv')
H2OPCA = pd.read_csv('./InputData/Water_Peak_1635_All.csv')
H2OPCA_Plot = pd.read_csv('./InputData/Water_Peak_1635_Plotting.csv')

n1 = 15
sz = 150
fig, ax = plt.subplots(2, 2, figsize = (13, 13))
ax = ax.flatten()
ax[0].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline, c = '#171008', lw = 2, label = '$\mathregular{\overline{Baseline}}$')
ax[0].plot(BaselinePCA.Wavenumber[0::n1], BaselinePCA.Average_Baseline[0::n1] + BaselinePCA.PCA_1[0::n1], c = '#0C7BDC', marker = '+', markersize = 7.5, linestyle = 'None', label = '$\mathregular{\overline{Baseline}}$ + PC1', zorder = 20)
ax[0].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline - BaselinePCA.PCA_1, c = '#0C7BDC', lw = 2, ls = '--', label = '$\mathregular{\overline{Baseline}}$ \N{MINUS SIGN} PC1',)

ax[1].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline, c = '#171008', lw = 2, label = '$\mathregular{\overline{Baseline}}$')
ax[1].plot(BaselinePCA.Wavenumber[0::n1], BaselinePCA.Average_Baseline[0::n1] + BaselinePCA.PCA_2[0::n1], c = '#E42211', marker = '+', markersize = 7.5, linestyle = 'None', label = '$\mathregular{\overline{Baseline}}$ + PC2', zorder = 15)
ax[1].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline - BaselinePCA.PCA_2, c = '#E42211', lw = 2, ls = '--', label = '$\mathregular{\overline{Baseline}}$ \N{MINUS SIGN} PC2',)

ax[2].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline, c = '#171008', lw = 2, label = '$\mathregular{\overline{Baseline}}$')
ax[2].plot(BaselinePCA.Wavenumber[0::n1], BaselinePCA.Average_Baseline[0::n1] + BaselinePCA.PCA_3[0::n1], c = '#5DB147', marker = '+', markersize = 7.5, linestyle = 'None', label = '$\mathregular{\overline{Baseline}}$ + PC3', zorder = 10)
ax[2].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline - BaselinePCA.PCA_3, c = '#5DB147', lw = 2, ls = '--', label = '$\mathregular{\overline{Baseline}}$ \N{MINUS SIGN} PC3',)

ax[3].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline, c = '#171008', lw = 2, label = '$\mathregular{\overline{Baseline}}$')
ax[3].plot(BaselinePCA.Wavenumber[0::n1], BaselinePCA.Average_Baseline[0::n1] + BaselinePCA.PCA_4[0::n1], c = '#F9C300', marker = '+', markersize = 7.5, linestyle = 'None', label = '$\mathregular{\overline{Baseline}}$ + PC4', zorder = 5)
ax[3].plot(BaselinePCA.Wavenumber, BaselinePCA.Average_Baseline - BaselinePCA.PCA_4, c = '#F9C300', lw = 2, ls = '--', label = '$\mathregular{\overline{Baseline}}$ \N{MINUS SIGN} PC4',)

ax[0].set_xlim([1125, 2475])
ax[1].set_xlim([1125, 2475])
ax[2].set_xlim([1125, 2475])
ax[3].set_xlim([1125, 2475])

ax[0].set_ylim([-0.6, 0.6])
ax[1].set_ylim([-0.6, 0.6])
ax[2].set_ylim([-0.6, 0.6])
ax[3].set_ylim([-0.6, 0.6])

ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5, labelbottom = False)
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5, labelbottom = False)
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[3].tick_params(axis="x", direction='in', length=5, pad = 6.5)

ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5, labelleft = False)
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[3].tick_params(axis="y", direction='in', length=5, pad = 6.5, labelleft = False)

ax[0].invert_xaxis()
ax[1].invert_xaxis()
ax[2].invert_xaxis()
ax[3].invert_xaxis()

ax[0].annotate("A.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[0].legend(loc = (0.025, 0.74), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[1].annotate("B.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[1].legend(loc = (0.025, 0.74), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[2].annotate("C.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[2].legend(loc = (0.025, 0.74), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[3].annotate("D.", xy=(0.0375, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[3].legend(loc = (0.025, 0.74), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y = 0.04)
fig.supylabel('Absorbance', x = 0.05)

plt.tight_layout()
# plt.savefig('BL+PCVectors_Subplot.pdf', bbox_inches='tight', pad_inches = 0.025)


# %%

fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak, c = '#171008', lw = 2, label = '$\mathregular{H_2O_{m, 1635}}$')
ax.plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak + H2OPCA['1630_Peak_PCA_1'], c = '#0C7BDC', lw = 2, label = '$\mathregular{H_2O_{m, 1635}}$ \N{PLUS-MINUS SIGN} PC1', zorder = 20)
ax.plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak - H2OPCA['1630_Peak_PCA_1'], c = '#0C7BDC', lw = 2, zorder = 20)
ax.plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak + H2OPCA['1630_Peak_PCA_2'], c = '#E42211', lw = 2, label = '$\mathregular{H_2O_{m, 1635}}$ \N{PLUS-MINUS SIGN} PC2', zorder = 10)
ax.plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak - H2OPCA['1630_Peak_PCA_2'], c = '#E42211', lw = 2, zorder = 10)
ax.legend(loc = 'upper left', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax.set_xlim([1125, 2475])
ax.set_ylim([-0.4, 1.2])
ax.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax.set_ylabel('Absorbance')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax.invert_xaxis()
plt.tight_layout()
# plt.savefig('H2Om1635+PCVectors.pdf')

# %%

H2OPCA_Plot = pd.read_csv('./InputData/Water_Peak_1635_Plotting.csv')

fig, ax = plt.subplots(1, 2, figsize = (14, 7))
ax = ax.flatten()
ax[0].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak, c = '#171008', lw = 2, label = '$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[0].plot(H2OPCA_Plot.Wavenumber, H2OPCA_Plot.Average_1630_Peak + H2OPCA_Plot['1630_Peak_PCA_1'], c = '#0C7BDC', marker = '+', markersize = 7.5,  linestyle = 'None', label = '$\mathregular{\overline{H_2O_{m, 1635}}}$ + PC1', zorder = 20)
ax[0].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak - H2OPCA['1630_Peak_PCA_1'], c = '#0C7BDC',  lw = 1.5, ls='--', label = '$\mathregular{\overline{H_2O_{m, 1635}}}$ + PC1', zorder = 20)

ax[1].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak, c = '#171008', lw = 2, label = '$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[1].plot(H2OPCA_Plot.Wavenumber, H2OPCA_Plot.Average_1630_Peak + H2OPCA_Plot['1630_Peak_PCA_2'], c = '#E42211', marker = '+', markersize = 7.5,  linestyle = 'None', label = '$\mathregular{\overline{H_2O_{m, 1635}}}$ + PC2', zorder = 20)
ax[1].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak - H2OPCA['1630_Peak_PCA_2'], c = '#E42211',  lw = 1.5, ls='--', label = '$\mathregular{\overline{H_2O_{m, 1635}}}$ + PC2', zorder = 20)
ax[0].legend(loc = 'upper left', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[1].legend(loc = 'upper left', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)

ax[0].set_xlim([1125, 2475])
ax[1].set_xlim([1125, 2475])
ax[0].set_ylim([-0.15, 1.15])
ax[1].set_ylim([-0.15, 1.15])

ax[0].set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax[1].set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax[0].set_ylabel('Absorbance')
ax[0].set_title('A.', fontweight='bold')
ax[1].set_title('B.', fontweight='bold')

ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[0].invert_xaxis()
ax[1].invert_xaxis()
plt.tight_layout()
# plt.savefig('H2Om1635+PCVectors_Subplot.pdf')

# %% 


H2OPCA_Plot = pd.read_csv('./InputData/Water_Peak_1635_Plotting.csv')

fig, ax = plt.subplots(2, 2, figsize = (13, 13))
ax = ax.flatten()
ax[0].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak, c = '#171008', lw = 2, label = '$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[0].plot(H2OPCA_Plot.Wavenumber, H2OPCA_Plot.Average_1630_Peak + H2OPCA_Plot['1630_Peak_PCA_1'], c = '#0C7BDC', marker = '+', markersize = 7.5,  linestyle = 'None', label = '$\mathregular{\overline{H_2O_{m, 1635}}}$ + PC1', zorder = 20)

ax[1].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak, c = '#171008', lw = 2, label = '$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[1].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak - H2OPCA['1630_Peak_PCA_1'], c = '#0C7BDC',  lw = 2.0, ls='--', label = '$\mathregular{\overline{H_2O_{m, 1635}}}$ - PC1', zorder = 20)

ax[2].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak, c = '#171008', lw = 2, label = '$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[2].plot(H2OPCA_Plot.Wavenumber, H2OPCA_Plot.Average_1630_Peak + H2OPCA_Plot['1630_Peak_PCA_2'], c = '#E42211', marker = '+', markersize = 7.5,  linestyle = 'None', label = '$\mathregular{\overline{H_2O_{m, 1635}}}$ + PC2', zorder = 20)

ax[3].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak, c = '#171008', lw = 2, label = '$\mathregular{\overline{H_2O_{m, 1635}}}$')
ax[3].plot(H2OPCA.Wavenumber, H2OPCA.Average_1630_Peak - H2OPCA['1630_Peak_PCA_2'], c = '#E42211',  lw = 2.0, ls='--', label = '$\mathregular{\overline{H_2O_{m, 1635}}}$ - PC2', zorder = 20)

ax[0].annotate("A.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[0].legend(loc = (0.025, 0.79), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[1].annotate("B.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[1].legend(loc = (0.025, 0.79), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[2].annotate("C.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[2].legend(loc = (0.025, 0.79), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[3].annotate("D.", xy=(0.04, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax[3].legend(loc = (0.025, 0.79), labelspacing = 0.3, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)

ax[0].set_xlim([1125, 2475])
ax[1].set_xlim([1125, 2475])
ax[2].set_xlim([1125, 2475])
ax[3].set_xlim([1125, 2475])
ax[0].set_ylim([-0.15, 1.15])
ax[1].set_ylim([-0.15, 1.15])
ax[2].set_ylim([-0.15, 1.15])
ax[3].set_ylim([-0.15, 1.15])

ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5, labelbottom = False)
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5, labelbottom = False)
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[3].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5, labelleft = False)
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[3].tick_params(axis="y", direction='in', length=5, pad = 6.5, labelleft = False)
ax[0].invert_xaxis()
ax[1].invert_xaxis()
ax[2].invert_xaxis()
ax[3].invert_xaxis()

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y = 0.04)
fig.supylabel('Absorbance', x = 0.05)
plt.tight_layout()
# plt.savefig('H2Om1635+PCVectors_Subplot+.pdf', bbox_inches='tight', pad_inches = 0.025)


# %%
# %% 

spec = pd.read_csv('./InputData/AC4_OL49_021920_30x30_H2O_a.csv')
x = pd.read_csv('./InputData/x.csv') 

data_H2O5200_1 = pd.read_csv('./InputData/data_H2O5200_1.csv')
data_H2O5200_2 = pd.read_csv('./InputData/data_H2O5200_2.csv')
data_H2O5200_3 = pd.read_csv('./InputData/data_H2O5200_3.csv')

data_H2O4500_1 = pd.read_csv('./InputData/data_H2O4500_1.csv')
data_H2O4500_2 = pd.read_csv('./InputData/data_H2O4500_2.csv')
data_H2O4500_3 = pd.read_csv('./InputData/data_H2O4500_3.csv')

krige_output_5200_1 = pd.read_csv('./InputData/krige_output_5200_1.csv')
krige_output_5200_2 = pd.read_csv('./InputData/krige_output_5200_2.csv')
krige_output_5200_3 = pd.read_csv('./InputData/krige_output_5200_3.csv')

krige_output_4500_1 = pd.read_csv('./InputData/krige_output_4500_1.csv')
krige_output_4500_2 = pd.read_csv('./InputData/krige_output_4500_2.csv')
krige_output_4500_3 = pd.read_csv('./InputData/krige_output_4500_3.csv')

plot_output_3550_1 = pd.read_csv('./InputData/plot_output_3550_1.csv') 
plot_output_3550_2 = pd.read_csv('./InputData/plot_output_3550_2.csv') 
plot_output_3550_3 = pd.read_csv('./InputData/plot_output_3550_3.csv') 

data_H2O3550_1 = pd.read_csv('./InputData/data_H2O3550_1.csv') 
data_H2O3550_2 = pd.read_csv('./InputData/data_H2O3550_2.csv') 
data_H2O3550_3 = pd.read_csv('./InputData/data_H2O3550_3.csv') 

Baseline_Solve_BP = pd.read_csv('./InputData/Baseline_Solve_BP.csv') 
Baseline_Array_Plot = pd.read_csv('./InputData/Baseline_Array_Plot.csv').to_numpy()
linearray = pd.read_csv('./InputData/linearray.csv') 

H1635_BP = pd.read_csv('./InputData/H1635_BP.csv') 
CO2P1515_BP = pd.read_csv('./InputData/CO2P1515_BP.csv') 
CO2P1430_BP = pd.read_csv('./InputData/CO2P1430_BP.csv') 
carbonate = pd.read_csv('./InputData/carbonate.csv') 

# %% NIR Peak Plotting

sz = 150
fig, ax = plt.subplots(2, 1, figsize = (8, 8))
ax = ax.flatten()
ax[0].plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax[0].plot(data_H2O5200_1.Wavenumber, data_H2O5200_1.Absorbance_Hat, c = '#0C7BDC', lw = 1, label = 'Median Filtered Peak')
ax[0].plot(data_H2O5200_2.Wavenumber, data_H2O5200_2.Absorbance_Hat, c = '#0C7BDC', lw = 1)
ax[0].plot(data_H2O5200_3.Wavenumber, data_H2O5200_3.Absorbance_Hat, c = '#0C7BDC', lw = 1)
ax[0].plot(data_H2O5200_1.Wavenumber, data_H2O5200_1.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1, label = 'ALS Baselines')
ax[0].plot(data_H2O5200_2.Wavenumber, data_H2O5200_2.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)
ax[0].plot(data_H2O5200_3.Wavenumber, data_H2O5200_3.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)

ax[0].plot(data_H2O4500_1.Wavenumber, data_H2O4500_1.Absorbance_Hat, c = '#5DB147', lw = 1)
ax[0].plot(data_H2O4500_2.Wavenumber, data_H2O4500_2.Absorbance_Hat, c = '#5DB147', lw = 1)
ax[0].plot(data_H2O4500_3.Wavenumber, data_H2O4500_3.Absorbance_Hat, c = '#5DB147', lw = 1)
ax[0].plot(data_H2O4500_1.Wavenumber, data_H2O4500_1.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)
ax[0].plot(data_H2O4500_2.Wavenumber, data_H2O4500_2.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)
ax[0].plot(data_H2O4500_3.Wavenumber, data_H2O4500_3.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)
ax[0].text(5150, 0.6625, '$\mathregular{H_2O_{m, 5200}}$', ha = 'center', fontsize = 16)
ax[0].text(4485, 0.5225, '$\mathregular{OH^-_{4500}}$', ha = 'center', fontsize = 16)

ax[0].legend(loc = 'lower left', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[0].set_xlim([4200, 5400])
ax[0].set_ylim([0.4, 0.7])
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[0].axes.xaxis.set_ticklabels([])
ax[0].invert_xaxis()

ax[1].plot(data_H2O5200_1.Wavenumber, data_H2O5200_1['Subtracted_Peak'] - np.min(krige_output_5200_1.Absorbance), c = 'k', lw = 1, label = 'Baseline Subtracted Peak')
ax[1].plot(data_H2O5200_2.Wavenumber, data_H2O5200_2['Subtracted_Peak'] - np.min(krige_output_5200_2.Absorbance), c = 'k', lw = 1)
ax[1].plot(data_H2O5200_3.Wavenumber, data_H2O5200_3['Subtracted_Peak'] - np.min(krige_output_5200_3.Absorbance), c = 'k', lw = 1)
ax[1].plot(krige_output_5200_1.Wavenumber, krige_output_5200_1.Absorbance - np.min(krige_output_5200_1.Absorbance), c = '#0C7BDC', lw = 2, label = 'Gaussian Kriged Peak')
ax[1].plot(krige_output_5200_2.Wavenumber, krige_output_5200_2.Absorbance - np.min(krige_output_5200_2.Absorbance), c = '#074984', lw = 2)
ax[1].plot(krige_output_5200_3.Wavenumber, krige_output_5200_3.Absorbance - np.min(krige_output_5200_3.Absorbance), c = '#6DAFEA', lw = 2)

ax[1].plot(data_H2O4500_1.Wavenumber, data_H2O4500_1['Subtracted_Peak'] - np.min(krige_output_4500_1.Absorbance), c = 'k', lw = 1)
ax[1].plot(data_H2O4500_2.Wavenumber, data_H2O4500_2['Subtracted_Peak'] - np.min(krige_output_4500_2.Absorbance), c = 'k', lw = 1)
ax[1].plot(data_H2O4500_3.Wavenumber, data_H2O4500_3['Subtracted_Peak'] - np.min(krige_output_4500_3.Absorbance), c = 'k', lw = 1)
ax[1].plot(krige_output_4500_1.Wavenumber, krige_output_4500_1.Absorbance - np.min(krige_output_4500_1.Absorbance), c = '#417B31', lw = 2)
ax[1].plot(krige_output_4500_2.Wavenumber, krige_output_4500_2.Absorbance - np.min(krige_output_4500_2.Absorbance), c = '#5DB147', lw = 2) 
ax[1].plot(krige_output_4500_3.Wavenumber, krige_output_4500_3.Absorbance - np.min(krige_output_4500_3.Absorbance), c = '#8DC87E', lw = 2)
ax[1].text(5150, 0.0015, '$\mathregular{H_2O_{m, 5200}}$', ha = 'center', fontsize = 16) # 02775
ax[1].text(4485, 0.0015, '$\mathregular{OH^-_{4500}}$', ha = 'center', fontsize = 16) # 0165

ax[1].legend(loc = 'upper right', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax[1].set_xlim([4200, 5400])
ax[1].set_ylim([0, 0.03])
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].invert_xaxis()

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y = 0.06)
fig.supylabel('Absorbance', x = 0.06)
plt.tight_layout()
# plt.savefig('NIRPeaks.pdf')

# %% $\mathregular{H_2O_{t, 3550}}$ Peak Plotting

sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax.plot(data_H2O3550_1.Wavenumber, data_H2O3550_1.BL_MIR_3550, linestyle = '--', dashes = (3, 4), c = '#5E5E5E', lw = 1.5, label = 'ALS Baseline')
# ax.plot(data_H2O3550_2.Wavenumber, data_H2O3550_2.BL_MIR_3550, linestyle = '--', dashes = (2, 8), c = '#5E5E5E', lw = 1.5)
# ax.plot(data_H2O3550_3.Wavenumber, data_H2O3550_3.BL_MIR_3550, linestyle = '--', dashes = (2, 8), c = '#5E5E5E', lw = 1.5)
ax.plot(plot_output_3550_1.Wavenumber, plot_output_3550_1['Subtracted_Peak_Hat']+plot_output_3550_1['BL_MIR_3550'], c = '#E42211', lw = 2.5, label = 'Median Filtered Peak')
ax.plot(plot_output_3550_2.Wavenumber, plot_output_3550_2['Subtracted_Peak_Hat']+plot_output_3550_1['BL_MIR_3550'], c = '#E42211', lw = 2.5)
ax.plot(plot_output_3550_3.Wavenumber, plot_output_3550_3['Subtracted_Peak_Hat']+plot_output_3550_1['BL_MIR_3550'], c = '#E42211', lw = 2.5)
ax.text(3250, 0.55, '$\mathregular{H_2O_{t, 3550}}$', ha = 'center', fontsize = 16)
ax.text(1645, 1.5, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 16)
ax.text(1470, 0.95, '$\mathregular{CO_3^{2-}}$', ha = 'center', fontsize = 16)

ax.legend(loc = 'upper right', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax.set_xlim([1275, 4000])
ax.set_ylim([0, 3])
ax.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax.set_ylabel('Absorbance')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax.invert_xaxis()
plt.tight_layout()
# plt.savefig('3550Peak.pdf')


# %% Carbonate Peak Plotting

fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax.plot(x.Wavenumber, Baseline_Array_Plot[1, :], c = '#5E5E5E', lw = 0.25, label = 'MC$\mathregular{{^3}}$ Sampled Baselines')
for i in range(0, 302, 2):
    plt.plot(x.Wavenumber, Baseline_Array_Plot[i, :], c = '#5E5E5E', lw = 0.1)
ax.plot(x.Wavenumber, H1635_BP['Wavenumber']+Baseline_Solve_BP['Wavenumber'], c = '#E69F00', lw = 2, label = '$\mathregular{H_2O_{m, 1635}}$')
ax.plot(x.Wavenumber, CO2P1515_BP['Wavenumber']+Baseline_Solve_BP['Wavenumber'], c = '#E42211', lw = 2, label = '$\mathregular{CO_3^{2-}}$1515')
ax.plot(x.Wavenumber, CO2P1430_BP['Wavenumber']+Baseline_Solve_BP['Wavenumber'], c = '#009E73', lw = 2, label = '$\mathregular{CO_3^{2-}}$1430')
ax.plot(x.Wavenumber, carbonate['Wavenumber'], c = '#9A5ABD', lw = 2, label = 'MC$\mathregular{{^3}}$ Best-Fit Spectrum')
ax.plot(x.Wavenumber, Baseline_Solve_BP['Wavenumber'], linestyle = '--', dashes = (2, 2), c = 'k', lw = 2, label = 'MC$\mathregular{{^3}}$ Best-Fit Baseline')
ax.text(1645, 1.44, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 14)
ax.text(1470, 0.92, '$\mathregular{CO^{2-}_{3, 1515 and 1430}}$ ', ha = 'center', fontsize = 14)

handles, labels = plt.gca().get_legend_handles_labels()
order = [0,2,3,4,1,6,5]
ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'upper left', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax.set_xlim([1275, 2200])
ax.set_ylim([0.4, 1.6])

ax.set_xlabel('Wavenumber ($\mathregular{cm^{-1}}$)')
ax.set_ylabel('Absorbance')
ax.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax.invert_xaxis()
plt.tight_layout()
# plt.savefig('CarbonatePeak.pdf')

# %% 

fig = plt.figure(figsize = (15, 15))
gs = fig.add_gridspec(ncols = 4, nrows = 5)

ax0 = plt.subplot(gs[0, :])
ax1 = plt.subplot(gs[1, 0:2])
ax2 = plt.subplot(gs[2, 0:2])
ax3 = plt.subplot(gs[1:3, 2:4])
ax4 = plt.subplot(gs[3:5, 0:2])
ax5 = plt.subplot(gs[3:5, 2:4])


ax0.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax0.set_xlim([1275, 5500])
ax0.set_ylim([0, 3])
ax0.invert_xaxis()

ax1.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax1.plot(data_H2O5200_1.Wavenumber, data_H2O5200_1.Absorbance_Hat, c = '#0C7BDC', lw = 1, label = 'Median Filtered Peak')
ax1.plot(data_H2O5200_2.Wavenumber, data_H2O5200_2.Absorbance_Hat, c = '#0C7BDC', lw = 1)
ax1.plot(data_H2O5200_3.Wavenumber, data_H2O5200_3.Absorbance_Hat, c = '#0C7BDC', lw = 1)
ax1.plot(data_H2O5200_1.Wavenumber, data_H2O5200_1.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1, label = 'ALS Baselines')
ax1.plot(data_H2O5200_2.Wavenumber, data_H2O5200_2.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)
ax1.plot(data_H2O5200_3.Wavenumber, data_H2O5200_3.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)

ax1.plot(data_H2O4500_1.Wavenumber, data_H2O4500_1.Absorbance_Hat, c = '#5DB147', lw = 1)
ax1.plot(data_H2O4500_2.Wavenumber, data_H2O4500_2.Absorbance_Hat, c = '#5DB147', lw = 1)
ax1.plot(data_H2O4500_3.Wavenumber, data_H2O4500_3.Absorbance_Hat, c = '#5DB147', lw = 1)
ax1.plot(data_H2O4500_1.Wavenumber, data_H2O4500_1.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)
ax1.plot(data_H2O4500_2.Wavenumber, data_H2O4500_2.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)
ax1.plot(data_H2O4500_3.Wavenumber, data_H2O4500_3.BL_NIR_H2O, linestyle = '--', dashes = (4, 8), c = '#5E5E5E', lw = 1)
ax1.text(5150, 0.6625, '$\mathregular{H_2O_{m, 5200}}$', ha = 'center', fontsize = 16)
ax1.text(4485, 0.5225, '$\mathregular{OH^-_{4500}}$', ha = 'center', fontsize = 16)

ax1.legend(loc = 'lower left', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax1.set_xlim([4100, 5500])
ax1.set_ylim([0.4, 0.7])
ax1.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax1.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax1.axes.xaxis.set_ticklabels([])
ax1.invert_xaxis()


ax2.plot(data_H2O5200_1.Wavenumber, data_H2O5200_1['Subtracted_Peak'] - np.min(krige_output_5200_1.Absorbance), c = 'k', lw = 1, label = 'Baseline Subtracted Peak')
ax2.plot(data_H2O5200_2.Wavenumber, data_H2O5200_2['Subtracted_Peak'] - np.min(krige_output_5200_2.Absorbance), c = 'k', lw = 1)
ax2.plot(data_H2O5200_3.Wavenumber, data_H2O5200_3['Subtracted_Peak'] - np.min(krige_output_5200_3.Absorbance), c = 'k', lw = 1)
ax2.plot(krige_output_5200_1.Wavenumber, krige_output_5200_1.Absorbance - np.min(krige_output_5200_1.Absorbance), c = '#0C7BDC', lw = 2, label = 'Gaussian Kriged Peak')
ax2.plot(krige_output_5200_2.Wavenumber, krige_output_5200_2.Absorbance - np.min(krige_output_5200_2.Absorbance), c = '#074984', lw = 2)
ax2.plot(krige_output_5200_3.Wavenumber, krige_output_5200_3.Absorbance - np.min(krige_output_5200_3.Absorbance), c = '#6DAFEA', lw = 2)

ax2.plot(data_H2O4500_1.Wavenumber, data_H2O4500_1['Subtracted_Peak'] - np.min(krige_output_4500_1.Absorbance), c = 'k', lw = 1)
ax2.plot(data_H2O4500_2.Wavenumber, data_H2O4500_2['Subtracted_Peak'] - np.min(krige_output_4500_2.Absorbance), c = 'k', lw = 1)
ax2.plot(data_H2O4500_3.Wavenumber, data_H2O4500_3['Subtracted_Peak'] - np.min(krige_output_4500_3.Absorbance), c = 'k', lw = 1)
ax2.plot(krige_output_4500_1.Wavenumber, krige_output_4500_1.Absorbance - np.min(krige_output_4500_1.Absorbance), c = '#417B31', lw = 2)
ax2.plot(krige_output_4500_2.Wavenumber, krige_output_4500_2.Absorbance - np.min(krige_output_4500_2.Absorbance), c = '#5DB147', lw = 2) 
ax2.plot(krige_output_4500_3.Wavenumber, krige_output_4500_3.Absorbance - np.min(krige_output_4500_3.Absorbance), c = '#8DC87E', lw = 2)
ax2.text(5150, 0.0015, '$\mathregular{H_2O_{m, 5200}}$', ha = 'center', fontsize = 16) # 02775
ax2.text(4485, 0.0015, '$\mathregular{OH^-_{4500}}$', ha = 'center', fontsize = 16) # 0165

ax2.legend(loc = 'upper right', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax2.set_xlim([4100, 5500])
ax2.set_ylim([0, 0.03])
ax2.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax2.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax2.invert_xaxis()


ax3.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax3.plot(data_H2O3550_1.Wavenumber, data_H2O3550_1.BL_MIR_3550, linestyle = '--', dashes = (3, 4), c = '#5E5E5E', lw = 1.5, label = 'ALS Baseline')
# ax3.plot(data_H2O3550_2.Wavenumber, data_H2O3550_2.BL_MIR_3550, linestyle = '--', dashes = (2, 8), c = '#5E5E5E', lw = 1.5)
# ax3.plot(data_H2O3550_3.Wavenumber, data_H2O3550_3.BL_MIR_3550, linestyle = '--', dashes = (2, 8), c = '#5E5E5E', lw = 1.5)
ax3.plot(plot_output_3550_1.Wavenumber, plot_output_3550_1['Subtracted_Peak_Hat']+plot_output_3550_1['BL_MIR_3550'], c = '#E42211', lw = 2.5, label = 'Median Filtered Peak')
ax3.plot(plot_output_3550_2.Wavenumber, plot_output_3550_2['Subtracted_Peak_Hat']+plot_output_3550_1['BL_MIR_3550'], c = '#E42211', lw = 2.5)
ax3.plot(plot_output_3550_3.Wavenumber, plot_output_3550_3['Subtracted_Peak_Hat']+plot_output_3550_1['BL_MIR_3550'], c = '#E42211', lw = 2.5)q
ax3.text(3250, 0.55, '$\mathregular{H_2O_{t, 3550}}$', ha = 'center', fontsize = 16)
ax3.text(1645, 1.5, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 16)
ax3.text(1470, 0.95, '$\mathregular{CO_3^{2-}}$', ha = 'center', fontsize = 16)

ax3.legend(loc = 'upper right', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax3.set_xlim([1275, 4000])
ax3.set_ylim([0, 3])
ax3.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax3.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax3.invert_xaxis()

ax4.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax4.plot(x.Wavenumber, Baseline_Array_Plot[1, :], c = '#5E5E5E', lw = 0.25, label = 'MC$\mathregular{{^3}}$ Sampled Baselines')
for i in range(0, 302, 2):
    ax4.plot(x.Wavenumber, Baseline_Array_Plot[i, :], c = '#5E5E5E', lw = 0.1)
ax4.plot(x.Wavenumber, H1635_BP['Wavenumber']+Baseline_Solve_BP['Wavenumber'], c = '#E69F00', lw = 2, label = '$\mathregular{H_2O_{m, 1635}}$')
ax4.plot(x.Wavenumber, CO2P1515_BP['Wavenumber']+Baseline_Solve_BP['Wavenumber'], c = '#E42211', lw = 2, label = '$\mathregular{CO_{3, 1515}^{2-}}$')
ax4.plot(x.Wavenumber, CO2P1430_BP['Wavenumber']+Baseline_Solve_BP['Wavenumber'], c = '#009E73', lw = 2, label = '$\mathregular{CO_{3, 1430}^{2-}}$')
ax4.plot(x.Wavenumber, carbonate['Wavenumber'], c = '#9A5ABD', lw = 2, label = 'MC$\mathregular{{^3}}$ Best-Fit Spectrum')
ax4.plot(x.Wavenumber, Baseline_Solve_BP['Wavenumber'], linestyle = '--', dashes = (2, 2), c = 'k', lw = 2, label = 'MC$\mathregular{{^3}}$ Best-Fit Baseline')
ax4.text(1645, 1.45, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 18)
ax4.text(1465, 0.94, '$\mathregular{CO_{3}^{2-}}$', ha = 'center', fontsize = 18)
ax4.text(1470, 0.9, '1515 and 1430', ha = 'center', fontsize = 10)

handles, labels = ax4.get_legend_handles_labels()
order = [0,2,3,4,1,6,5]
ax4.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'upper left', labelspacing = 0.2, handletextpad = 0.5, handlelength = 1.0, prop={'size': 16}, frameon=False)
ax4.set_xlim([1275, 2200])
ax4.set_ylim([0.4, 1.6])

ax4.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax4.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax4.invert_xaxis()


spec_fit = spec[(spec.Wavenumber>1275) & (spec.Wavenumber<2200)]

ax5.scatter(spec_fit.Wavenumber[2::10], spec_fit.Absorbance[2::10] - carbonate['Wavenumber'].values[2::10], c = 'k', label = 'Residual')
ax5.errorbar(spec_fit.Wavenumber[2::10], spec_fit.Absorbance[2::10] - carbonate['Wavenumber'].values[2::10], yerr = np.abs(spec_fit.Absorbance[2::10] - carbonate['Wavenumber'].values[2::10]) * 0.1, lw = 0.5, c = 'k', label = 'Residual')

# ax[0].errorbar(H2O_expmean(STD_D1010), H2O_mean(STD_D1010), xerr = H2O_expstd(STD_D1010), yerr = H2O_mean(STD_D1010) * H2O_rsd(STD_D1010), lw = 0.5, c = 'k', zorder = 10)



ax5.set_xlim([1275, 2200])
ax5.set_ylim([-0.02, 0.02])

ax5.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax5.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax5.invert_xaxis()




fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y = 0.03)
fig.supylabel('Absorbance', x = 0.03)

plt.tight_layout()
# plt.savefig('AllPeak_new.pdf')

# %%
# %%

spec1 = pd.read_csv('./InputData/AC4_OL53_101220_256s_30x30_a.csv')

fig, ax = plt.subplots(2, 2, figsize = (14, 14))
ax = ax.flatten()
ax0 = ax[0]
ax1 = ax[1]
ax3 = ax[2]
ax4 = ax[3]
# gs = fig.add_gridspec(ncols = 4, nrows = 5)

# ax0 = plt.subplot(gs[0, :])
# ax1 = plt.subplot(gs[1:3, 0:2])
# ax3 = plt.subplot(gs[1:3, 2:4])
# ax4 = plt.subplot(gs[3:5, 0:2])

ax0.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax0.plot(spec1.Wavenumber, spec1.Absorbance, c = 'grey', lw = 2, label = 'FTIR Spectrum')
ax0.axhline(3, c = 'k')
ax0.axvspan(5150, 5250, 0, 3, color = '#0C7BDC', lw = 0, alpha=0.2)
ax0.axvspan(4450, 4550, 0, 3, color = '#5DB147', lw = 0, alpha=0.2)
ax0.axvspan(3500, 3600, 0, 3, color = '#E42211', lw = 0, alpha=0.2)
ax0.axvspan(1600, 1665, 0, 3, color = '#F9C300', lw = 0, alpha=0.2)
ax0.axvspan(1380, 1565, 0, 3, color = 'k', lw = 0, alpha=0.2)
ax0.scatter(4000, 2.95, s = 100, marker = '|', c = 'k')
ax0.text(4750, 2.6, 'Near IR', ha = 'center', fontsize = 16)
ax0.text(2637.5, 2.6, 'Mid IR', ha = 'center', fontsize = 16)
ax0.text(5200, 1.1, '$\mathregular{H_2O_{m}}$', ha = 'center', fontsize = 16)
ax0.text(4500, 1.1, '$\mathregular{OH^-}$', ha = 'center', fontsize = 16)
ax0.text(3550, 0.5, '$\mathregular{H_2O_{t}}$', ha = 'center', fontsize = 16)
ax0.text(1645, 1.6, '$\mathregular{H_2O_{m}}$', ha = 'center', fontsize = 16)
ax0.text(1430, 1.1, '$\mathregular{CO_{3}^{2-}}$', ha = 'center', fontsize = 16)
ax0.text(2350, 1.1, '$\mathregular{CO_{2}}$', ha = 'center', fontsize = 16)

ax0.annotate("A.", xy=(0.0075, 0.87), xycoords="axes fraction", fontsize=20, weight='bold')
ax0.set_xlim([1200, 5500])
ax0.set_ylim([0, 3])
ax0.invert_xaxis()

ax1.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax1.plot(spec1.Wavenumber, spec1.Absorbance + 0.325, c = 'grey', lw = 2, label = 'FTIR Spectrum')
ax1.axvspan(5150, 5250, 0, 3, color = '#0C7BDC', lw = 0, alpha=0.2)
ax1.axvspan(4450, 4550, 0, 3, color = '#5DB147', lw = 0, alpha=0.2)
ax1.text(5200, 0.42, '$\mathregular{H_2O_{m, 5200}}$', ha = 'center', fontsize = 18)
ax1.text(4500, 0.42, '$\mathregular{OH^-_{4500}}$', ha = 'center', fontsize = 18)
ax1.annotate("B.", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax1.set_xlim([4100, 5500])
ax1.set_ylim([0.4, 0.7])
ax1.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax1.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax1.invert_xaxis()

ax3.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax3.plot(spec1.Wavenumber, spec1.Absorbance, c = 'grey', lw = 2, label = 'FTIR Spectrum')
ax3.axvspan(3500, 3600, 0, 3, color = '#E42211', lw = 0, alpha=0.2)
ax3.axvspan(1600, 1665, 0, 3, color = '#F9C300', lw = 0, alpha=0.2)
ax3.axvspan(1380, 1565, 0, 3, color = 'k', lw = 0, alpha=0.2)
ax3.text(3250, 0.24, '$\mathregular{H_2O_{t, 3550}}$', ha = 'center', fontsize = 18)
ax3.text(1645, 1.5, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 18)
ax3.text(1455, 1.1, '$\mathregular{CO_3^{2-}}$', ha = 'center', fontsize = 18)
ax3.text(2350, 0.24, '$\mathregular{CO_2}$', ha = 'center', fontsize = 18)
# ax3.text(1475, 0.05, '1515 and 1430', ha = 'center', fontsize = 10)
ax3.annotate("C.", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')

ax3.set_xlim([1275, 4000])
ax3.set_ylim([0, 3])
ax3.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax3.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax3.invert_xaxis()

ax4.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax4.plot(spec1.Wavenumber, spec1.Absorbance, c = 'grey', lw = 2, label = 'FTIR Spectrum')
ax4.axvspan(3500, 3600, 0, 3, color = '#E42211', lw = 0, alpha=0.2)
ax4.axvspan(1600, 1665, 0, 3, color = '#F9C300', lw = 0, alpha=0.2)
ax4.axvspan(1380, 1565, 0, 3, color = 'k', lw = 0, alpha=0.2)

ax4.annotate("D.", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax4.set_xlim([1275, 2200])
ax4.set_ylim([0, 1.6])
ax4.text(1645, 1.45, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 18)
ax4.text(1465, 1.0, '$\mathregular{CO_{3}^{2-}}$', ha = 'center', fontsize = 18)
ax4.text(1470, 0.925, '1515 and 1430', ha = 'center', fontsize = 10)

ax4.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax4.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax4.invert_xaxis()

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y = 0.03)
fig.supylabel('Absorbance', x = 0.03)

plt.tight_layout()
# plt.savefig('AllPeaks_Prelim.pdf', bbox_inches='tight', pad_inches = 0.025)


# %%


spec1 = pd.read_csv('./InputData/AC4_OL53_101220_256s_30x30_a.csv')

fig, ax = plt.subplots(2, 2, figsize = (13, 13))
ax = ax.flatten()
ax0 = ax[0]
ax1 = ax[1]
ax3 = ax[2]
ax4 = ax[3]


ax0.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax0.plot(spec1.Wavenumber, spec1.Absorbance, c = 'grey', lw = 2, label = 'FTIR Spectrum')
ax0.axhline(3, c = 'k')
ax0.axvspan(5150, 5250, 0, 3, color = '#0C7BDC', lw = 0, alpha=0.2)
ax0.axvspan(4450, 4550, 0, 3, color = '#5DB147', lw = 0, alpha=0.2)
ax0.axvspan(3500, 3600, 0, 3, color = '#E42211', lw = 0, alpha=0.2)
ax0.axvspan(1600, 1665, 0, 3, color = '#F9C300', lw = 0, alpha=0.2)
ax0.axvspan(1380, 1565, 0, 3, color = 'k', lw = 0, alpha=0.2)
ax0.scatter(4000, 3.0, s = 200, marker = '|', c = 'k')
ax0.text(4750, 2.8, 'Near IR', ha = 'center', fontsize = 16)
ax0.text(2637.5, 2.8, 'Mid IR', ha = 'center', fontsize = 16)
t1 = ax0.text(5200, 0.8, '$\mathregular{H_2O_{m}}$\n$\mathregular{5200}$', ha = 'center', fontsize = 16)
t1.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.025))
t2 = ax0.text(4500, 0.8, '$\mathregular{OH^-}$\n$\mathregular{4500}$', ha = 'center', fontsize = 16)
t2.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.025))
t3 = ax0.text(3398, 0.5, '$\mathregular{H_2O_{t}}$\n$\mathregular{3550}$', ha = 'center', fontsize = 16)
t3.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.01))
t4 = ax0.text(1645, 1.55, '$\mathregular{H_2O_{m}}$\n$\mathregular{1635}$', ha = 'center', fontsize = 16)
t4.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.025))
t5 = ax0.text(1450, 0.225, '$\mathregular{CO_{3}^{2-}}$', ha = 'center', fontsize = 15)
t5.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.00))

t5_1 = ax0.text(1450, 0.0435, '$\mathregular{1515}$\n$\mathregular{1430}$', ha = 'center', fontsize = 10)
t5_1.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.00))


t6 = ax0.text(2350, 0.8, '$\mathregular{CO_{2}}$\n$\mathregular{2350}$', ha = 'center', fontsize = 16)
t6.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.025))
ax0.annotate("A.", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax0.set_xlim([1200, 5500])
ax0.set_ylim([0, 3])
ax0.invert_xaxis()

ax1.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax1.plot(spec1.Wavenumber, spec1.Absorbance + 0.325, c = 'grey', lw = 2, label = 'FTIR Spectrum')
ax1.axvspan(5150, 5250, 0, 3, color = '#0C7BDC', lw = 0, alpha=0.2)
ax1.axvspan(4450, 4550, 0, 3, color = '#5DB147', lw = 0, alpha=0.2)
t1 = ax1.text(5200, 0.42, '$\mathregular{H_2O_{m, 5200}}$', ha = 'center', fontsize = 18)
t1.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.05))
t2 = ax1.text(4500, 0.42, '$\mathregular{OH^-_{4500}}$', ha = 'center', fontsize = 18)
t2.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.05))

ax1.annotate("B.", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax1.set_xlim([4100, 5500])
ax1.set_ylim([0.4, 0.7])
ax1.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax1.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax1.invert_xaxis()

ax3.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax3.plot(spec1.Wavenumber, spec1.Absorbance, c = 'grey', lw = 2, label = 'FTIR Spectrum')
ax3.axvspan(3500, 3600, 0, 3, color = '#E42211', lw = 0, alpha=0.2)
ax3.axvspan(1600, 1665, 0, 3, color = '#F9C300', lw = 0, alpha=0.2)
ax3.axvspan(1380, 1565, 0, 3, color = 'k', lw = 0, alpha=0.2)
t1 = ax3.text(3400, 0.2, '$\mathregular{H_2O_{t, 3550}}$', ha = 'center', fontsize = 18)
t1.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.025))
t2 = ax3.text(1645, 1.5, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 18)
t2.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.05))
t3 = ax3.text(1455, 0.2, '$\mathregular{CO_3^{2-}}$', ha = 'center', fontsize = 18)
t3.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.05))
t4 = ax3.text(2350, 0.7, '$\mathregular{CO_{2, 2350}}$', ha = 'center', fontsize = 18)
# t4.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.05))
# ax3.text(1475, 0.05, '1515 and 1430', ha = 'center', fontsize = 10)
ax3.annotate("C.", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')

ax3.set_xlim([1275, 4000])
ax3.set_ylim([0, 3])
ax3.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax3.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax3.invert_xaxis()

ax4.plot(spec.Wavenumber, spec.Absorbance, c = 'k', lw = 2, label = 'FTIR Spectrum')
ax4.plot(spec1.Wavenumber, spec1.Absorbance, c = 'grey', lw = 2, label = 'FTIR Spectrum')
ax4.axvspan(3500, 3600, 0, 3, color = '#E42211', lw = 0, alpha=0.2)
ax4.axvspan(1600, 1665, 0, 3, color = '#F9C300', lw = 0, alpha=0.2)
ax4.axvspan(1380, 1565, 0, 3, color = 'k', lw = 0, alpha=0.2)

ax4.annotate("D.", xy=(0.02, 0.94), xycoords="axes fraction", fontsize=20, weight='bold')
ax4.set_xlim([1275, 2200])
ax4.set_ylim([0, 1.6])
t1 = ax4.text(1645, 1.45, '$\mathregular{H_2O_{m, 1635}}$', ha = 'center', fontsize = 18)
t1.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.05))
t2 = ax4.text(1465, 1.0, '$\mathregular{CO_{3}^{2-}}$', ha = 'center', fontsize = 18)
t2.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.05))
t3 = ax4.text(1470, 0.945, '1515 and 1430', ha = 'center', fontsize = 10)
t3.set_bbox(dict(facecolor='white', alpha=1, edgecolor='white', pad=0.05))

ax4.tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax4.tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax4.invert_xaxis()

fig.supxlabel('Wavenumber ($\mathregular{cm^{-1}}$)', y = 0.04)
fig.supylabel('Absorbance', x = 0.05)


plt.tight_layout()
plt.savefig('AllPeaks_Prelim_new.pdf', bbox_inches='tight', pad_inches = 0.025)

# %%
