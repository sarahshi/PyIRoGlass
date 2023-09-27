# %% -*- coding: utf-8 -*-
""" Created on June 12, 2021 // @author: Sarah Shi """

# Import packages
import os
import sys
import glob
import numpy as np
import pandas as pd
import mc3

sys.path.append('../src/')
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

stdno = 1

MEGA_SPREADSHEET = pd.read_csv('../FINALDATA/' + OUTPUT_PATHS[stdno] + '_DF.csv', index_col = 0) 
MEGA_SPREADSHEET['Sample ID'] = MEGA_SPREADSHEET.index

CONC = pd.read_csv('../FINALDATA/' + OUTPUT_PATHS[stdno] + '_H2OCO2.csv', index_col = 0) 
CONC['Sample ID'] = MEGA_SPREADSHEET.index

HJ = pd.read_csv('PyIRoGlass-LHJ_091623.csv', index_col=0)
HJ_peaks = HJ[['Sample ID', 'Repeats', 'PH_1430', 'PH_1515', 'Thickness']]

CONC_conc = CONC[['CO2_MEAN', 'CO2_STD']]

merge = MEGA_SPREADSHEET.merge(HJ_peaks, on='Sample ID')
merge = merge.merge(CONC, on='Sample ID')
merge = merge.set_index('Sample ID')
merge.to_csv('PHComparison.csv')

merge

# %%

badspec = np.array(['CI_IPGP_B6_1_50x50_256s_sp1', 'CI_IPGP_B6_2_50x50_256s_sp1', 'CI_IPGP_B6_1_50x50_256s_sp2', 'CI_IPGP_NBO_2_2_1_100x100_256s_sp1', 
                    'CI_Ref_13_1_100x100_256s_sp1', 'CI_Ref_13_1_100x100_256s_sp2', 'CI_Ref_13_1_100x100_256s_sp3', 'CI_Ref_13_1_100x100_256s_sp4', 
                    'CI_Ref_22_1_100x100_256s_sp1', 'CI_Ref_22_1_100x100_256s_sp2', 'CI_Ref_22_1_100x100_256s_sp3', 
                    'CI_Ref_23_1_100x100_256s_040523_sp1', 'CI_Ref_23_1_100x100_256s_040523_sp3', 'CI_Ref_23_1_100x100_256s_sp4', 'CI_Ref_23_1_100x100_256s_sp5',
                    'CI_Ref_25_1_100x100_256s_sp3',
                    'CI_Ref_bas_1_1_100x100_256s_sp1', 'CI_Ref_bas_1_1_100x100_256s_sp2', 
                    'CI_Ref_bas_1_2_100x100_256s_sp1', 'CI_Ref_bas_1_2_100x100_256s_sp2', 
                    'CI_Ref_bas_2_1_100x100_256s_sp1', 
                    'CI_Ref_bas_2_2_100x100_256s_4sp1', 'CI_Ref_bas_2_2_100x100_256s_sp2', 'CI_Ref_bas_2_2_100x100_256s_sp3', 
                    'CI_Ref_bas_2_3_100x100_256s_sp1', 
                    'CI_Ref_bas_3_1_100x100_256s_051423_sp1', 'CI_Ref_bas_3_2_100x100_256s_051423_sp1', 'CI_Ref_bas_3_3_100x100_256s_sp1', 
                    'CI_Ref_bas_4_1_100x100_256s_sp1', 'CI_Ref_bas_4_1_100x100_256s_sp2',
                    'LMT_BA3_2_50x50_256s_sp1', 'LMT_BA3_2_50x50_256s_sp2', 'CI_LMT_BA5_2_50x50x_256s_sp1', 'ND70_02_01_06032022_150x150_sp1',
                    'ND70_5_2_29June2022_150x150_sp2', 'ND70_05_03_06032022_80x100_sp3', 'ND70_0503_29June2022_95x80_256s_sp2', 'ND70_05_02_06032022_150x150_sp1', 'ND70_05_02_06032022_150x150_sp2', 'ND70_05_02_06032022_150x150_sp3',
                    'ND70_06_02_75um', 'ND70_6-2_08042022_150x150_sp1', 'ND70_6-2_08042022_150x150_sp2', 'ND70_6-2_08042022_150x150_sp3'
                    ])

merge = merge[~merge.index.isin(badspec)]

merge = merge[~merge.index.str.contains('map_actual_focusedProperly', case=False, na=False)]

merge

# %% 
# %% 

from sklearn.metrics import mean_squared_error
import scipy

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

line = np.array([0, 3])
sz = 80
ticks = np.arange(0, 3, 0.5)
tick_labels = [str(t) if t % 0.5 == 0 else "" for t in ticks]


merge['PH_1515_norm'] = merge['PH_1515']/merge.Thickness*50
merge['PH_1515_BP_norm'] = merge.PH_1515_BP/merge.Thickness*50
merge['PH_1515_STD_norm'] = merge.PH_1515_STD/merge.Thickness*50
merge['PH_1430_norm'] = merge['PH_1430']/merge.Thickness*50
merge['PH_1430_BP_norm'] = merge.PH_1430_BP/merge.Thickness*50
merge['PH_1430_STD_norm'] = merge.PH_1430_STD/merge.Thickness*50
merge['Py_Devol_1430'] = merge['PH_1430_BP_norm'] / merge['PH_1430_norm']
merge['Py_Devol_1515'] = merge['PH_1515_BP_norm'] / merge['PH_1515_norm']


# Calculate the mean and standard deviation
mean_1430 = np.mean(merge['Py_Devol_1430'])
std_1430 = np.std(merge['Py_Devol_1430'])

mean_1515 = np.mean(merge['Py_Devol_1515'])
std_1515 = np.std(merge['Py_Devol_1515'])

# Filter dataframe
merge_int = merge[abs(merge['Py_Devol_1430'] - mean_1430) < 1.5 * std_1430]
merge_filt = merge_int[abs(merge_int['Py_Devol_1515'] - mean_1515) < 1.5 * std_1515]

merge_filt.to_csv('PHComparison_lim.csv')
merge_filt



slope0, intercept0, r_value0, p_value0, std_err0 = scipy.stats.linregress(merge_filt.PH_1515_norm, merge_filt.PH_1515_BP_norm)
ccc0 = concordance_correlation_coefficient(merge_filt.PH_1515_norm, merge_filt.PH_1515_BP_norm)
rmse0 = mean_squared_error(merge_filt.PH_1515_norm, merge_filt.PH_1515_BP_norm, squared=False)

slope1, intercept1, r_value1, p_value1, std_err1 = scipy.stats.linregress(merge_filt.PH_1430_norm, merge_filt.PH_1430_BP_norm)
ccc1 = concordance_correlation_coefficient(merge_filt.PH_1430_norm, merge_filt.PH_1430_BP_norm)
rmse1 = mean_squared_error(merge_filt.PH_1430_norm, merge_filt.PH_1430_BP_norm, squared=False)


fig, ax = plt.subplots(2, 2, figsize=(14, 14))
ax=ax.flatten()
ax[0].plot(line, line, 'k', lw = 1, zorder = 0, label='1-1 Line')
ax[0].fill_between(line, 0.9*line, 1.1*line, color='gray', edgecolor=None, alpha=0.25, label='10% Uncertainty')
ax[0].scatter(merge_filt.PH_1515_norm, merge_filt.PH_1515_BP_norm, s = sz, c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20,)
ax[0].errorbar(merge_filt.PH_1515_norm, merge_filt.PH_1515_BP_norm, yerr=merge_filt.PH_1515_STD/merge_filt.Thickness*200, xerr=merge_filt['PH_1515']/merge_filt.Thickness*50*0.05, fmt='none', lw = 0.5, c = 'k', zorder = 10)
ax[0].annotate(r'A. $\mathregular{CO_{3, 1430}^{2-}}, n=$'+str(len(merge_filt)), xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va ='bottom', size = 20)

ax[0].set_xlim([0, 2.5])
ax[0].set_ylim([0, 2.5])
ax[0].set_xticks(ticks)  # Set x ticks
ax[0].set_yticks(ticks)  # Set y ticks
ax[0].set_xticklabels(tick_labels)
ax[0].set_ylabel('PyIRoGlass Peak Height')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

ax[0].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value0**2, 3)), xy=(0.03, 0.7975), xycoords="axes fraction", fontsize=14)
ax[0].annotate("RMSE="+str(np.round(rmse0, 3)), xy=(0.03, 0.84), xycoords="axes fraction", fontsize=14)
ax[0].annotate("CCC="+str(np.round(ccc0, 3)), xy=(0.03, 0.88), xycoords="axes fraction", fontsize=14)
ax[0].annotate("m="+str(np.round(slope0, 3)), xy=(0.03, 0.76), xycoords="axes fraction", fontsize=14)
ax[0].annotate("b="+str(np.round(intercept0, 3)), xy=(0.03, 0.72), xycoords="axes fraction", fontsize=14)


ax[1].plot(line, line, 'k', lw = 1, zorder = 0, label='1-1 Line')
ax[1].fill_between(line, 0.9*line, 1.1*line, color='gray', edgecolor=None, alpha=0.25, label='10% Uncertainty')
ax[1].scatter(merge_filt.PH_1430_norm, merge_filt.PH_1430_BP_norm, s = sz, c = '#0C7BDC', ec = '#171008', lw = 0.5, zorder = 20)
ax[1].errorbar(merge_filt.PH_1430_norm, merge_filt.PH_1430_BP_norm, yerr=merge_filt.PH_1430_STD/merge_filt.Thickness*200, xerr=merge_filt['PH_1430']/merge_filt.Thickness*50*0.05, fmt='none', lw = 0.5, c = 'k', zorder = 10)
ax[1].annotate(r'B. $\mathregular{CO_{3, 1430}^{2-}}, n=$'+str(len(merge_filt)), xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va ='bottom', size = 20)
ax[1].set_xlim([0, 2.5])
ax[1].set_ylim([0, 2.5])
ax[1].set_xticks(ticks)  # Set x ticks
ax[1].set_yticks(ticks)  # Set y ticks
ax[1].set_xticklabels(tick_labels)

# ax[1].set_xlabel('Devolatilized Peak Height')
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].legend(loc='lower right', labelspacing = 0.2, handletextpad = 0.25, handlelength = 1.00, prop={'size': 13}, frameon=False)

ax[1].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value1**2, 3)), xy=(0.03, 0.7975), xycoords="axes fraction", fontsize=14)
ax[1].annotate("CCC="+str(np.round(ccc1, 3)), xy=(0.03, 0.88), xycoords="axes fraction", fontsize=14)
ax[1].annotate("RMSE="+str(np.round(rmse1, 3)), xy=(0.03, 0.84), xycoords="axes fraction", fontsize=14)
ax[1].annotate("m="+str(np.round(slope1, 3)), xy=(0.03, 0.76), xycoords="axes fraction", fontsize=14)
ax[1].annotate("b="+str(np.round(intercept1, 3)), xy=(0.03, 0.72), xycoords="axes fraction", fontsize=14)

# fig, ax = plt.subplots(1, 2, figsize=(14, 7))
# ax.flatten()
sc2 = ax[2].scatter(merge_filt.PH_1515_norm, (merge_filt['Py_Devol_1515']), s = sz, 
                    c = '#0C7BDC', ec = '#171008', lw = 0.5, 
                    zorder = 20)
sc1 = ax[3].scatter(merge_filt.PH_1430_norm, (merge_filt['Py_Devol_1430']), s = sz, 
                    c = '#0C7BDC', ec = '#171008', lw = 0.5, 
                    zorder = 20)
ax[2].axhline(np.mean(merge_filt['Py_Devol_1515']), color='k', linestyle='--', dashes = (10, 10), linewidth=0.75,)
ax[2].annotate(r'C. $\mathregular{CO_{3, 1515}^{2-}}, n=$'+str(len(merge_filt)), xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va ='bottom', size = 20)
ax[2].text(1.9, 0.9575, r'$\overline{\frac{P}{D}}$='+str(round(np.mean(merge_filt['Py_Devol_1515']), 4)), ha='left', va ='bottom', size = 20)

ax[2].fill_between(line, np.mean(merge_filt['Py_Devol_1515'])-np.std(merge_filt['Py_Devol_1515']), np.mean(merge_filt['Py_Devol_1515'])+np.std(merge_filt['Py_Devol_1515']), color = 'k', alpha=0.10, edgecolor = None,
    zorder = -5, label='68% Confidence Interval')

ax[3].axhline(np.mean(merge_filt['Py_Devol_1430']), color='k', linestyle='--', dashes = (10, 10), linewidth=0.75, label='Mean')
ax[3].annotate(r'D. $\mathregular{CO_{3, 1430}^{2-}}, n=$'+str(len(merge_filt)), xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va ='bottom', size = 20)
ax[3].text(1.9, 0.9575, r'$\overline{\frac{P}{D}}$='+str(round(np.mean(merge_filt['Py_Devol_1430']), 4)), ha='left', va ='bottom', size = 20)

ax[3].fill_between(line, np.mean(merge_filt['Py_Devol_1430'])-np.std(merge_filt['Py_Devol_1430']), np.mean(merge_filt['Py_Devol_1430'])+np.std(merge_filt['Py_Devol_1430']), color = 'k', alpha=0.10, edgecolor = None,
    zorder = -5, label='68% Confidence Interval')
ax[3].legend(loc='lower right', labelspacing = 0.2, handletextpad = 0.25, handlelength = 1.00, prop={'size': 13}, frameon=False)

ax[2].set_xlabel('Devolatilized Peak Height')
ax[2].set_ylabel('PyIRoGlass/Devolatilized Peak Height')
ax[2].set_xlim([0, 2.5])
ax[2].set_xticks(ticks)  # Set x ticks
ax[2].set_xticklabels(tick_labels)
ax[2].set_ylim([0.85, 1.05])
ax[2].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[2].tick_params(axis="y", direction='in', length=5, pad = 6.5)

ax[3].set_xlim([0, 2.5])
ax[3].set_xticks(ticks)  # Set x ticks
ax[3].set_xticklabels(tick_labels)
ax[3].set_ylim([0.85, 1.05])
ax[3].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[3].tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
plt.savefig('PHCombined_HJND.pdf', bbox_inches='tight', pad_inches = 0.025)

# %% 

col_means = ['PH_1515_norm', 'PH_1515_BP_norm', 'PH_1515_STD_norm', 'PH_1430_norm', 'PH_1430_BP_norm', 'PH_1430_STD_norm', 
             'Py_Devol_1430', 'Py_Devol_1515', 'H2OT_MEAN', 'H2OT_STD', 'CO2_MEAN', 'CO2_STD']


means = merge_filt.groupby('Repeats')[col_means].mean()
display(means)

std = merge_filt.groupby('Repeats')[col_means].std()
std.to_csv('std.csv')
display(std)

means['per_analysis_1515'] = means['PH_1515_STD_norm'] / means['PH_1515_BP_norm']
means['per_analysis_1430'] = means['PH_1430_STD_norm'] / means['PH_1430_BP_norm']
means['per_repeat_1515'] = std['PH_1515_BP_norm'] / means['PH_1515_BP_norm']
means['per_repeat_1430'] = std['PH_1430_BP_norm'] / means['PH_1430_BP_norm']
means['error_prop_1515'] = np.where(means['per_repeat_1515'].isna(), means['per_analysis_1515'],
    np.sqrt(means['per_analysis_1515']**2 + means['per_repeat_1515']**2))
means['error_prop_1430'] = np.where(means['per_repeat_1430'].isna(), means['per_analysis_1430'],
    np.sqrt(means['per_analysis_1430']**2 + means['per_repeat_1430']**2))
means['PH_1515_STD_net'] = means['error_prop_1515'] * means['PH_1515_BP_norm']
means['PH_1430_STD_net'] = means['error_prop_1430'] * means['PH_1430_BP_norm']


means['per_analysis_h2o'] = means['H2OT_STD'] / means['H2OT_MEAN']
means['per_analysis_co2'] = means['CO2_STD'] / means['CO2_MEAN']
means['per_repeat_h2o'] = std['H2OT_MEAN'] / means['H2OT_MEAN']
means['per_repeat_co2'] = std['CO2_MEAN'] / means['CO2_MEAN']
means['error_prop_h2o'] = np.where(means['per_repeat_h2o'].isna(), means['per_analysis_h2o'],
    np.sqrt(means['per_analysis_h2o']**2 + means['per_repeat_h2o']**2))
means['error_prop_co2'] = np.where(means['per_repeat_co2'].isna(), means['per_analysis_co2'],
    np.sqrt(means['per_analysis_co2']**2 + means['per_repeat_co2']**2))
means['H2OT_STD_net'] = means['error_prop_h2o'] * means['H2OT_MEAN']
means['CO2_STD_net'] = means['error_prop_co2'] * means['CO2_MEAN']
means.to_csv('means.csv')



counts = merge_filt.groupby('Repeats')['PH_1515_norm'].count()
counts.to_csv('counts.csv')
display(counts)


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

fuegono = 0 
stdno = 1
fuegorhno = 2 

MICOMP0, THICKNESS0 = pig.Load_ChemistryThickness('../Inputs/FuegoChemThick.csv')
MICOMP1, THICKNESS1 = pig.Load_ChemistryThickness('../Inputs/StandardChemThick.csv')
MICOMP2, THICKNESS2 = pig.Load_ChemistryThickness('../Inputs/FuegoRHChemThick.csv')

MICOMP = pd.concat([MICOMP0, MICOMP1, MICOMP2])
THICKNESS = pd.concat([THICKNESS0, THICKNESS1, THICKNESS2])

nbo_t = NBO_T(MICOMP)
nbo_t_lim = nbo_t 
THICKNESS_lim = THICKNESS[MICOMP.Fe2O3 != 0]
nbo_t_lim = nbo_t[MICOMP.Fe2O3 != 0]
THICKNESS_only = THICKNESS_lim.Thickness

MICOMP_merge = MICOMP[MICOMP.Fe2O3 != 0]

MEGA_SPREADSHEET0 = pd.read_csv('../' + output_dir[-1] + '/' + OUTPUT_PATHS[fuegono] + '_DF.csv') #, index_col=0)
MEGA_SPREADSHEET1 = pd.read_csv('../' +output_dir[-1] + '/' + OUTPUT_PATHS[stdno] + '_DF.csv') #, index_col=0)
MEGA_SPREADSHEET2 = pd.read_csv('../' +output_dir[-1] + '/' + OUTPUT_PATHS[fuegorhno] + '_DF.csv') #, index_col=0)
H2OCO20 = pd.read_csv('../' +output_dir[-1] + '/' + OUTPUT_PATHS[fuegono] + '_H2OCO2.csv') #, index_col=0)
H2OCO21 = pd.read_csv('../' +output_dir[-1] + '/' + OUTPUT_PATHS[stdno] + '_H2OCO2.csv') #, index_col=0)
H2OCO22 = pd.read_csv('../' +output_dir[-1] + '/' + OUTPUT_PATHS[fuegorhno] + '_H2OCO2.csv') #, index_col=0)
DENSEPS0 = pd.read_csv('../' +output_dir[-1] + '/' + OUTPUT_PATHS[fuegono] + '_DensityEpsilon.csv') #, index_col=0)
DENSEPS1 = pd.read_csv('../' +output_dir[-1] + '/' + OUTPUT_PATHS[stdno] + '_DensityEpsilon.csv') #, index_col=0)
DENSEPS2 = pd.read_csv('../' +output_dir[-1] + '/' + OUTPUT_PATHS[fuegorhno] + '_DensityEpsilon.csv') #, index_col=0)

MEGA_SPREADSHEET = pd.concat([MEGA_SPREADSHEET0, MEGA_SPREADSHEET1, MEGA_SPREADSHEET2])
MEGA_SPREADSHEET = MEGA_SPREADSHEET.rename(columns={'Unnamed: 0': 'Sample'})
MEGA_SPREADSHEET = MEGA_SPREADSHEET.set_index('Sample')
MEGA_SPREADSHEET_lim = MEGA_SPREADSHEET[MICOMP.Fe2O3 != 0]

DENSEPS = pd.concat([DENSEPS0, DENSEPS1, DENSEPS2])
DENSEPS = DENSEPS.rename(columns={'Unnamed: 0': 'Sample'})
DENSEPS = DENSEPS.set_index('Sample')
DENSEPS1 = DENSEPS[MICOMP.Fe2O3 != 0]

H2OCO2 = pd.concat([H2OCO20, H2OCO21, H2OCO22])
H2OCO2 = H2OCO2.rename(columns={'Unnamed: 0': 'Sample'})
H2OCO2 = H2OCO2.set_index('Sample')
H2OCO2_1 = H2OCO2[MICOMP.Fe2O3 != 0]

THICKNESS_lim = THICKNESS_only.values[~np.isnan(DENSEPS1.Density)]
DENSEPS_lim = DENSEPS1[~np.isnan(DENSEPS1.Density)]

MEGA_SPREADSHEET_lim = MEGA_SPREADSHEET_lim[['AVG_BL_BP', 'PC1_BP', 'PC2_BP', 'PC3_BP', 'PC4_BP', 'm_BP', 'b_BP', 'PH_1635_BP', 'PH_1635_PC1_BP', 'PH_1635_PC2_BP']]
MEGA_SPREADSHEET_norm = MEGA_SPREADSHEET_lim.divide(THICKNESS_lim, axis=0) * 100
plots = MEGA_SPREADSHEET_norm.join([nbo_t_lim])

plots = plots.join([MICOMP_merge])
plots = plots.join([DENSEPS_lim[['Density_Sat', 'Tau', 'Eta']]])
plots = plots.join([H2OCO2_1['H2OT_MEAN']])
plots = plots.rename(columns={0: 'NBO_T'})
plots = plots[abs(plots.NBO_T - np.mean(plots.NBO_T)) < 3 * np.std(plots.NBO_T)]
plots = plots[abs(plots.SiO2 - np.mean(plots.SiO2)) < 3 * np.std(plots.SiO2)]
plots = plots[abs(plots.Al2O3 - np.mean(plots.Al2O3)) < 3 * np.std(plots.Al2O3)]
plots = plots[abs(plots.MnO - np.mean(plots.MnO)) < 2 * np.std(plots.MnO)]

plots_lim = plots[abs(plots.AVG_BL_BP - np.mean(plots.AVG_BL_BP)) < 2 * np.std(plots.AVG_BL_BP)]
plots_lim = plots_lim[abs(plots_lim.PC1_BP - np.mean(plots_lim.PC1_BP)) < 2 * np.std(plots_lim.PC1_BP)]
plots_lim = plots_lim[abs(plots_lim.PC2_BP - np.mean(plots_lim.PC2_BP)) < 2 * np.std(plots_lim.PC2_BP)]
plots_lim = plots_lim[abs(plots_lim.PC3_BP - np.mean(plots_lim.PC3_BP)) < 2 * np.std(plots_lim.PC3_BP)]
plots_lim = plots_lim[abs(plots_lim.PC4_BP - np.mean(plots_lim.PC4_BP)) < 2 * np.std(plots_lim.PC4_BP)]

plots_lim = plots_lim[['AVG_BL_BP', 'PC1_BP', 'PC2_BP', 'PC3_BP', 'PC4_BP', 'm_BP', 'b_BP', 'PH_1635_BP', 'PH_1635_PC1_BP', 'PH_1635_PC2_BP', 'SiO2', 'TiO2', 
                       'Al2O3', 'Fe2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2OT_MEAN', 'NBO_T', 'Tau', 'Eta', 'Density_Sat']]
plots_f = plots_lim.rename(columns={'H2OT_MEAN': 'H2O'})

# p = sns.pairplot(plots_f, kind='kde', corner=True)
# plt.tight_layout()
# plt.savefig('pairplot.pdf')

# %% 

MI_Composition = MICOMP_merge
MI_Composition = MI_Composition.join(plots_f['H2O'])

molar_mass = {'SiO2': 60.08, 'TiO2': 79.866, 'Al2O3': 101.96, 'Fe2O3': 159.69, 'FeO': 71.844, 'MnO': 70.9374, 
            'MgO': 40.3044, 'CaO': 56.0774, 'Na2O': 61.9789, 'K2O': 94.2, 'P2O5': 141.9445, 'H2O': 18.01528, 'CO2': 44.01}

# Create an empty dataframe to store the mole fraction of each oxide in the MI composition
mol = pd.DataFrame()
# Calculate the mole fraction of each oxide by dividing its mole fraction by its molar mass
for oxide in MI_Composition:
    mol[oxide] = MI_Composition[oxide]/molar_mass[oxide]

# Calculate the total mole fraction for the MI composition
mol_tot = pd.DataFrame()
mol_tot = mol.sum(axis = 1)

mol_frac = pd.DataFrame()

for oxide in MI_Composition:
    mol_frac[oxide] = mol[oxide]/mol_tot

plots_molfrac = plots_lim[['AVG_BL_BP', 'PC1_BP', 'PC2_BP', 'PC3_BP', 'PC4_BP', 'm_BP', 'b_BP', 'PH_1635_BP', 'PH_1635_PC1_BP', 'PH_1635_PC2_BP', 'NBO_T', 'Density_Sat', 'Tau', 'Eta']]

plots_molfrac = plots_molfrac.join([mol_frac])
plots_molfrac['Na_K'] = plots_molfrac.Na2O + plots_molfrac.K2O
plots_molfrac['FeT'] = plots_molfrac.FeO + plots_molfrac.Fe2O3/1.11134
plots_molfrac = plots_molfrac[['AVG_BL_BP', 'PC1_BP', 'PC2_BP', 'PC3_BP', 'PC4_BP', 'm_BP', 'b_BP', 'PH_1635_BP', 'PH_1635_PC1_BP', 'PH_1635_PC2_BP', 'SiO2', 'TiO2', 
                               'Al2O3', 'Fe2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O', 'NBO_T', 'FeT', 'Na_K','Tau', 'Eta', 'Density_Sat']]

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

chem_param = ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O']
molfrac_chem = plots_molfrac[['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(molfrac_chem)
# scaled_data[np.isnan(scaled_data)] = 0.000001

# Apply PCA
pca = PCA()
principalComponents = pca.fit_transform(scaled_data)

# Convert to a DataFrame for easier viewing
principal_df = pd.DataFrame(data = principalComponents, columns = ['PC' + str(i) for i in range(1, len(chem_param))])
principal_df.index = plots_molfrac.index

# p = sns.pairplot(plots_molfrac, kind='kde', corner=True) # kind='kde'
# plt.tight_layout()
# plt.savefig('pairplot_molfrac_kde.pdf')



# %% 

def biplot(reduced_data, pca):
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # scatterplot of the reduced data 
    ax.scatter(reduced_data[:, 0], reduced_data[:, 1], facecolors='tab:blue', edgecolors='k', s=70, alpha=0.5)
    
    # Add feature vectors (loadings)
    feature_vectors = pca.components_.T
    arrow_size, text_pos = 7.0, 8.0
    for i, v in enumerate(feature_vectors):
        ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], head_width=0.2, head_length=0.2, linewidth=2, color='tab:red')
        ax.text(v[0]*text_pos, v[1]*text_pos, chem_param[i], color='black', ha='center', va='center', fontsize=12)
    
    ax.set_xlabel("PC1", fontsize=14)
    ax.set_ylabel("PC2", fontsize=14)
    ax.set_title("PCA biplot", fontsize=16)
    return ax

biplot(principalComponents, pca)
plt.show()



# %% 


plots_pca = plots_molfrac[['AVG_BL_BP', 'PC1_BP', 'PC2_BP', 'PC3_BP', 'PC4_BP', 'm_BP', 'b_BP', 'PH_1635_BP', 'PH_1635_PC1_BP', 'PH_1635_PC2_BP',]].join(principal_df[['PC1', 'PC2', 'PC3', 'PC4']])
plots_pca = plots_pca.join(plots_molfrac[['NBO_T', 'Tau', 'Eta', 'Density_Sat']])
p = sns.pairplot(plots_pca, kind='kde', corner=True)
plt.tight_layout()
plt.savefig('pairplot_pca.pdf')

# %%

corr = plots_lim.corr()
corr.to_csv('correlation.csv')
display(corr)

corr_frac = plots_molfrac.corr()
corr_frac.to_csv('correlation_molfrac.csv')
display(corr_frac)

corr_pca = plots_pca.corr()
corr_pca.to_csv('correlation_pca.csv')
display(corr_pca)

