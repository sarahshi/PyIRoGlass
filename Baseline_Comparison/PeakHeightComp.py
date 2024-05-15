# %% -*- coding: utf-8 -*-
""" Created on June 12, 2021 // @author: Sarah Shi """

# Import packages
import sys
import numpy as np
import pandas as pd
import scipy

sys.path.append('../src/')
import PyIRoGlass as pig

from matplotlib import pyplot as plt
from matplotlib import rc
import seaborn as sns
import matplotlib as mpl
mpl.rcParams['mathtext.default'] = 'regular'

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
rc('font',**{'family':'Avenir', 'size': 20})
plt.rcParams['pdf.fonttype'] = 42

plt.rcParams["xtick.major.size"] = 4
plt.rcParams["ytick.major.size"] = 4
plt.rcParams["xtick.labelsize"] = 20
plt.rcParams["ytick.labelsize"] = 20
plt.rcParams["axes.titlesize"] = 22
plt.rcParams["axes.labelsize"] = 22

# %% 

MEGA_SPREADSHEET = pd.read_csv('../FINALDATA/STD_DF.csv', index_col=0) 
CONC = pd.read_csv('../FINALDATA/STD_H2OCO2.csv', index_col=0) 
HJ = pd.read_csv('PyIRoGlass-LHJ_091623.csv', index_col=0)[['Repeats', 'PH_1430', 'PH_1515', 'Thickness']]

merge = pd.concat([MEGA_SPREADSHEET, CONC, HJ], axis=1)
merge = merge.loc[HJ.index]

columns_to_normalize = ['PH_1515', 'PH_1515_BP', 'PH_1515_STD', 'PH_1430', 'PH_1430_BP', 'PH_1430_STD']
for col in columns_to_normalize:
    merge[f'{col}_norm'] = merge[col] / merge['Thickness'] * 50

merge['Py_Devol_1430'] = merge['PH_1430_BP_norm'] / merge['PH_1430_norm']
merge['Py_Devol_1515'] = merge['PH_1515_BP_norm'] / merge['PH_1515_norm']
stats = {temp: (merge[f'Py_Devol_{temp}'].mean(), merge[f'Py_Devol_{temp}'].std()) for temp in ['1430', '1515']}
mean_1430, std_1430 = stats['1430']
mean_1515, std_1515 = stats['1515']

mask_no_nd70 = ~merge.index.astype(str).str.contains('ND70')
mask_py_devol_1430 = abs(merge['Py_Devol_1430'] - mean_1430) < 1.5 * std_1430
mask_py_devol_1515 = abs(merge['Py_Devol_1515'] - mean_1515) < 1.55 * std_1515
merge_filt_no_nd70 = merge.loc[mask_no_nd70 & mask_py_devol_1430 & mask_py_devol_1515]

merge_nd70 = merge.loc[merge.index.astype(str).str.contains('ND70')]
merge_filt = pd.concat([merge_filt_no_nd70, merge_nd70])

merge_filt.to_csv('PHComparison.csv')
merge_nd70.to_csv('PHComparison_ND70.csv')

standards = pd.read_csv('PyIRoGlass_Standards.csv', index_col=0)
merge_std = pd.concat([MEGA_SPREADSHEET, standards, CONC], axis=1)
merge_std.to_csv('Comparison_Standards.csv')

# %% 

def Error_Prop(mean_std, mean_mean, std_mean): 
    
    sigma_analysis = mean_std/mean_mean
    sigma_repeat = std_mean/mean_mean
    sigma_prop = np.where(sigma_repeat.isna(), sigma_analysis,
        np.sqrt(sigma_analysis**2 + sigma_repeat**2))
    uncert_prop = mean_mean * sigma_prop

    return uncert_prop

def NBO_T(MI_Composition): 

    # Define a dictionary of molar masses for each oxide
    molar_mass = {'SiO2': 60.08, 'TiO2': 79.866, 'Al2O3': 101.96, 'Fe2O3': 159.69, 'FeO': 71.844, 'MnO': 70.9374, 
                'MgO': 40.3044, 'CaO': 56.0774, 'Na2O': 61.9789, 'K2O': 94.2, 'P2O5': 141.9445}

    # Create an empty dataframe to store the mole fraction of each oxide in the MI composition
    mol = pd.DataFrame()
    # Calculate the mole fraction of each oxide by dividing its mole fraction by its molar mass
    for oxide in MI_Composition:
        mol[oxide] = MI_Composition[oxide]/molar_mass[oxide]

    t = mol['SiO2'] + 2*mol['Al2O3'] + 2*mol['Fe2O3']
    o = mol.sum(axis=1) + mol['SiO2'] + mol['TiO2'] + 2*mol['Al2O3'] + 2*mol['Fe2O3'] + 4*mol['P2O5']

    nbo_t = ((2*o)-(4*t))/t

    return nbo_t

# %% 

sz = 80
line = np.array([0, 3])
ticks = np.arange(0, 3, 0.5)
tick_labels = [str(t) if t % 0.5 == 0 else "" for t in ticks]

slope0, intercept0, r_value0, _, _ = scipy.stats.linregress(merge_filt.PH_1515_norm, merge_filt.PH_1515_BP_norm)
ccc0 = pig.calculate_CCC(merge_filt.PH_1515_norm, merge_filt.PH_1515_BP_norm)
rmse0 = pig.calculate_RMSE(merge_filt.PH_1515_norm-merge_filt.PH_1515_BP_norm)

slope1, intercept1, r_value1, _, _ = scipy.stats.linregress(merge_filt.PH_1430_norm, merge_filt.PH_1430_BP_norm)
ccc1 = pig.calculate_CCC(merge_filt.PH_1430_norm, merge_filt.PH_1430_BP_norm)
rmse1 = pig.calculate_RMSE(merge_filt.PH_1430_norm-merge_filt.PH_1430_BP_norm)

fig, ax = plt.subplots(2, 2, figsize=(13, 13))
ax=ax.flatten()
ax[0].plot(line, line, 'k', lw=1, zorder=0, label='1-1 Line')
ax[0].fill_between(line, 0.9*line, 1.1*line, color='gray', edgecolor=None, alpha=0.25, label='10% Uncertainty')
ax[0].scatter(merge_filt.PH_1515_norm, merge_filt.PH_1515_BP_norm, s=sz, c='#0C7BDC', ec='#171008', lw=0.5, zorder=20,)
ax[0].errorbar(merge_filt.PH_1515_norm, merge_filt.PH_1515_BP_norm, yerr=merge_filt.PH_1515_STD/merge_filt.Thickness*200, xerr=merge_filt['PH_1515']/merge_filt.Thickness*50*0.05, fmt='none', lw=0.5, c='k', zorder=10)
ax[0].annotate(r'A. $\mathregular{CO_{3, 1515}^{2-}}, n=$'+str(len(merge_filt)), xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va='bottom', size=20)
ax[0].set_xlim([0, 2.5])
ax[0].set_ylim([0, 2.5])
ax[0].set_xticks(ticks)
ax[0].set_yticks(ticks)
ax[0].set_xticklabels(tick_labels)
ax[0].set_ylabel('PyIRoGlass Peak Heights')
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5, labelbottom = False)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[0].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value0**2, 3)), xy=(0.03, 0.7975), xycoords="axes fraction", fontsize=16)
ax[0].annotate("RMSE="+str(np.round(rmse0, 3))+"; RRMSE="+str(np.round(pig.calculate_RRMSE(merge_filt.PH_1515_norm, merge_filt.PH_1515_BP_norm)*100, 3))+'%', xy=(0.03, 0.84), xycoords="axes fraction", fontsize=16)
ax[0].annotate("CCC="+str(np.round(ccc0, 3)), xy=(0.03, 0.88), xycoords="axes fraction", fontsize=16)
ax[0].annotate("m="+str(np.round(slope0, 3)), xy=(0.03, 0.76), xycoords="axes fraction", fontsize=16)
ax[0].annotate("b="+str(np.round(intercept0, 3)), xy=(0.03, 0.72), xycoords="axes fraction", fontsize=16)

ax[1].plot(line, line, 'k', lw=1, zorder=0, label='1-1 Line')
ax[1].fill_between(line, 0.9*line, 1.1*line, color='gray', edgecolor=None, alpha=0.25, label='10% Uncertainty')
ax[1].scatter(merge_filt.PH_1430_norm, merge_filt.PH_1430_BP_norm, s=sz, c='#0C7BDC', ec='#171008', lw=0.5, zorder=20)
ax[1].errorbar(merge_filt.PH_1430_norm, merge_filt.PH_1430_BP_norm, yerr=merge_filt.PH_1430_STD/merge_filt.Thickness*200, xerr=merge_filt['PH_1430']/merge_filt.Thickness*50*0.05, fmt='none', lw=0.5, c='k', zorder=10)
ax[1].annotate(r'B. $\mathregular{CO_{3, 1430}^{2-}}, n=$'+str(len(merge_filt)), xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va='bottom', size=20)
ax[1].set_xlim([0, 2.5])
ax[1].set_ylim([0, 2.5])
ax[1].set_xticks(ticks)  # Set x ticks
ax[1].set_yticks(ticks)  # Set y ticks
ax[1].set_xticklabels(tick_labels)
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5, labelbottom = False)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
ax[1].legend(loc='lower right', labelspacing = 0.2, handletextpad = 0.25, handlelength = 1.00, prop={'size': 16}, frameon=False)
ax[1].annotate("$\mathregular{R^{2}}$="+str(np.round(r_value1**2, 3)), xy=(0.03, 0.7975), xycoords="axes fraction", fontsize=16)
ax[1].annotate("CCC="+str(np.round(ccc1, 3)), xy=(0.03, 0.88), xycoords="axes fraction", fontsize=16)
ax[1].annotate("RMSE="+str(np.round(rmse1, 3))+"; RRMSE="+str(np.round(pig.calculate_RRMSE(merge_filt.PH_1430_norm, merge_filt.PH_1430_BP_norm)*100, 3))+'%', xy=(0.03, 0.84), xycoords="axes fraction", fontsize=16)
ax[1].annotate("m="+str(np.round(slope1, 3)), xy=(0.03, 0.76), xycoords="axes fraction", fontsize=16)
ax[1].annotate("b="+str(np.round(intercept1, 3)), xy=(0.03, 0.72), xycoords="axes fraction", fontsize=16)

ticks_y = np.arange(0.85, 1.05, 0.05)
tick_labels_y = [f"{t:.2f}" for t in ticks_y]

sc2 = ax[2].scatter(merge_filt.PH_1515_norm, (merge_filt['Py_Devol_1515']), s=sz, 
                    c='#0C7BDC', ec='#171008', lw=0.5, 
                    zorder=20)
sc1 = ax[3].scatter(merge_filt.PH_1430_norm, (merge_filt['Py_Devol_1430']), s=sz, 
                    c='#0C7BDC', ec='#171008', lw=0.5, 
                    zorder=20)
ax[2].axhline(np.mean(merge_filt['Py_Devol_1515']), color='k', linestyle='--', dashes = (10, 10), linewidth=0.75,)
ax[2].annotate(r'C. $\mathregular{CO_{3, 1515}^{2-}}, n=$'+str(len(merge_filt)), xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va='bottom', size=20)
ax[2].text(1.9, 0.945, r'$\overline{\frac{P}{D}}$='+str(round(np.mean(merge_filt['Py_Devol_1515']), 3)), ha='left', va='bottom', size=20)
ax[2].fill_between(line, np.mean(merge_filt['Py_Devol_1515'])-np.std(merge_filt['Py_Devol_1515']), np.mean(merge_filt['Py_Devol_1515'])+np.std(merge_filt['Py_Devol_1515']), color = 'k', alpha=0.10, edgecolor = None,
    zorder=-5, label='68% Confidence Interval')
ax[2].set_xlabel('Measured Devolatilized Spectrum Peak Heights')
ax[2].set_ylabel('PyIRoGlass/Measured Devolatilized Spectrum Peak Height')
ax[2].set_xlim([0, 2.5])
ax[2].set_xticks(ticks)
ax[2].set_xticklabels(tick_labels)
ax[2].set_ylim([0.85, 1.05])
ax[2].set_yticks(ticks_y)
ax[2].set_yticklabels(tick_labels_y)
ax[2].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[2].tick_params(axis="y", direction='in', length=5, pad=6.5)

ax[3].axhline(np.mean(merge_filt['Py_Devol_1430']), color='k', linestyle='--', dashes = (10, 10), linewidth=0.75, label='Mean')
ax[3].annotate(r'D. $\mathregular{CO_{3, 1430}^{2-}}, n=$'+str(len(merge_filt)), xy=(0.03, 0.92), xycoords='axes fraction', ha='left', va='bottom', size=20)
ax[3].text(1.9, 0.965, r'$\overline{\frac{P}{D}}$='+str(round(np.mean(merge_filt['Py_Devol_1430']), 3)), ha='left', va='bottom', size=20)
ax[3].fill_between(line, np.mean(merge_filt['Py_Devol_1430'])-np.std(merge_filt['Py_Devol_1430']), np.mean(merge_filt['Py_Devol_1430'])+np.std(merge_filt['Py_Devol_1430']), color = 'k', alpha=0.10, edgecolor = None,
    zorder=-5, label='68% Confidence Interval')
ax[3].legend(loc='lower right', labelspacing = 0.2, handletextpad = 0.25, handlelength = 1.00, prop={'size': 16}, frameon=False)
ax[3].set_xlim([0, 2.5])
ax[3].set_xticks(ticks)
ax[3].set_xticklabels(tick_labels)
ax[3].set_ylim([0.85, 1.05])
ax[3].set_yticks(ticks_y)
ax[3].set_yticklabels(tick_labels_y)
ax[3].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[3].tick_params(axis="y", direction='in', length=5, pad=6.5)
plt.tight_layout()
# plt.savefig('PHCombined_new.pdf', bbox_inches='tight', pad_inches = 0.025)

# %% 

col_means = ['PH_1515_norm', 'PH_1515_BP_norm', 'PH_1515_STD_norm', 'PH_1430_norm', 'PH_1430_BP_norm', 'PH_1430_STD_norm', 
             'Py_Devol_1430', 'Py_Devol_1515', 'H2Ot_MEAN', 'H2Ot_STD', 'CO2_MEAN', 'CO2_STD']
counts = merge_filt.groupby('Repeats')['PH_1515_norm'].count()
std = merge_filt.groupby('Repeats')[col_means].std()

means = merge_filt.groupby('Repeats')[col_means].mean()
means['Counts'] = counts
means['PH_1515_STD_net'] = Error_Prop(means['PH_1515_STD_norm'], means['PH_1515_BP_norm'], std['PH_1515_BP_norm'])
means['PH_1430_STD_net'] = Error_Prop(means['PH_1430_STD_norm'], means['PH_1430_BP_norm'], std['PH_1430_BP_norm'])
means['H2Ot_STD_net'] = Error_Prop(means['H2Ot_STD'], means['H2Ot_MEAN'], std['H2Ot_MEAN'])
means['CO2_STD_net']  = Error_Prop(means['CO2_STD'], means['CO2_MEAN'], std['CO2_MEAN'])

def assign_composition(sample_name):
    if 'CI_LMT' in sample_name:
        return 'Synthetic Basaltic Andesite'
    elif 'CI_Ref_bas' in sample_name:
        return 'Synthetic Basanite (El Hierro)'
    elif 'CI_Ref_' in sample_name:
        return 'Synthetic MORB'
    elif 'ND70' in sample_name:
        return 'Synthetic Basalt'
    else:
        return 'Unknown'
means['Composition'] = means.index.map(assign_composition)

means = means[['Composition', 'Counts',
               'PH_1515_norm', 'PH_1515_BP_norm', 'PH_1515_STD_norm', 
               'PH_1430_norm', 'PH_1430_BP_norm', 'PH_1430_STD_norm',
               'H2Ot_MEAN', 'H2Ot_STD_net', 'CO2_MEAN', 'CO2_STD_net']]

display(means)
# means.to_csv('CI_statistics.csv')

# %% 

sdl1 = pig.SampleDataLoader(chemistry_thickness_path='../Inputs/FuegoChemThick.csv')
MICOMP0, THICKNESS0 = sdl1.load_chemistry_thickness()
sdl2 = pig.SampleDataLoader(chemistry_thickness_path='../Inputs/StandardChemThick.csv')
MICOMP1, THICKNESS1 = sdl2.load_chemistry_thickness()
sdl3 = pig.SampleDataLoader(chemistry_thickness_path='../Inputs/FuegoRHChemThick.csv')
MICOMP2, THICKNESS2 = sdl3.load_chemistry_thickness()

MICOMP = pd.concat([MICOMP0, MICOMP1, MICOMP2])
THICKNESS = pd.concat([THICKNESS0, THICKNESS1, THICKNESS2])

nbo_t = NBO_T(MICOMP)
nbo_t_lim = nbo_t 
THICKNESS_lim = THICKNESS[MICOMP.Fe2O3 != 0]
nbo_t_lim = nbo_t[MICOMP.Fe2O3 != 0]
THICKNESS_only = THICKNESS_lim.Thickness

MICOMP_merge = MICOMP[MICOMP.Fe2O3 != 0]

MEGA_SPREADSHEET0 = pd.read_csv('../FINALDATA/FUEGO_DF.csv', index_col=0).rename_axis('Sample')
MEGA_SPREADSHEET1 = pd.read_csv('../FINALDATA/STD_DF.csv', index_col=0).rename_axis('Sample')
MEGA_SPREADSHEET2 = pd.read_csv('../FINALDATA/FRH_DF.csv', index_col=0).rename_axis('Sample')
H2OCO20 = pd.read_csv('../FINALDATA/FUEGO_H2OCO2.csv', index_col=0).rename_axis('Sample')
H2OCO21 = pd.read_csv('../FINALDATA/STD_H2OCO2.csv', index_col=0).rename_axis('Sample')
H2OCO22 = pd.read_csv('../FINALDATA/FRH_H2OCO2.csv', index_col=0).rename_axis('Sample')

MEGA_SPREADSHEET = pd.concat([MEGA_SPREADSHEET0, MEGA_SPREADSHEET1, MEGA_SPREADSHEET2])
MEGA_SPREADSHEET_lim = MEGA_SPREADSHEET[MICOMP.Fe2O3 != 0]

H2OCO2 = pd.concat([H2OCO20, H2OCO21, H2OCO22])
H2OCO2_lim = H2OCO2[MICOMP.Fe2O3 != 0]

MEGA_SPREADSHEET_lim = MEGA_SPREADSHEET_lim[['AVG_BL_BP', 'PC1_BP', 'PC2_BP', 'PC3_BP', 'PC4_BP', 'm_BP', 'b_BP', 'PH_1635_BP', 'PH_1635_PC1_BP', 'PH_1635_PC2_BP']]
MEGA_SPREADSHEET_norm = MEGA_SPREADSHEET_lim.divide(THICKNESS_lim['Thickness'], axis=0) * 100
plots = MEGA_SPREADSHEET_norm.join([nbo_t_lim])

plots = plots.merge(MICOMP_merge, left_index=True, right_index=True, how='left')
plots = plots.merge(H2OCO2_lim[['H2Ot_MEAN', 'Density_Sat', 'Tau', 'Eta']], left_index=True, right_index=True, how='left')
plots = plots.rename(columns={0: 'NBO_T'})
plots = plots[abs(plots.NBO_T - np.mean(plots.NBO_T)) < 2 * np.std(plots.NBO_T)]
plots = plots[abs(plots.SiO2 - np.mean(plots.SiO2)) < 2 * np.std(plots.SiO2)]

plots_lim = plots[abs(plots.AVG_BL_BP - np.mean(plots.AVG_BL_BP)) < 2 * np.std(plots.AVG_BL_BP)]
plots_lim = plots_lim[abs(plots_lim.PC1_BP - np.mean(plots_lim.PC1_BP)) < 2 * np.std(plots_lim.PC1_BP)]
plots_lim = plots_lim[abs(plots_lim.PC2_BP - np.mean(plots_lim.PC2_BP)) < 2 * np.std(plots_lim.PC2_BP)]
plots_lim = plots_lim[abs(plots_lim.PC3_BP - np.mean(plots_lim.PC3_BP)) < 2 * np.std(plots_lim.PC3_BP)]
plots_lim = plots_lim[abs(plots_lim.PC4_BP - np.mean(plots_lim.PC4_BP)) < 2 * np.std(plots_lim.PC4_BP)]

plots_lim = plots_lim[['AVG_BL_BP', 'PC1_BP', 'PC2_BP', 'PC3_BP', 'PC4_BP', 'SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 
                       'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2Ot_MEAN', 'NBO_T', 'Tau', 'Eta', 'Density_Sat']] 
plots_f = plots_lim.rename(columns={'H2Ot_MEAN': 'H2O'})

corr = plots_lim.corr()
display(corr)

# %% 

MI_Composition = MICOMP_merge
MI_Composition = MI_Composition.join(plots_f['H2O'])

molar_mass = {'SiO2': 60.08, 'TiO2': 79.866, 'Al2O3': 101.96, 'Fe2O3': 159.69, 'FeO': 71.844, 'MnO': 70.9374, 
            'MgO': 40.3044, 'CaO': 56.0774, 'Na2O': 61.9789, 'K2O': 94.2, 'P2O5': 141.9445, 'H2O': 18.01528, 'CO2': 44.01}

mol = pd.DataFrame()
for oxide in MI_Composition:
    mol[oxide] = MI_Composition[oxide]/molar_mass[oxide]
mol_tot = pd.DataFrame()
mol_tot = mol.sum(axis = 1)

mol_frac=pd.DataFrame()
for oxide in MI_Composition:
    mol_frac[oxide] = mol[oxide]/mol_tot

plots_molfrac=plots_lim[['AVG_BL_BP', 'PC1_BP', 'PC2_BP', 'PC3_BP', 'PC4_BP', 'NBO_T', 'Density_Sat', 'Tau', 'Eta']]
plots_molfrac=plots_molfrac.join([mol_frac])
plots_molfrac['Na_K'] = plots_molfrac.Na2O + plots_molfrac.K2O
plots_molfrac['FeT'] = plots_molfrac.FeO + plots_molfrac.Fe2O3/1.11134
plots_molfrac=plots_molfrac[['AVG_BL_BP', 'PC1_BP', 'PC2_BP', 'PC3_BP', 'PC4_BP', 'SiO2', 'TiO2', 
                               'Al2O3', 'Fe2O3', 'FeO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'H2O', 'NBO_T', 'Tau', 'Eta', 'Density_Sat']]
corr_frac=plots_molfrac.corr()

# %% Figure adapted in illustrator for Figure 13. 

plots_molfrac_lim = plots_molfrac[['AVG_BL_BP', 'PC1_BP', 'PC2_BP', 'PC3_BP', 'PC4_BP', 'SiO2', 'TiO2', 
                'Al2O3', 'Fe2O3', 'FeO', 'MgO', 'CaO', 'H2O', 'NBO_T']]
zero_rows = plots_molfrac_lim[plots_molfrac_lim['FeO'] == 0]
non_zero_rows = plots_molfrac_lim[plots_molfrac_lim['FeO'] != 0]

zero_sub = zero_rows[abs(zero_rows['AVG_BL_BP'] - np.mean(zero_rows['AVG_BL_BP'])) < 1.5 * np.std(zero_rows['AVG_BL_BP'])]
resampled_df = pd.concat([zero_sub, non_zero_rows], axis=0).reset_index(drop=True)

sns.set(rc={'figure.figsize':(10, 10)})
sns.set_style("white")
g = sns.pairplot(resampled_df, diag_kind="kde", corner=True, plot_kws={'alpha': 0.5, 'color': '#95b0f2'}, height=1)
g.map_lower(sns.kdeplot, levels=5, color="#022270", linewidth=0.01)
# plt.savefig('BLChem_lim.pdf')

# %%
