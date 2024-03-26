# %% 

""" Created on March 8, 2024 // @author: Sarah Shi and Bill Menke """

import os
import sys
import pickle 
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import rc
from scipy.integrate import simps

sys.path.append('../src/')
import PyIRoGlass as pig

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

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['xtick.major.pad'] = 6.5
plt.rcParams['ytick.major.pad'] = 6.5

# method treats d, z, m1 and m2 as unknowns. model f(i) = 0 = m1 + m2*z(i) - d(i);
# unknowns:  m1, m2 and the predicted (z,d)'s

# %% 

cols = ['ERDA_H2O',
        'ERDA_H2O_STD',
        'ERDA_SIMS_H2O',
        'ERDA_SIMS_H2O_STD',
        'NRA_CO2',
        'NRA_CO2_STD',
        'PH_3550_M',
        'PH_3550_STD',
        'PH_1515_BP',
        'PH_1515_STD',
        'STD_1515_BP',
        'PH_1430_BP',
        'PH_1430_STD',
        'STD_1430_BP',
        'Density_Sat',
        'Thickness',
        'Sigma_Thickness',
        'Repeats',
        'Tau',
        'Eta'
        ]

cols_lim = ['ERDA_H2O',
        'ERDA_H2O_STD',
        'ERDA_SIMS_H2O',
        'ERDA_SIMS_H2O_STD',
        'NRA_CO2',
        'NRA_CO2_STD',
        'PH_3550_M',
        'PH_3550_STD',
        'PH_1515_BP',
        'PH_1515_STD',
        'STD_1515_BP',
        'PH_1430_BP',
        'PH_1430_STD',
        'STD_1430_BP',
        'Density_Sat',
        'Thickness',
        'Sigma_Thickness',
        'Repeats',
        'Area_3550_M',
        'Area_3550_STD',
        'Tau',
        'Eta'
        ]

df_orig = pd.read_csv('PHComparison_ERDA_NRA.csv', index_col=0)
df = df_orig[cols].groupby('Repeats').mean()
df_counts_1 = df_orig.groupby('Repeats').count().iloc[:, 0].rename('count')
df = df.join(df_counts_1)


df_h2o = df_orig[(df_orig.index != 'ND70_04_02_06032022_150x150_sp1') &
                 (df_orig.index != 'ND70_04_02_06032022_150x150_sp2') &
                 (df_orig.index != 'ND70_04_02_06032022_150x150_sp3') & 
                 (df_orig.index != 'ND70_03_01_06032022_150x150_sp2') & 
                 (df_orig.index != 'ND70_03_01_06032022_150x150_sp3') 
                 ]
sloader = pig.SampleDataLoader(spectrum_path='../Inputs/TransmissionSpectra/Standards/')
sdfs_dict = sloader.load_spectrum_directory()

nd70_dict = {key: value for key, value in sdfs_dict.items() if 'ND70' in key}
nd70_df_lim = df_h2o[(df_h2o.index != 'ND70_04_02_06032022_150x150_sp1') &
                     (df_h2o.index != 'ND70_04_02_06032022_150x150_sp2') &
                     (df_h2o.index != 'ND70_04_02_06032022_150x150_sp3') &
                     (df_h2o.index != 'ND70_03_01_06032022_150x150_sp2') &
                     (df_h2o.index != 'ND70_03_01_06032022_150x150_sp3') &
                     (~df_h2o.index.str.contains("Glass_")) & 
                     (~df_h2o.index.str.contains("_org_")) & 
                     (~df_h2o.index.str.contains("_degassed_"))]
nd70_dict_lim = {key: value for key, value in nd70_dict.items() if key in nd70_df_lim.index}

df_h2o = df_h2o.copy() 

for files, data in nd70_dict_lim.items():
    file_path = '../PKLFILES/ND70/' + files + ".pkl"
    with open(file_path, "rb") as handle:
        als_bls=pickle.load(handle)
    H2Ot_3550_results = als_bls["H2Ot_3550_results"]
    subtract0 = (H2Ot_3550_results[0]["peak_fit"]["Absorbance"] - 
                 H2Ot_3550_results[0]["peak_fit"]["Baseline_MIR"])
    subtract1 = (H2Ot_3550_results[1]["peak_fit"]["Absorbance"] - 
                 H2Ot_3550_results[1]["peak_fit"]["Baseline_MIR"])
    subtract2 = (H2Ot_3550_results[2]["peak_fit"]["Absorbance"] - 
                 H2Ot_3550_results[2]["peak_fit"]["Baseline_MIR"])
    asimps0 = simps(subtract0.values, subtract0.index)
    asimps1 = simps(subtract1.values, subtract1.index)
    asimps2 = simps(subtract2.values, subtract2.index)
    df_h2o.loc[files, 'Area_3550_M'] = np.mean([asimps0, asimps1, asimps2])
    df_h2o.loc[files, 'Area_3550_STD'] = np.std([asimps0, asimps1, asimps2])

df_means = df_h2o[cols_lim].groupby('Repeats').mean()
df_counts_2 = df_h2o.groupby('Repeats').count().iloc[:, 0].rename('count')
df_h2o = df_means.join(df_counts_2)

h2o_erda = df_h2o['ERDA_SIMS_H2O'] 
h2o_erda_std = df_h2o['ERDA_SIMS_H2O_STD'] 
h2o_range = np.linspace(0, 7, 5)

h2o_y = 18.01528*df_h2o['PH_3550_M']/(df_h2o['Density_Sat']*(df_h2o['Thickness']/1e6))
sigma_h2o_y = h2o_y * (np.mean(np.sqrt((df['PH_3550_STD']/df['PH_3550_M'])**2 + (0.025**2) + (df['Sigma_Thickness']/df['Thickness'])**2)))

co2_nra = df['NRA_CO2']/10000
co2_nra_x = pd.concat([co2_nra, co2_nra])
co2_nra_std = df['NRA_CO2_STD']/10000 
co2_nra_std_x = pd.concat([co2_nra_std, co2_nra_std])
co2_range = np.linspace(0, 2, 5)

co2_y_1430 = 44.01*df['PH_1430_BP']/(df['Density_Sat']*(df['Thickness']/1e6))
co2_y_1515 = 44.01*df['PH_1515_BP']/(df['Density_Sat']*(df['Thickness']/1e6))
co2_y = pd.concat([co2_y_1430, co2_y_1515])
sigma_co2_y_1430 = co2_y_1430 * (np.mean(np.sqrt((df['PH_1430_STD']/df['PH_1430_BP'])**2 + (0.025**2) + (df['Sigma_Thickness']/df['Thickness'])**2)))
# sigma_co2_y_1430.iloc[-2:] = sigma_co2_y_1430.iloc[-2]*0.75
sigma_co2_y_1515 = co2_y_1515 * (np.mean(np.sqrt((df['PH_1515_STD']/df['PH_1515_BP'])**2 + (0.025**2) + (df['Sigma_Thickness']/df['Thickness'])**2)))
# sigma_co2_y_1515.iloc[-2:] = sigma_co2_y_1515.iloc[-2]*0.75
sigma_co2_y = pd.concat([sigma_co2_y_1430, sigma_co2_y_1515])

mest_1430, covm_est_1430, covy_1430 = pig.inversion(co2_nra, co2_y_1430, co2_nra_std, sigma_co2_y_1430, intercept_zero=True)
mls_1430, covls_1430 = pig.least_squares(co2_nra, co2_y_1430, sigma_co2_y_1430)
E_calib_1430, see_1430, r2_1430, rmse_1430, rrmse_1430, ccc_1430 = pig.inversion_fit_errors(co2_nra, co2_y_1430, mest_1430, covy_1430)
line_x_1430, line_y_1430, conf_low_1430, conf_up_1430, pred_low_1430, pred_up_1430 = pig.inversion_fit_errors_plotting(co2_nra, co2_y_1430, mest_1430)
co2_pred_1430 = pig.calculate_y_inversion(mest_1430, co2_nra)
print('mest_{1430} ' + str(mest_1430[1]))
print('95% CI final ' + str(2*np.sqrt(np.diag(covm_est_1430))[1]))

mest_1515, covm_est_1515, covy_1515 = pig.inversion(co2_nra, co2_y_1515, co2_nra_std, sigma_co2_y_1515, intercept_zero=True)
mls_1515, covls_1515 = pig.least_squares(co2_nra, co2_y_1515, sigma_co2_y_1515)
E_calib_1515, see_1515, r2_1515, rmse_1515, rrmse_1515, ccc_1515 = pig.inversion_fit_errors(co2_nra, co2_y_1515, mest_1515, covy_1515)
line_x_1515, line_y_1515, conf_low_1515, conf_up_1515, pred_low_1515, pred_up_1515 = pig.inversion_fit_errors_plotting(co2_nra, co2_y_1515, mest_1515)
print('mest_{1515} ' + str(mest_1515[1]))
print('95% CI final ' + str(2*np.sqrt(np.diag(covm_est_1515))[1]))

mest_co2, covm_est_co2, covy_co2 = pig.inversion(co2_nra_x, co2_y, co2_nra_std_x, sigma_co2_y, intercept_zero=True)
mls_co2, covls_co2 = pig.least_squares(co2_nra_x, co2_y, sigma_co2_y)
E_calib_co2, see_co2, r2_co2, rmse_co2, rrmse_co2, ccc_co2 = pig.inversion_fit_errors(co2_nra_x, co2_y, mest_co2, covy_co2)
line_x_co2, line_y_co2, conf_low_co2, conf_up_co2, pred_low_co2, pred_up_co2 = pig.inversion_fit_errors_plotting(co2_nra, co2_y, mest_co2)
print('mest_{carbonate} ' + str(mest_co2[1]))
print('95% CI final ' + str(2*np.sqrt(np.diag(covm_est_co2))[1]))

mest_h2o, covm_est_h2o, covy_h2o = pig.inversion(h2o_erda, h2o_y, h2o_erda_std, sigma_h2o_y, intercept_zero=True)
mls_h2o, covls_h2o = pig.least_squares(h2o_erda, h2o_y, sigma_h2o_y)
E_calib_h2o, see_h2o, r2_h2o, rmse_h2o, rrmse_h2o, ccc_h2o = pig.inversion_fit_errors(h2o_erda, h2o_y, mest_h2o, covy_h2o)
line_x_h2o, line_y_h2o, conf_low_h2o, conf_up_h2o, pred_low_h2o, pred_up_h2o = pig.inversion_fit_errors_plotting(h2o_erda, h2o_y, mest_h2o)
print('mest_{H2O} ' + str(mest_h2o[1]))
print('95% CI final ' + str(2*np.sqrt(np.diag(covm_est_h2o))[1]))

# %%

sz = 125
fig, ax = plt.subplots(1, 2, figsize = (14, 7))
ax = ax.flatten()
h2o_inv = pig.calculate_y_inversion(mest_h2o, h2o_range)
h2o_label = '$\mathregular{Ɛ_{H_{2}O_{3550}}}$=' + f'{round(mest_h2o[1],3)}(±{round(np.sqrt(np.diag(covm_est_h2o))[1],3)})' + ' L/mol$\mathregular{\cdot}$cm'
ax[0].plot(h2o_range, h2o_inv, 'k', lw=1, label=h2o_label)
ax[0].scatter(h2o_erda, h2o_y, s = sz, c = '#0C7BDC', ec = '#171008', lw= 0.5, zorder = 20, label='ND70-Series, n=6')
ax[0].errorbar(h2o_erda, h2o_y, xerr=h2o_erda_std, yerr=sigma_h2o_y, fmt='none', lw= 0.5, c = 'k', zorder = 10)
ax[0].fill_between(line_x_h2o, conf_low_h2o, conf_up_h2o, color = 'k', alpha=0.20, edgecolor=None,
                   zorder=-5, label='68% Confidence Interval')
ax[0].plot(line_x_h2o, pred_up_h2o, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[0].plot(line_x_h2o, pred_low_h2o, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[0].annotate("A.", xy=(0.03, 0.945), xycoords="axes fraction", ha='left', fontsize=20)
ax[0].annotate("$\mathregular{R^{2}}$="+str(np.round(r2_h2o, 3)), xy=(0.975, 0.125), xycoords="axes fraction", ha='right', fontsize=14)
ax[0].annotate("CCC="+str(np.round(ccc_h2o, 3)), xy=(0.975, 0.075), xycoords="axes fraction", ha='right', fontsize=14)
ax[0].annotate("RRMSE="+str(np.round(rrmse_h2o*100, 3))+'%', xy=(0.975, 0.025), xycoords="axes fraction", ha='right', fontsize=14)
ax[0].legend(loc=(0.02, 0.75), labelspacing=0.2, handletextpad=0.5, handlelength=0.75, prop={'size': 14}, frameon=False)
ax[0].set_xlabel('$\mathregular{H_{2}O}$, ERDA/SIMS (wt.%)')
ax[0].set_ylabel('18.02$\mathregular{\cdot A_{H_2O_{t, 3550}} \cdot}$Density$\mathregular{\cdot}$Thickness')
ax[0].set_xlim([0, 7])
ax[0].set_ylim([0, 400])
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

co2_inv = pig.calculate_y_inversion(mest_co2, co2_range)
co2_1430_label = '$\mathregular{Ɛ_{CO_{3, 1515/1430}^{2-}}}$=' + f'{round(mest_co2[1],3)}(±{round(np.sqrt(np.diag(covm_est_co2))[1],3)})' + ' L/mol$\mathregular{\cdot}$cm'
ax[1].plot(co2_range, co2_inv, 'k', lw=1, label=co2_1430_label)
ax[1].scatter(co2_nra_x, co2_y, s = sz, c='#E42211', marker = 's', ec = '#171008', lw= 0.5, zorder = 20, label='ND70-Series, n=14')
ax[1].errorbar(co2_nra_x, co2_y, xerr=co2_nra_std_x, yerr=sigma_co2_y, fmt='none', lw= 0.5, c = 'k', zorder = 10)
ax[1].fill_between(line_x_co2, conf_low_co2, conf_up_co2, color = 'k', alpha=0.20, edgecolor=None,
                   zorder=-5, label='68% Confidence Interval')
ax[1].plot(line_x_co2, pred_up_co2, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[1].plot(line_x_co2, pred_low_co2, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[1].annotate("B.", xy=(0.03, 0.945), xycoords="axes fraction", ha='left', fontsize=20)
ax[1].annotate("$\mathregular{R^{2}}$="+str(np.round(r2_co2, 3)), xy=(0.975, 0.125), xycoords="axes fraction", ha='right', fontsize=14)
ax[1].annotate("CCC="+str(np.round(ccc_co2, 3)), xy=(0.975, 0.075), xycoords="axes fraction", ha='right', fontsize=14)
ax[1].annotate("RRMSE="+str(np.round(rrmse_co2*100, 3))+'%', xy=(0.975, 0.025), xycoords="axes fraction", ha='right', fontsize=14)
ax[1].legend(loc=(0.02, 0.75), labelspacing=0.2, handletextpad=0.5, handlelength=0.75, prop={'size': 14}, frameon=False)
ax[1].set_xlabel('$\mathregular{CO_{2}}$, NRA (wt.%)')
ax[1].set_ylabel('44.01$\mathregular{\cdot A_{CO_3^{2-}} \cdot}$Density$\mathregular{\cdot}$Thickness')
ax[1].set_xlim([0, 2])
ax[1].set_ylim([0, 600])
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('Epsilon_PeakHeight.pdf')

# %% 

h2o_3550_area = (18.01528*df_h2o['Area_3550_M']) / (df_h2o['Density_Sat']*(df_h2o['Thickness']/1e6))
sigma_h2o_3550_area = h2o_3550_area * np.mean(np.sqrt((2*df_h2o.Area_3550_STD/df_h2o.Area_3550_M)**2) + (0.025**2) + (df_h2o['Sigma_Thickness']/df_h2o['Thickness'])**2)
mest_h2o_area, covm_est_h2o_area, covy_h2o_area = pig.inversion(h2o_erda, h2o_3550_area, h2o_erda_std, sigma_h2o_3550_area, intercept_zero=True)
mls_h2o_area, covls_h2o_area = pig.least_squares(h2o_erda, h2o_3550_area, sigma_h2o_3550_area)
E_calib_h2o_area, see_h2o_area, r2_h2o_area, rmse_h2o_area, rrmse_h2o_area, ccc_h2o_area = pig.inversion_fit_errors(h2o_erda, h2o_3550_area, mest_h2o_area, covy_h2o_area)
line_x_h2o_area, line_y_h2o_area, conf_low_h2o_area, conf_up_h2o_area, pred_low_h2o_area, pred_up_h2o_area = pig.inversion_fit_errors_plotting(h2o_erda, h2o_3550_area, mest_h2o_area)
print('mest_{H2O area} ' + str(mest_h2o_area[1]))
print('95% CI final ' + str(2*np.sqrt(np.diag(covm_est_h2o_area))[1]))

co2_y_1430_area = 44.01*((df['PH_1430_BP'])*(np.sqrt(2*np.pi*(df['STD_1430_BP'])**2))) / (df['Density_Sat']*(df['Thickness']/1e6))
co2_y_1515_area = 44.01*((df['PH_1515_BP'])*(np.sqrt(2*np.pi*(df['STD_1515_BP'])**2))) / (df['Density_Sat']*(df['Thickness']/1e6))
sigma_co2_y_1430_area = co2_y_1430_area * np.mean(np.sqrt((df['PH_1430_STD']/df['PH_1430_BP'])**2 + (0.025**2) + (df['Sigma_Thickness']/df['Thickness'])**2))
sigma_co2_y_1515_area = co2_y_1515_area * np.mean(np.sqrt((df['PH_1515_STD']/df['PH_1515_BP'])**2 + (0.025**2) + (df['Sigma_Thickness']/df['Thickness'])**2))
co2_y_area = pd.concat([co2_y_1430_area, co2_y_1515_area])
sigma_co2_y_area = pd.concat([sigma_co2_y_1430_area, sigma_co2_y_1515_area])
mest_co2_area, covm_est_co2_area, covy_co2_area = pig.inversion(co2_nra_x, co2_y_area, co2_nra_std_x, sigma_co2_y_area, intercept_zero=True)
mls_co2_area, covls_co2_area = pig.least_squares(co2_nra_x, co2_y_area, sigma_co2_y_area)
E_calib_co2_area, see_co2_area, r2_co2_area, rmse_co2_area, rrmse_co2_area, ccc_co2_area = pig.inversion_fit_errors(co2_nra_x, co2_y_area, mest_co2_area, covy_co2_area)
line_x_co2_area, line_y_co2_area, conf_low_co2_area, conf_up_co2_area, pred_low_co2_area, pred_up_co2_area = pig.inversion_fit_errors_plotting(co2_nra_x, co2_y_area, mest_co2_area)
print('mest_{carbonate area} ' + str(mest_co2_area[1]))
print('95% CI final ' + str(2*np.sqrt(np.diag(covm_est_co2_area))[1]))

# %% 

sz = 125
fig, ax = plt.subplots(1, 2, figsize = (14, 7))
ax = ax.flatten()
h2o_inv_area = pig.calculate_y_inversion(mest_h2o_area, h2o_range)
h2o_label = '$\mathregular{Ɛ_{i, H_{2}O_{3550}}}$=' + f'{round(mest_h2o_area[1])}(±{round(np.sqrt(np.diag(covm_est_h2o_area))[1])})' + ' L/mol$\mathregular{\cdot cm^{2}}$'
ax[0].plot(h2o_range, h2o_inv_area, 'k', lw=1, label=h2o_label)
ax[0].scatter(h2o_erda, h2o_3550_area, s = sz, c = '#0C7BDC', ec = '#171008', lw= 0.5, zorder = 20, label='ND70-Series, n=6')
ax[0].errorbar(h2o_erda, h2o_3550_area, xerr=h2o_erda_std, yerr=sigma_h2o_3550_area, fmt='none', lw= 0.5, c = 'k', zorder = 10)

ax[0].fill_between(line_x_h2o_area, conf_low_h2o_area, conf_up_h2o_area, color = 'k', alpha=0.20, edgecolor=None,
                   zorder=-5, label='68% Confidence Interval')
ax[0].plot(line_x_h2o_area, pred_up_h2o_area, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[0].plot(line_x_h2o_area, pred_low_h2o_area, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[0].annotate("A.", xy=(0.03, 0.945), xycoords="axes fraction", ha='left', fontsize=20)
ax[0].annotate("$\mathregular{R^{2}}$="+str(np.round(r2_h2o_area, 3)), xy=(0.975, 0.125), xycoords="axes fraction", ha='right', fontsize=14)
ax[0].annotate("CCC="+str(np.round(ccc_h2o_area, 3)), xy=(0.975, 0.075), xycoords="axes fraction", ha='right', fontsize=14)
ax[0].annotate("RRMSE="+str(np.round(rrmse_h2o_area*100, 3))+'%', xy=(0.975, 0.025), xycoords="axes fraction", ha='right', fontsize=14)
ax[0].legend(loc=(0.02, 0.75), labelspacing=0.2, handletextpad=0.5, handlelength=0.75, prop={'size': 14}, frameon=False)
ax[0].set_xlabel('$\mathregular{H_{2}O}$, ERDA/SIMS (wt.%)')
ax[0].set_ylabel('18.02$\mathregular{\cdot A_{i, H_2O_{t, 3550}} \cdot}$Density$\mathregular{\cdot}$Thickness')
ax[0].set_xlim([0, 7])
ax[0].set_ylim([0, 200000])
ax[0].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad = 6.5)

co2_inv_area = pig.calculate_y_inversion(mest_co2_area, co2_range)
co2_1430_label = '$\mathregular{Ɛ_{i, CO_{3, 1515/1430}^{2-}}}$=' + f'{round(mest_co2_area[1])}(±{round(np.sqrt(np.diag(covm_est_co2_area))[1])})' + ' L/mol$\mathregular{\cdot cm^{2}}$'
ax[1].plot(co2_range, co2_inv_area, 'k', lw=1, label=co2_1430_label)
ax[1].scatter(co2_nra_x, co2_y_area, s = sz, c='#E42211', marker = 's', ec = '#171008', lw= 0.5, zorder = 20, label='ND70-Series, n=14')
ax[1].errorbar(co2_nra_x, co2_y_area, xerr=co2_nra_std_x, yerr=sigma_co2_y_area, fmt='none', lw= 0.5, c = 'k', zorder = 10)
ax[1].fill_between(line_x_co2_area, conf_low_co2_area, conf_up_co2_area, color = 'k', alpha=0.20, edgecolor=None,
                   zorder=-5, label='68% Confidence Interval')
ax[1].plot(line_x_co2_area, pred_up_co2_area, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[1].plot(line_x_co2_area, pred_low_co2_area, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[1].annotate("B.", xy=(0.03, 0.945), xycoords="axes fraction", ha='left', fontsize=20)
ax[1].annotate("$\mathregular{R^{2}}$="+str(np.round(r2_co2_area, 3)), xy=(0.975, 0.125), xycoords="axes fraction", ha='right', fontsize=14)
ax[1].annotate("CCC="+str(np.round(ccc_co2_area, 3)), xy=(0.975, 0.075), xycoords="axes fraction", ha='right', fontsize=14)
ax[1].annotate("RRMSE="+str(np.round(rrmse_co2_area*100, 3))+'%', xy=(0.975, 0.025), xycoords="axes fraction", ha='right', fontsize=14)
ax[1].legend(loc=(0.02, 0.75), labelspacing=0.2, handletextpad=0.5, handlelength=0.75, prop={'size': 14}, frameon=False)
ax[1].set_xlabel('$\mathregular{CO_{2}}$, NRA (wt.%)')
ax[1].set_ylabel('44.01$\mathregular{\cdot A_{i, CO_3^{2-}} \cdot}$Density$\mathregular{\cdot}$Thickness')
ax[1].set_xlim([0, 2])
ax[1].set_ylim([0, 50000])
ax[1].tick_params(axis="x", direction='in', length=5, pad = 6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad = 6.5)
plt.tight_layout()
# plt.savefig('Epsilon_PeakArea.pdf')

# %%
