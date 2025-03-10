# %% 
""" Created on May 25, 2021 // @author: Sarah Shi and Bill Menke """

import os
import sys
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import rc

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

# method treats d, z, m1 and m2 as unknowns. model f(i) = 0 = m1 + m2*z(i) - d(i);
# unknowns:  m1, m2 and the predicted (z,d)'s

# %% 

df_5200 = pd.read_excel('./EpsilonRegression.xlsx', sheet_name='NIRRegress')
tau_5200 = df_5200['Tau']
sigma_tau_5200 = tau_5200 * 0.025
epsilon_5200 = df_5200['Epsilon_5200']
sigma_epsilon_5200 = epsilon_5200 * 0.10

df_4500 = pd.read_excel('./EpsilonRegression.xlsx', sheet_name='NIRRegress')
tau_4500 = df_4500['Tau']
sigma_tau_4500 = tau_4500 * 0.025
epsilon_4500 = df_4500['Epsilon_4500']
sigma_epsilon_4500 = epsilon_4500 * 0.20

df_3550 = pd.read_excel('./EpsilonRegression.xlsx', sheet_name='3550Regress')
tau_3550 = df_3550['Tau']
sigma_tau_3550 = tau_3550 * 0.025
epsilon_3550 = df_3550['Epsilon_3550']
sigma_epsilon_3550 = epsilon_3550 * 0.10

df_1635 = pd.read_excel('./EpsilonRegression.xlsx', sheet_name='1635Regress')
tau_1635 = df_1635['Tau']
sigma_tau_1635 = tau_1635 * 0.025
epsilon_1635 = df_1635['Epsilon_1635']
sigma_epsilon_1635 = epsilon_1635 * 0.05

df_carbonate = pd.read_excel('./EpsilonRegression.xlsx', sheet_name='CarbonateRegress')
eta = df_carbonate['Eta']
sigma_eta = eta * 0.025
epsilon_carbonate = df_carbonate['Epsilon_Carbonate']
sigma_epsilon_carbonate = epsilon_carbonate * 0.10

# %%

mest_5200, covm_est_5200, covepsilon_5200 = pig.inversion(tau_5200, epsilon_5200, sigma_tau_5200, sigma_epsilon_5200)
mls_5200, covls_5200 = pig.least_squares(tau_5200, epsilon_5200, sigma_epsilon_5200)
E_calib_5200, SEE_5200, R2_5200, RMSE_5200, RRMSE_5200, CCC_5200 = pig.inversion_fit_errors(tau_5200, epsilon_5200, mest_5200, covepsilon_5200)
tau_arr_5200, epsilon_5200_arr, conf_lower_5200, conf_upper_5200, pred_lower_5200, pred_upper_5200 = pig.inversion_fit_errors_plotting(tau_5200, epsilon_5200, mest_5200)

mest_4500, covm_est_4500, covepsilon_4500 = pig.inversion(tau_4500, epsilon_4500, sigma_tau_4500, sigma_epsilon_4500)
mls_4500, covls_4500 = pig.least_squares(tau_4500, epsilon_4500, sigma_epsilon_4500)
E_calib_4500, SEE_4500, R2_4500, RMSE_4500, RRMSE_4500, CCC_4500 = pig.inversion_fit_errors(tau_4500, epsilon_4500, mest_4500, covepsilon_4500)
tau_arr_4500, epsilon_4500_arr, conf_lower_4500, conf_upper_4500, pred_lower_4500, pred_upper_4500 = pig.inversion_fit_errors_plotting(tau_4500, epsilon_4500, mest_4500)

mest_3550, covm_est_3550, covepsilon_3550 = pig.inversion(tau_3550, epsilon_3550, sigma_tau_3550, sigma_epsilon_3550)
mls_3550, covls_3550 = pig.least_squares(tau_3550, epsilon_3550, sigma_epsilon_3550)
E_calib_3550, SEE_3550, R2_3550, RMSE_3550, RRMSE_3550, CCC_3550 = pig.inversion_fit_errors(tau_3550, epsilon_3550, mest_3550, covepsilon_3550)
tau_arr_3550, epsilon_3550_arr, conf_lower_3550, conf_upper_3550, pred_lower_3550, pred_upper_3550 = pig.inversion_fit_errors_plotting(tau_3550, epsilon_3550, mest_3550)

mest_1635, covm_est_1635, covepsilon_1635 = pig.inversion(tau_1635, epsilon_1635, sigma_tau_1635, sigma_epsilon_1635)
mls_1635, covls_1635 = pig.least_squares(tau_1635, epsilon_1635, sigma_epsilon_1635)
E_calib_1635, SEE_1635, R2_1635, RMSE_1635, RRMSE_1635, CCC_1635 = pig.inversion_fit_errors(tau_1635, epsilon_1635, mest_1635, covepsilon_1635)
tau_arr_1635, epsilon_1635_arr, conf_lower_1635, conf_upper_1635, pred_lower_1635, pred_upper_1635 = pig.inversion_fit_errors_plotting(tau_1635, epsilon_1635, mest_1635)

mest_carbonate, covm_est_carbonate, covepsilon_carbonate = pig.inversion(eta, epsilon_carbonate, sigma_eta, sigma_epsilon_carbonate)
mls_carbonate, covls_carbonate = pig.least_squares(eta, epsilon_carbonate, sigma_epsilon_carbonate)
E_calib_carbonate, SEE_carbonate, R2_carbonate, RMSE_carbonate, RRMSE_carbonate, CCC_carbonate = pig.inversion_fit_errors(eta, epsilon_carbonate, mest_carbonate, covepsilon_carbonate)
eta_arr, epsilon_carbonate_arr, conf_lower_carbonate, conf_upper_carbonate, pred_lower_carbonate, pred_upper_carbonate = pig.inversion_fit_errors_plotting(eta, epsilon_carbonate, mest_carbonate)

# %% 5200

epsilon_5200_mandeville = -2.463 + 4.899*tau_arr_5200
fuego_idx = np.where((tau_arr_5200 > 0.653) & (tau_arr_5200 < 0.715))
legend_5200 = '$\mathregular{ƐH_2O_{m, 5200}}$ = ' + f'{round(mest_5200[0],3)}(±{round(np.sqrt(np.diag(covm_est_5200))[0],3)}) + {round(mest_5200[1],3)}(±{round(np.sqrt(np.diag(covm_est_5200))[1],3)})'+ '·' + '$\\tau$'+ f', n={len(tau_5200)}'

sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot(tau_arr_5200, epsilon_5200_arr, 'k', lw=2, zorder=0, label=legend_5200)
mand, = ax.plot(tau_arr_5200, epsilon_5200_mandeville, 'k-.', lw=2, zorder=0, label='Mandeville et al., 2002')
mand.set_dashes([1.5, 1, 3, 1])
ax.fill_between(tau_arr_5200, conf_lower_5200, conf_upper_5200, color='k', alpha=0.20, edgecolor=None,
    zorder=-5, label='68% Confidence Interval')
ax.plot(tau_arr_5200, pred_upper_5200, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax.plot(tau_arr_5200, pred_lower_5200, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax.errorbar(tau_5200, epsilon_5200, yerr=sigma_epsilon_5200, xerr=sigma_tau_5200, ls='none', elinewidth=0.5, ecolor='k')
ax.scatter(tau_5200, epsilon_5200, s=sz, c='#0C7BDC', edgecolors='black', linewidth=0.5, zorder=15)
ax.set_xlim([ 0.5, 1.0])
ax.set_ylim([-0.5, 3.5])
xlabel_5200 = '$\\tau$='+'(Si+Al)/Total Cations'

ax.set_xlabel(xlabel_5200)
ax.set_ylabel('$\mathregular{ƐH_2O_{m, 5200}}$')

ax.legend(loc='upper left', labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax.tick_params(axis="x", direction='in', length=5, pad=6.5)
ax.tick_params(axis="y", direction='in', length=5, pad=6.5)
plt.tight_layout()
# plt.savefig('Epsilon5200Regress.pdf')

# %% 
# %% 4500

epsilon_4500_mandeville = -2.026+4.054*tau_arr_4500
fuego_idx = np.where((tau_arr_4500 > 0.653) & (tau_arr_4500 < 0.715))
legend_4500 = '$\mathregular{ƐOH^{-}_{4500}}$ = ' + f'{round(mest_4500[0],3)}(±{round(np.sqrt(np.diag(covm_est_4500))[0],3)}) + {round(mest_4500[1],3)}(±{round(np.sqrt(np.diag(covm_est_4500))[1],3)})'+ '·' + '$\\tau$'+ f', n={len(tau_4500)}'

sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot(tau_arr_4500, epsilon_4500_arr, 'k', lw=2, zorder=0, label=legend_4500)
mand, = ax.plot(tau_arr_4500, epsilon_4500_mandeville, 'k-.', lw=2, zorder=0, label='Mandeville et al., 2002')
mand.set_dashes([1.5, 1, 3, 1])
ax.fill_between(tau_arr_4500, conf_lower_4500, conf_upper_4500, color='k', alpha=0.20, edgecolor=None,
    zorder=-5, label='68% Confidence Interval')
ax.plot(tau_arr_4500, pred_upper_4500, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax.plot(tau_arr_4500, pred_lower_4500, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
# ax.fill_between(tau_arr_4500[fuego_idx], conf_lower_4500[fuego_idx], conf_upper_4500[fuego_idx], color='r', alpha=0.30, edgecolor=None, zorder=-5, label='Fuego Interval')
ax.errorbar(tau_4500, epsilon_4500, yerr=sigma_epsilon_4500, xerr=sigma_tau_4500, ls='none', elinewidth=0.5, ecolor='k')
ax.scatter(tau_4500, epsilon_4500, s=sz, c='#0C7BDC', edgecolors='black', linewidth=0.5, zorder=15)
ax.set_xlim([ 0.5, 1.0])
ax.set_ylim([-0.5, 3.5])
xlabel_4500 = '$\\tau$=' + '(Si+Al)/Total Cations'
ax.set_xlabel(xlabel_4500) 
ax.set_ylabel('$\mathregular{ƐOH^{-}_{4500}}$')
ax.legend(loc='upper left', labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax.tick_params(axis="x", direction='in', length=5, pad=6.5)
ax.tick_params(axis="y", direction='in', length=5, pad=6.5)
plt.tight_layout()
# plt.savefig('Epsilon4500Regress.pdf')

# %%
# %% 3550

fuego_idx = np.where((tau_arr_3550 > 0.653) & (tau_arr_3550 < 0.715))
legend_3550 = '$\mathregular{ƐH_2O_{t, 3550}}$ = ' + f'{round(mest_3550[0],3)}(±{round(np.sqrt(np.diag(covm_est_3550))[0],3)}) + {round(mest_3550[1],3)}(±{round(np.sqrt(np.diag(covm_est_3550))[1],3)})'+ '·' + '$\\tau$'+ f', n={len(tau_3550)}'

sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot(tau_arr_3550, epsilon_3550_arr, 'k', lw=2, zorder=0, label=legend_3550)
mand.set_dashes([1.5, 1, 3, 1])
ax.fill_between(tau_arr_3550, conf_lower_3550, conf_upper_3550, color='k', alpha=0.20, edgecolor=None, zorder=-5, label='68% Confidence Interval')
ax.plot(tau_arr_3550, pred_upper_3550, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax.plot(tau_arr_3550, pred_lower_3550, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax.errorbar(tau_3550, epsilon_3550, yerr=sigma_epsilon_3550, xerr=sigma_tau_3550, ls='none', elinewidth=0.5, ecolor='k')
ax.scatter(tau_3550, epsilon_3550, s=sz, c='#0C7BDC', edgecolors='black', linewidth=0.5, zorder=15)
ax.set_xlim([0.4, 1.0])
ax.set_ylim([20, 110])
xlabel_3550 = '$\\tau$=' + '(Si+Al)/Total Cations'
ax.set_xlabel(xlabel_3550) 
ax.set_ylabel('$\mathregular{ƐH_2O_{t, 3550}}$')
ax.legend(loc='upper left', labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax.tick_params(axis="x", direction='in', length=5, pad=6.5)
ax.tick_params(axis="y", direction='in', length=5, pad=6.5)
plt.tight_layout()
# plt.savefig('Epsilon3550Regress.pdf')

# %% 1635

epsilon_1635_mandeville = -57.813+131.94*tau_arr_1635
fuego_idx = np.where((tau_arr_1635 > 0.653) & (tau_arr_1635 < 0.715))
legend_1635 = '$\mathregular{ƐH_2O_{m, 1635}}$ = ' + f'{round(mest_1635[0],3)}(±{round(np.sqrt(np.diag(covm_est_1635))[0],3)}) + {round(mest_1635[1],3)}(±{round(np.sqrt(np.diag(covm_est_1635))[1],3)})'+ '·' + '$\\tau$'+ f', n={len(tau_1635)}'

sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))
ax.plot(tau_arr_1635, epsilon_1635_arr, 'k', lw=2, zorder=0, label=legend_1635)
mand, = ax.plot(tau_arr_1635, epsilon_1635_mandeville, 'k-.', lw=2, zorder=0, label='Mandeville et al., 2002')
mand.set_dashes([1.5, 1, 3, 1])
ax.fill_between(tau_arr_1635, conf_lower_1635, conf_upper_1635, color='k', alpha=0.20, edgecolor=None, zorder=-5, label='68% Confidence Interval')
ax.plot(tau_arr_1635, pred_upper_1635, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax.plot(tau_arr_1635, pred_lower_1635, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax.errorbar(tau_1635, epsilon_1635, yerr=sigma_epsilon_1635, xerr=sigma_tau_1635, ls='none', elinewidth=0.5, ecolor='k')
ax.scatter(tau_1635, epsilon_1635, s=sz, c='#0C7BDC', edgecolors='black', linewidth=0.5, zorder=15)
ax.set_xlim([0.5, 1.0])
ax.set_ylim([0, 90])
xlabel_1635 = '$\\tau$=' + '(Si+Al)/Total Cations'
ax.set_xlabel(xlabel_1635) 
ax.set_ylabel('$\mathregular{ƐH_2O_{m, 1635}}$')

ax.legend(loc='upper left', labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax.tick_params(axis="x", direction='in', length=5, pad=6.5)
ax.tick_params(axis="y", direction='in', length=5, pad=6.5)
plt.tight_layout()
# plt.savefig('Epsilon1635Regress.pdf')

# %% Carbonate

epsilon_carbonate_dixonpan = 451-342*eta_arr
# epsilon_carbonate_old = 440.696-355.205*eta_arr
fuego_idx = np.where((eta_arr > 0.389) & (eta_arr < 0.554))
df_carbonate = pd.read_excel('./EpsilonRegression.xlsx', sheet_name='CarbonateRegress')
low_df = df_carbonate[df_carbonate.Epsilon_Location == 'Low']
high_df = df_carbonate[df_carbonate.Epsilon_Location == 'High']
brounce = low_df[low_df.Compilation == 'Brounce']

sz = 150
fig, ax = plt.subplots(1, 1, figsize = (8, 8))

ax.errorbar(low_df['Eta'], low_df['Epsilon_Carbonate'], yerr=low_df['Epsilon_Carbonate']*0.1, xerr=low_df['Eta']*0.025, ls='none', elinewidth=0.5, ecolor='k')
ax.scatter(low_df['Eta'], low_df['Epsilon_Carbonate'], s=sz, c='#0C7BDC', edgecolors='black', linewidth=0.5, zorder=15, label='$\mathregular{CO_{3, 1430}^{2-}}$, n='+str(len(low_df)))
# ax.scatter(brounce['Eta'], brounce['Epsilon_Carbonate'], s=sz, c='#0C7BDC', edgecolors='black', linewidth=2, zorder=15, label='Brounce et al., 2021')
ax.errorbar(high_df['Eta'], high_df['Epsilon_Carbonate'], yerr=high_df['Epsilon_Carbonate']*0.10, xerr=high_df['Eta']*0.025, ls='none', elinewidth=0.5, ecolor='k')
ax.scatter(high_df['Eta'], high_df['Epsilon_Carbonate'], s=sz, c='#E42211', marker = 's', edgecolors='black', linewidth=0.5, zorder=15, label='$\mathregular{CO_{3, 1515}^{2-}}$, n='+str(len(high_df)))

dixonpan, = ax.plot(eta_arr, epsilon_carbonate_dixonpan, 'k-.', lw=1.5, zorder=0, label='Dixon and Pan, 1995')
dixonpan.set_dashes([1.5, 1, 3, 1])
legend_carbonate = '$\mathregular{ƐCO_3^{2-}}$= ' + f'{round(mest_carbonate[0],3)}(±{round(np.sqrt(np.diag(covm_est_carbonate))[0],3)}) - {round(mest_carbonate[1],3)*-1}(±{round(np.sqrt(np.diag(covm_est_carbonate))[1],3)})' + '·' + f'$\\eta$, n={len(eta)}'
ax.plot(eta_arr, epsilon_carbonate_arr, 'k', lw=2, zorder=0, label=legend_carbonate)
# ax.plot(eta_arr, epsilon_carbonate_old, 'green', lw=2, zorder=0)

ax.fill_between(eta_arr, conf_lower_carbonate, conf_upper_carbonate, color='k', alpha=0.20, edgecolor=None, zorder=-5, label='68% Confidence Interval')
ax.plot(eta_arr, pred_upper_carbonate, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax.plot(eta_arr, pred_lower_carbonate, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax.set_xlim([0.1, 0.9])
ax.set_ylim([0, 500])
ax.set_xlabel('$\mathregular{\\eta=Na/(Na+Ca)}$') 
ax.set_ylabel('$\mathregular{ƐCO_3^{2-}}$')

ax.legend(loc='lower left', labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax.tick_params(axis="x", direction='in', length=5, pad=6.5)
ax.tick_params(axis="y", direction='in', length=5, pad=6.5)
plt.tight_layout()
# plt.savefig('EpsilonCarbonateRegress.pdf')

# %% 
# %%

sz = 150
fig, ax = plt.subplots(3, 2, figsize = (14, 17)) 
ax = ax.flatten()

epsilon_5200_mandeville = -2.463 + 4.899*tau_arr_5200
fuego_idx = np.where((tau_arr_5200 > 0.653) & (tau_arr_5200 < 0.715))
legend_5200 = '$\mathregular{ƐH_2O_{m, 5200}}$=' + f'{round(mest_5200[0],3)}(±{round(np.sqrt(np.diag(covm_est_5200))[0],3)})+{round(mest_5200[1],3)}(±{round(np.sqrt(np.diag(covm_est_5200))[1],3)})'+ '·' + '$\\tau$'+ f', n={len(tau_5200)}'
ax[0].plot(tau_arr_5200, epsilon_5200_arr, 'k', lw=2, zorder=0, label=legend_5200)
mand, = ax[0].plot(tau_arr_5200, epsilon_5200_mandeville, 'k-.', lw=2, zorder=0, label='Mandeville et al., 2002')
mand.set_dashes([1.5, 1, 3, 1])
ax[0].fill_between(tau_arr_5200, conf_lower_5200, conf_upper_5200, color='k', alpha=0.20, edgecolor=None, zorder=-5, label='68% Confidence Interval')
ax[0].plot(tau_arr_5200, pred_upper_5200, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[0].plot(tau_arr_5200, pred_lower_5200, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[0].errorbar(tau_5200, epsilon_5200, yerr=sigma_epsilon_5200, xerr=sigma_tau_5200, ls='none', elinewidth=0.5, ecolor='k')
ax[0].scatter(tau_5200, epsilon_5200, s=sz, c='#0C7BDC', edgecolors='black', linewidth=0.5, zorder=15)
ax[0].set_xlim([ 0.5, 1.0])
ax[0].set_ylim([-0.5, 3.5])
ax[0].annotate("A.", xy=(0.032, 0.935), xycoords="axes fraction", fontsize=20, weight='bold')
xlabel_5200 = '$\\tau$='+'(Si+Al)/Total Cations'
ax[0].set_xlabel(xlabel_5200)
ax[0].set_ylabel('$\mathregular{ƐH_2O_{m, 5200}}$')
ax[0].legend(loc=(0.02, 0.69), labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax[0].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[0].tick_params(axis="y", direction='in', length=5, pad=6.5)

epsilon_4500_mandeville = -2.026+4.054*tau_arr_4500
fuego_idx = np.where((tau_arr_4500 > 0.653) & (tau_arr_4500 < 0.715))
legend_4500 = '$\mathregular{ƐOH^{-}_{4500}}$=' + f'{round(mest_4500[0],3)}(±{round(np.sqrt(np.diag(covm_est_4500))[0],3)})+{round(mest_4500[1],3)}(±{round(np.sqrt(np.diag(covm_est_4500))[1],3)})'+ '·' + '$\\tau$'+ f', n={len(tau_4500)}'
ax[1].plot(tau_arr_4500, epsilon_4500_arr, 'k', lw=2, zorder=0, label=legend_4500)
mand, = ax[1].plot(tau_arr_4500, epsilon_4500_mandeville, 'k-.', lw=2, zorder=0, label='Mandeville et al., 2002')
mand.set_dashes([1.5, 1, 3, 1])
ax[1].fill_between(tau_arr_4500, conf_lower_4500, conf_upper_4500, color='k', alpha=0.20, edgecolor=None,
    zorder=-5, label='68% Confidence Interval')
ax[1].plot(tau_arr_4500, pred_upper_4500, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[1].plot(tau_arr_4500, pred_lower_4500, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[1].errorbar(tau_4500, epsilon_4500, yerr=sigma_epsilon_4500, xerr=sigma_tau_4500, ls='none', elinewidth=0.5, ecolor='k')
ax[1].scatter(tau_4500, epsilon_4500, s=sz, c='#0C7BDC', edgecolors='black', linewidth=0.5, zorder=15)
ax[1].set_xlim([ 0.5, 1.0])
ax[1].set_ylim([-0.5, 3.5])
ax[1].annotate("B.", xy=(0.032, 0.935), xycoords="axes fraction", fontsize=20, weight='bold')
xlabel_4500 = '$\\tau$=' + '(Si+Al)/Total Cations'
ax[1].set_xlabel(xlabel_4500) 
ax[1].set_ylabel('$\mathregular{ƐOH^{-}_{4500}}$')
ax[1].legend(loc=(0.02, 0.69), labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax[1].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[1].tick_params(axis="y", direction='in', length=5, pad=6.5)


fuego_idx = np.where((tau_arr_3550 > 0.653) & (tau_arr_3550 < 0.715))
legend_3550 = '$\mathregular{ƐH_2O_{t, 3550}}$=' + f'{round(mest_3550[0],3)}(±{round(np.sqrt(np.diag(covm_est_3550))[0],3)})+{round(mest_3550[1],3)}(±{round(np.sqrt(np.diag(covm_est_3550))[1],3)})'+ '·' + '$\\tau$'+ f', n={len(tau_3550)}'
ax[2].plot(tau_arr_3550, epsilon_3550_arr, 'k', lw=2, zorder=0, label=legend_3550)
mand.set_dashes([1.5, 1, 3, 1])
ax[2].fill_between(tau_arr_3550, conf_lower_3550, conf_upper_3550, color='k', alpha=0.20, edgecolor=None,
    zorder=-5, label='68% Confidence Interval')
ax[2].plot(tau_arr_3550, pred_upper_3550, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[2].plot(tau_arr_3550, pred_lower_3550, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[2].errorbar(tau_3550, epsilon_3550, yerr=sigma_epsilon_3550, xerr=sigma_tau_3550, ls='none', elinewidth=0.5, ecolor='k')
ax[2].scatter(tau_3550, epsilon_3550, s=sz, c='#0C7BDC', edgecolors='black', linewidth=0.5, zorder=15)
ax[2].set_xlim([0.4, 1.0])
ax[2].set_ylim([20, 120])
ax[2].annotate("C.", xy=(0.032, 0.935), xycoords="axes fraction", fontsize=20, weight='bold')
xlabel_3550 = '$\\tau$=' + '(Si+Al)/Total Cations'
ax[2].set_xlabel(xlabel_3550) 
ax[2].set_ylabel('$\mathregular{ƐH_2O_{t, 3550}}$')
ax[2].legend(loc=(0.02, 0.74), labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax[2].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[2].tick_params(axis="y", direction='in', length=5, pad=6.5)
ax[2].scatter(tau_3550[epsilon_3550==63.027], epsilon_3550[epsilon_3550==63.027], s=sz, c='#0C7BDC', edgecolors='black', linewidth=2, zorder=15)


epsilon_1635_mandeville = -57.813+131.94*tau_arr_1635
fuego_idx = np.where((tau_arr_1635 > 0.653) & (tau_arr_1635 < 0.715))
legend_1635 = '$\mathregular{ƐH_2O_{m, 1635}}$=' + f'{round(mest_1635[0],3)}(±{round(np.sqrt(np.diag(covm_est_1635))[0],3)})+{round(mest_1635[1],3)}(±{round(np.sqrt(np.diag(covm_est_1635))[1],3)})'+ '·' + '$\\tau$'+ f', n={len(tau_1635)}'
ax[3].plot(tau_arr_1635, epsilon_1635_arr, 'k', lw=2, zorder=0, label=legend_1635)
mand, = ax[3].plot(tau_arr_1635, epsilon_1635_mandeville, 'k-.', lw=2, zorder=0, label='Mandeville et al., 2002')
mand.set_dashes([1.5, 1, 3, 1])
ax[3].fill_between(tau_arr_1635, conf_lower_1635, conf_upper_1635, color='k', alpha=0.20, edgecolor=None,
    zorder=-5, label='68% Confidence Interval')
ax[3].plot(tau_arr_1635, pred_upper_1635, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[3].plot(tau_arr_1635, pred_lower_1635, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[3].errorbar(tau_1635, epsilon_1635, yerr=sigma_epsilon_1635, xerr=sigma_tau_1635, ls='none', elinewidth=0.5, ecolor='k')
ax[3].scatter(tau_1635, epsilon_1635, s=sz, c='#0C7BDC', edgecolors='black', linewidth=0.5, zorder=15)
ax[3].set_xlim([0.5, 1.0])
ax[3].set_ylim([0, 90])
ax[3].annotate("D.", xy=(0.032, 0.935), xycoords="axes fraction", fontsize=20, weight='bold')
xlabel_1635 = '$\\tau$=' + '(Si+Al)/Total Cations'
ax[3].set_xlabel(xlabel_1635) 
ax[3].set_ylabel('$\mathregular{ƐH_2O_{m, 1635}}$')
ax[3].legend(loc=(0.02, 0.69), labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax[3].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[3].tick_params(axis="y", direction='in', length=5, pad=6.5)

epsilon_carbonate_dixonpan = 451-342*eta_arr
fuego_idx = np.where((eta_arr > 0.389) & (eta_arr < 0.554))
df_carbonate = pd.read_excel('./EpsilonRegression.xlsx', sheet_name='CarbonateRegress')
low_df = df_carbonate[df_carbonate.Epsilon_Location == 'Low']
shi_low = low_df[low_df.Compilation=='Shi']
high_df = df_carbonate[df_carbonate.Epsilon_Location == 'High']
shi_high = low_df[low_df.Compilation=='Shi']

ax[4].errorbar(low_df['Eta'], low_df['Epsilon_Carbonate'], yerr=low_df['Epsilon_Carbonate']*0.1, xerr=low_df['Eta']*0.025, ls='none', elinewidth=0.5, ecolor='k')
ax[4].scatter(low_df['Eta'], low_df['Epsilon_Carbonate'], s=sz, c='#0C7BDC', edgecolors='black', linewidth=0.5, zorder=15, label='$\mathregular{CO_{3, 1430}^{2-}}$, n='+str(len(low_df)))
ax[4].errorbar(high_df['Eta'], high_df['Epsilon_Carbonate'], yerr=high_df['Epsilon_Carbonate']*0.10, xerr=high_df['Eta']*0.025, ls='none', elinewidth=0.5, ecolor='k')
ax[4].scatter(high_df['Eta'], high_df['Epsilon_Carbonate'], s=sz, c='#E42211', marker = 's', edgecolors='black', linewidth=0.5, zorder=15, label='$\mathregular{CO_{3, 1515}^{2-}}$, n='+str(len(high_df)))

ax[4].scatter(shi_high['Eta'], shi_high['Epsilon_Carbonate'], s=sz, c='#E42211', marker = 's', edgecolors='black', linewidth=2, zorder=15)
ax[4].scatter(shi_low['Eta'], shi_low['Epsilon_Carbonate'], s=sz, c='#0C7BDC', edgecolors='black', linewidth=2, zorder=15)#

dixonpan, = ax[4].plot(eta_arr, epsilon_carbonate_dixonpan, 'k-.', lw=1.5, zorder=0, label='Dixon and Pan, 1995')
dixonpan.set_dashes([1.5, 1, 3, 1])
legend_carbonate = '$\mathregular{ƐCO_3^{2-}}$=' + f'{round(mest_carbonate[0],3)}(±{round(np.sqrt(np.diag(covm_est_carbonate))[0],3)})-{round(mest_carbonate[1],3)*-1}(±{round(np.sqrt(np.diag(covm_est_carbonate))[1],3)})' + '·' + f'$\\eta$'
ax[4].plot(eta_arr, epsilon_carbonate_arr, 'k', lw=2, zorder=0, label=legend_carbonate)
ax[4].fill_between(eta_arr, conf_lower_carbonate, conf_upper_carbonate, color='k', alpha=0.20, edgecolor=None,
    zorder=-5, label='68% Confidence Interval')
ax[4].plot(eta_arr, pred_upper_carbonate, 'k--', lw=0.5, zorder=0, dashes=(16, 10))
ax[4].plot(eta_arr, pred_lower_carbonate, 'k--', lw=0.5, zorder=0, dashes=(16, 10), label='68% Prediction Interval')
ax[4].set_xlim([0.1, 0.9])
ax[4].set_ylim([0, 500])
ax[4].annotate("E.", xy=(0.032, 0.935), xycoords="axes fraction", fontsize=20, weight='bold')
ax[4].set_xlabel('$\mathregular{\\eta=Na/(Na+Ca)}$') 
ax[4].set_ylabel('$\mathregular{ƐCO_3^{2-}}$')
ax[4].legend(loc='lower left', labelspacing=0.2, handletextpad=0.5, handlelength=1.0, prop={'size': 14}, frameon=False)
ax[4].tick_params(axis="x", direction='in', length=5, pad=6.5)
ax[4].tick_params(axis="y", direction='in', length=5, pad=6.5)

fig.delaxes(ax[5])
plt.tight_layout()
plt.savefig('AllEpsilonRegress_SHI1.pdf', bbox_inches='tight', pad_inches=0.025)

# %%
