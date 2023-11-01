# %% 
""" Created on October 29, 2023 // @author: Sarah Shi """

import numpy as np
import pandas as pd

# %%

chem_col = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'P2O5', 'NiO', 'Cr2O3']
sc = pd.read_csv('SanCarlos.csv', index_col=0)
sc_chem = sc[chem_col]

mask = (sc_chem >= (sc_chem.mean() - 2*sc_chem.std())) & (sc_chem <= (sc_chem.mean() + 2*sc_chem.std()))
sc_chem_filt = sc_chem[mask.all(axis=1)]

sc_mean = sc_chem_filt.mean()
sc_std = sc_chem_filt.std()
sc_precision = sc_std/sc_mean*100
sc_precision

# %%

san_carlos_data = {
    'SiO2': 40.81,
    'TiO2': 0.00239,
    'Al2O3': 0.01910,
    'FeO': 9.55000,
    'MnO': 0.14000,
    'MgO': 49.42000,
    'CaO': 0.10000,
    'P2O5': 0.00223,
    'NiO': 0.29000,
    'Cr2O3': 0.009878356
}

san_carlos = pd.DataFrame([san_carlos_data])

sc_mean/san_carlos*100

# %% 

((sc_mean - san_carlos) / san_carlos) * 100

# %%

nd70_chem_col = ['SiO2', 'TiO2', 'Al2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5', 'F', 'Cl', 'SO2']
nd70 = pd.read_csv('ND70.csv', index_col=0)
nd70_chem = nd70[nd70_chem_col]

nd70_mask = (nd70_chem >= (nd70_chem.mean() - 2*nd70_chem.std())) & (nd70_chem <= (nd70_chem.mean() + 2*nd70_chem.std()))
nd70_chem_filt = nd70_chem[nd70_mask.all(axis=1)]

nd70_chem_filt.drop("70_1", inplace=True)
nd70_chem_filt.drop("2_1", inplace=True)

nd70_mean = nd70_chem_filt.mean()
nd70_std = nd70_chem_filt.std()
nd70_precision = nd70_std/nd70_mean*100
nd70_precision

# %%

nd70_data = {
    'SiO2': 49.77,
    'TiO2': 0.83,
    'Al2O3': 16.39,
    'FeO': 8.16,
    'MnO': 0.15,
    'MgO': 8.27,
    'CaO': 12.78,
    'Na2O': 2.05,
    'K2O': 0.16,
    'P2O5': 0.09,
    'F': 0.0155, 
    'Cl': 254.04/10000,
    'SO2': 634.94/5000, 
}

nd70_exp = pd.DataFrame([nd70_data])

100 - (nd70_mean/nd70_exp*100)

# %%

((nd70_mean - nd70_exp) / nd70_exp) * 100

# %% 