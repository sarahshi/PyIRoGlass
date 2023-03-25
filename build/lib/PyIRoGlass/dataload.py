# %% 
__author__ = 'Sarah Shi, Henry Towbin'

import os
import copy
import numpy as np
import pandas as pd
import mc3
import warnings

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
from matplotlib import pyplot as plt
mpl.use('pgf')
import mc3.utils as mu
import mc3.stats as ms

from pykrige import OrdinaryKriging
import scipy.signal as signal
from scipy.linalg import solveh_banded
import scipy.interpolate as interpolate

from ._version import __version__

# %%

def Load_SampleCSV(paths, wn_high, wn_low): 

    """
    The Load_SampleCSV function takes the inputs of the path to a directory with all sample CSVs, 
    wavenumber high, and wavenumber low values. 

    Parameters:
        paths (list): A list of file paths to CSV files containing spectral data.
        wn_high (float): The highest wavenumber to include in the output dictionary.
        wn_low (float): The lowest wavenumber to include in the output dictionary.

    Returns:
        Tuple containing the following elements:
            files (list): List of names for each sample in the directory.
            dfs_dict (dictionary): Dictionary where each key is a file name, and the 
                corresponding value is a pandas dataframe containing
                wavenumbers and absorbances for the given sample.
    """

    dfs = []
    files = []

    for path in paths:
        head_tail = os.path.split(path)
        file = head_tail[1][0:-4]

        df = pd.read_csv(path, names= ['Wavenumber', 'Absorbance'])
        df.set_index('Wavenumber', inplace = True)
        spectrum = df.loc[wn_low:wn_high]
        dfs.append(spectrum)
        files.append(file)

    zipobj = zip(files, dfs)
    dfs_dict = dict(zipobj)

    return files, dfs_dict

def Load_PCA(file_name):

    """
    Loads predetermined PCA components from an NPZ file.

    Parameters:
        file_name (str): The file name of an NPZ file containing PCA components.
    
    Returns:
        PCA_matrix (matrix): Matrix containing the PCA components.
    """

    wn_high = 2200
    wn_low = 1275

    file_path = os.path.join(os.path.dirname(__file__), file_name)
    npz = np.load(file_path)
    PCA_DF = pd.DataFrame(npz['data'], columns = npz['columns'])
    PCA_DF = PCA_DF.set_index('Wavenumber')

    PCA_DF = PCA_DF[wn_low:wn_high]
    PCA_matrix = np.matrix(PCA_DF.to_numpy())

    return PCA_matrix

def Load_Wavenumber(file_name):

    """
    Loads predetermined wavenumbers from an NPZ file.

    Parameters:
        file_name (str): Path to a CSV file containing wavenumbers.

    Returns:
        Wavenumber (np.ndarray): An array of wavenumbers.
    """

    wn_high = 2200
    wn_low = 1275

    file_path = os.path.join(os.path.dirname(__file__), file_name)
    npz = np.load(file_path)

    Wavenumber_DF = pd.DataFrame(npz['data'], columns = npz['columns'])
    Wavenumber_DF = Wavenumber_DF.set_index('Wavenumber')

    Wavenumber_DF = Wavenumber_DF[wn_low:wn_high]
    Wavenumber = np.array(Wavenumber_DF.index)

    return Wavenumber

def Load_ChemistryThickness(ChemistryThickness_Path):

    """
    Loads dataframes with glass chemistry and thickness data.

    Parameters:
        ChemistryThickness_Path (str): The path to a CSV file containing the chemistry and thickness data.
    
    Returns:
        Tuple containing the following elements:
            Chemistry (pd.DataFrame): The dataframe containing the glass chemistry data. 
            Thickness (pd.DataFrame): The dataframe containing the thickness and standard deviation data. 
    """

    ChemistryThickness = pd.read_csv(ChemistryThickness_Path)
    ChemistryThickness.set_index('Sample', inplace = True)

    Chemistry = ChemistryThickness.loc[:, ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']]
    Thickness = ChemistryThickness.loc[:, ['Thickness', 'Sigma_Thickness']]


    return Chemistry, Thickness
