# %%

__author__ = 'Sarah Shi'

import os
import warnings
import numpy as np
import pandas as pd
import mc3

import matplotlib as mpl
from matplotlib import pyplot as plt
import mc3.plots as mp
import mc3.stats as ms
import mc3.utils as mu

from pykrige import OrdinaryKriging
import scipy.signal as signal
from scipy.linalg import solveh_banded
import scipy.interpolate as interpolate


# %% Core functions

def Load_SampleCSV(paths, wn_high=5500, wn_low=1000):

    """
    The Load_SampleCSV function takes the inputs of the path to a directory
    with all sample CSVs, wavenumber high, and wavenumber low values. The
    function ensures that the wavenumbers are in ascending order and skips
    headers if present.

    Parameters:
        paths (list): A list of file paths to CSV files containing spectral
            data.
        wn_high (float): The highest wavenumber to include in the output
            dictionary.
        wn_low (float): The lowest wavenumber to include in the output
            dictionary.

    Returns:
        tuple: Tuple containing the following elements:
            - files (list): List of names for each sample in the directory.
            - dfs_dict (dict): Dictionary where each key is a file name, and
              the corresponding value is a pandas dataframe containing
              wavenumbers and absorbances for the given sample.
    """

    dfs = []
    files = []

    for path in paths:
        file = os.path.split(path)[1][0:-4]

        try:
            # Attempt to read the CSV assuming no header
            df = pd.read_csv(path, names=['Wavenumber', 'Absorbance'],
                             dtype={'Wavenumber': float})
        except ValueError:
            df = pd.read_csv(path, skiprows=1,
                             names=['Wavenumber', 'Absorbance'],
                             dtype={'Wavenumber': float})

        if df['Wavenumber'].iloc[0] > df['Wavenumber'].iloc[-1]:
            df = df.iloc[::-1]
        df.set_index('Wavenumber', inplace=True)
        spectrum = df.loc[wn_low:wn_high]
        dfs.append(spectrum)
        files.append(file)

    zipobj = zip(files, dfs)
    dfs_dict = dict(zipobj)

    return files, dfs_dict


def Load_PC(file_name):

    """
    Loads predetermined principal components from an NPZ file.

    Parameters:
        file_name (str): The file name of an NPZ file containing principal
            components.

    Returns:
        PC_matrix (matrix): Matrix containing the principal components.
    """

    wn_high = 2400
    wn_low = 1250

    file_path = os.path.join(os.path.dirname(__file__), file_name)
    npz = np.load(file_path)
    PC_DF = pd.DataFrame(npz['data'], columns=npz['columns'])
    PC_DF = PC_DF.set_index('Wavenumber')

    PC_DF = PC_DF.loc[wn_low: wn_high]
    PC_matrix = PC_DF.to_numpy()

    return PC_matrix


def Load_Wavenumber(file_name):

    """
    Loads predetermined wavenumbers from an NPZ file.

    Parameters:
        file_name (str): Path to a CSV file containing wavenumbers.

    Returns:
        Wavenumber (np.ndarray): An array of wavenumbers.
    """

    wn_high = 2400
    wn_low = 1250

    file_path = os.path.join(os.path.dirname(__file__), file_name)
    npz = np.load(file_path)

    Wavenumber_DF = pd.DataFrame(npz['data'], columns=npz['columns'])
    Wavenumber_DF = Wavenumber_DF.set_index('Wavenumber')

    Wavenumber_DF = Wavenumber_DF.loc[wn_low:wn_high]
    Wavenumber = np.array(Wavenumber_DF.index)

    return Wavenumber


def Load_ChemistryThickness(ChemistryThickness_Path):

    """
    Loads dataframes with glass chemistry and thickness data.

    Parameters:
        ChemistryThickness_Path (str): The path to a CSV file containing the
            chemistry and thickness data.

    Returns:
        Tuple containing the following elements:
            Chemistry (pd.DataFrame): The dataframe containing the glass
                chemistry data.
            Thickness (pd.DataFrame): The dataframe containing the thickness
                and standard deviation data.
    """

    ChemistryThickness = pd.read_csv(ChemistryThickness_Path)
    ChemistryThickness.set_index('Sample', inplace=True)

    Chemistry = ChemistryThickness.loc[:, ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3',
                                           'FeO', 'MnO', 'MgO', 'CaO', 'Na2O',
                                           'K2O', 'P2O5']]
    Chemistry = Chemistry.fillna(0)
    Thickness = ChemistryThickness.loc[:, ['Thickness', 'Sigma_Thickness']]

    return Chemistry, Thickness


def Gauss(x, mu, sd, A=1):

    """
    Return a Gaussian fit for the CO3^{2-} doublet peaks at 1515 and
    1430 cm^-1 peaks.

    Parameters:
        x (numeric): The wavenumbers of interest.
        mu (numeric): The center of the peak.
        sd (numeric): The standard deviation (or width of peak).
        A (numeric, optional): The amplitude. Defaults to 1.

    Returns:
        G (np.ndarray): The Gaussian fit.
    """

    G = A * np.exp(-(x - mu) ** 2 / (2 * sd ** 2))

    return G


def Linear(x, m, b):

    """
    Calculate a linear offset for adjusting model data.

    Parameters:
        x (np.ndarray): Input values.
        m (float): Tilt of the linear offset.
        b (float): Offset of the linear offset.

    Returns:
        tilt_offset (np.ndarray): Linear offset.
    """

    tilt_offset = m * np.arange(0, x.size) + b

    return tilt_offset


def Carbonate(P, x, PCmatrix, Peak_1635_PCmatrix, Nvectors=5):

    """
    The Carbonate function takes in the inputs of fitting parameters P,
    wavenumbers x, principal component matrix, number of principal
    component vectors of interest. The function calculates the
    molecular H2O_{1635} peak, CO3^{2-} Gaussian peaks, linear
    offset, and baseline. The function then returns the model data.

    Parameters:
        P (np.ndarray): Fitting parameters including principal component
            and peak weights, Gaussian parameters, linear offset slope
            and intercept.
        x (np.ndarray): Wavenumbers of interest.
        PCmatrix (matrix): Principal components matrix.
        Peak_1635_PCmatrix (matrix): Principal components matrix for the
            1635 peak.
        Nvectors (int, optional): Number of principal components vectors
            of interest. Default is 5.

    Returns:
        model_data (np.ndarray): Model data for the carbonate spectra.
    """

    PC_Weights = np.array([P[0:Nvectors]])
    Peak_Weights = np.array([P[-5:-2]])

    (peak_G1430, std_G1430, G1430_amplitude,
     peak_G1515, std_G1515, G1515_amplitude) = P[Nvectors:-5]
    m, b = P[-2:None]

    Peak_1635 = Peak_Weights @ Peak_1635_PCmatrix.T
    G1515 = Gauss(x, peak_G1515, std_G1515, A=G1515_amplitude)
    G1430 = Gauss(x, peak_G1430, std_G1430, A=G1430_amplitude)

    linear_offset = Linear(x, m, b)

    baseline = PC_Weights @ PCmatrix.T
    model_data = baseline + linear_offset + Peak_1635 + G1515 + G1430
    model_data = np.array(model_data)[0, :]

    return model_data


def als_baseline(intensities, asymmetry_param=0.05, smoothness_param=5e5,
                 max_iters=15, conv_thresh=1e-5, verbose=False):

    """
    Computes the asymmetric least squares baseline.
    http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf

    Parameters:
        intensities (array-like): Input signal to baseline.
        asymmetry_param (float, optional): Asymmetry parameter (p) that
            controls the weights. Defaults to 0.05.
        smoothness_param (float, optional): Smoothness parameter (s) that
            controls the smoothness of the baseline.
        max_iters (int, optional): Maximum number of iterations.
        conv_thresh (float, optional): Convergence threshold. Iteration
            stops when the change in the weights < threshold.
        verbose (bool, optional): If True, prints the iteration number and
            convergence at each iteration.

    Returns:
        z (np.ndarray): Baseline of the input signal.
    """

    smoother = WhittakerSmoother(intensities, smoothness_param, deriv_order=2)
    # Rename p to be concise.
    p = asymmetry_param
    # Initialize weights.
    w = np.ones(intensities.shape[0])

    for i in range(max_iters):
        z = smoother.smooth(w)
        mask = intensities > z
        new_w = p * mask + (1 - p) * (~mask)
        conv = np.linalg.norm(new_w - w)
        if verbose:
            print(i + 1, conv)
        if conv < conv_thresh:
            break
        w = new_w

    else:
        print("ALS did not converge in %d iterations" % max_iters)

    return z


class WhittakerSmoother(object):

    """
    Implements the Whittaker smoother for smoothing a signal.

    Parameters:
        signal (array-like): Input signal to be smoothed.
        smoothness_param (float): Relative importance of smoothness of the
            predicted response.
        deriv_order (int, optional): Order of the derivative of the identity
            matrix.

    Attributes:
        y (array-like): Input signal to be smoothed.
        upper_bands (array-like): Upper triangular bands of the matrix used
            for smoothing.
    """

    def __init__(self, signal, smoothness_param, deriv_order=1):

        self.y = signal
        assert deriv_order > 0, "deriv_order must be an int > 0"
        # Compute the fixed derivative of identity (D).
        d = np.zeros(deriv_order * 2 + 1, dtype=int)
        d[deriv_order] = 1
        d = np.diff(d, n=deriv_order)
        n = self.y.shape[0]
        k = len(d)
        s = float(smoothness_param)

        # Here be dragons: essentially we're faking a big banded matrix D,
        # doing s * D.T.dot(D) with it, then taking the upper triangular bands.
        diag_sums = np.vstack([
            np.pad(s * np.cumsum(d[-i:] * d[:i]), ((k - i, 0),), "constant")
            for i in range(1, k + 1)])

        upper_bands = np.tile(diag_sums[:, -1:], n)
        upper_bands[:, :k] = diag_sums
        for i, ds in enumerate(diag_sums):
            upper_bands[i, -i-1:] = ds[::-1][:i+1]
        self.upper_bands = upper_bands

    def smooth(self, w):
        foo = self.upper_bands.copy()
        foo[-1] += w  # last row is the diagonal
        return solveh_banded(foo, w * self.y, overwrite_ab=True,
                             overwrite_b=True)


def NearIR_Process(data, wn_low, wn_high, peak):

    """
    The NearIR_Process function processes Near-IR absorbance data and returns
    relevant information, such as the fit absorbance data, kriged data, peak
    height, and signal to noise ratio. The function first filters the data,
    then fits a baseline and subtracts it to determine the peak absorbance.
    Next, the data are kriged to further reduce noise and obtain peak height.
    Finally, the signal to noise ratio is calculated, and a warning is issued
    if the ratio is high.  The function is used three times with different
    H2Om wavenumber ranges for uncertainty assessment.

    Parameters:
        data (pd.DataFrame): A dataframe of absorbance data.
        wn_low (int): The Near-IR H2Om or OH wavenumber of interest lower
            bound.
        wn_high (int): The Near-IR H2Om or OH wavenumber of interest upper
            bound.
        peak (str): The H2Om or OH peak of interest.

    Returns:
        Tuple containing the following elements:
            data_output (pd.DataFrame): A dataframe of the absorbance data in
                the region of interest, median filtered data, baseline
                subtracted absorbance, and the subtracted peak.
            krige_output (pd.DataFrame): A dataframe of the kriged data output,
                including the absorbance and standard deviation.
            PH_krige (float): The peak height obtained after kriging.
            STN (float): The signal to noise ratio.
    """

    data_H2O = data.loc[wn_low:wn_high]
    data_output = pd.DataFrame(
        columns=['Absorbance', 'Absorbance_Hat',
                 'BL_NIR_H2O', 'Subtracted_Peak'],
        index=data_H2O.index
    )

    data_output['Absorbance'] = data_H2O
    data_output['Absorbance_Hat'] = signal.medfilt(data_H2O['Absorbance'], 5)
    data_output['BL_NIR_H2O'] = als_baseline(data_output['Absorbance_Hat'],
                                             asymmetry_param=0.001,
                                             smoothness_param=1e9,
                                             max_iters=10, conv_thresh=1e-5)
    data_output['Subtracted_Peak'] = (data_output['Absorbance_Hat'] -
                                      data_output['BL_NIR_H2O'])

    krige_wn_range = np.linspace(wn_low-5, wn_high+5, wn_high-wn_low+11)
    krige_peak = OrdinaryKriging(data_H2O.index,
                                 np.zeros(
                                     data_output['Subtracted_Peak'].shape
                                     ),
                                 data_output['Subtracted_Peak'],
                                 variogram_model='gaussian')
    krige_abs, krige_std = krige_peak.execute("grid", krige_wn_range,
                                              np.array([0.0]))
    krige_output = pd.DataFrame({
        'Absorbance': krige_abs.squeeze(),
        'STD': krige_std.squeeze()
        }, index=krige_wn_range)

    if peak == 'OH':  # 4500 peak
        pr_low, pr_high = 4400, 4600
    elif peak == 'H2O':  # 5200 peak
        pr_low, pr_high = 5100, 5300
    else:
        raise ValueError(f'Invalid peak type: {peak}')

    PH_max = data_output['Subtracted_Peak'].loc[pr_low:pr_high].max()
    PH_krige = (krige_output['Absorbance'].loc[pr_low:pr_high].max() -
                krige_output['Absorbance'].min())

    PH_krige_index = int(data_output['Subtracted_Peak']
                         [data_output['Subtracted_Peak'] == PH_max]
                         .index.to_numpy())
    PH_std = data_output['Subtracted_Peak'].loc[PH_krige_index-50:
                                                PH_krige_index+50].std()
    STN = PH_krige / PH_std

    return data_output, krige_output, PH_krige, STN


def MidIR_Process(data, wn_low, wn_high):

    """
    The MidIR_Process function processes the Mid-IR H2Ot, 3550 peak and
    returns the fit absorbance data, kriged data, peak height. The
    function first filters the data, then fits a baseline and subtracts
    it to determine the peak absorbance. Next, the data are kriged to
    further reduce noise and obtain peak height. The function is used
    three times with different H2Ot wavenumber ranges for uncertainty
    assessment.

    Parameters:
        data (pd.DataFrame): A dataframe of absorbance data.
        wn_low (int): The Near-IR H2O/OH wavenumber of interest lower bound.
        wn_high (int): The Near-IR H2O/OH wavenumber of interest upper bound.
        peak (str): The H2O or OH peak of interest.

    Returns:
        Tuple containing the following elements:
            data_output (pd.DataFrame): A dataframe of the absorbance data in
                the region of interest, median filtered data, baseline
                subtracted absorbance, and the subtracted peak.
            krige_output (pd.DataFrame): A dataframe of the kriged data output,
                including the absorbance and standard deviation.
            PH_krige (float): The peak height obtained after kriging.
    """

    data_H2O3550 = data.loc[wn_low:wn_high]
    data_output = pd.DataFrame(
        columns=[
            'Absorbance', 'BL_MIR_3550',
            'Subtracted_Peak', 'Subtracted_Peak_Hat'
        ],
        index=data_H2O3550.index
    )

    data_output['Absorbance'] = data_H2O3550['Absorbance']
    data_output['BL_MIR_3550'] = als_baseline(data_H2O3550['Absorbance'],
                                              asymmetry_param=0.0010,
                                              smoothness_param=1e11,
                                              max_iters=20,
                                              conv_thresh=1e-7)
    data_output['Subtracted_Peak'] = (data_H2O3550['Absorbance'] -
                                      data_output['BL_MIR_3550'])
    data_output['Subtracted_Peak_Hat'] = signal.medfilt(
        data_output['Subtracted_Peak'], 21)

    peak_wn_low, peak_wn_high = 3300, 3600
    plot_output = data_output.loc[peak_wn_low:peak_wn_high]
    plotindex = np.argmax(plot_output['Absorbance'].index.to_numpy() > 3400)
    PH_3550 = np.max(plot_output['Subtracted_Peak_Hat'])

    return data_output, plot_output, PH_3550, plotindex


def MCMC(data, uncert, indparams, log, savefile):

    """
    Runs Monte Carlo-Markov Chain and outputs the best fit parameters and
    standard deviations.

    Parameters:
        data (np.ndarray): 1D array of data points.
        uncert (np.ndarray): 1D array of uncertainties on the data.
        indparams (np.ndarray): 2D array of independent parameters. Each row
            corresponds to one data point and each column corresponds to a
            different independent parameter.
        log (bool, optional): Flag indicating whether to log the output or not.
            Defaults to False.
        savefile (str, optional): File name to save output to. If None, output
            is not saved. Defaults to None.

    Returns:
        mc3_output (mc3.output): The output of the Monte Carlo-Markov Chain.
    """

    # Define initial values, limits, and step sizes for parameters
    func = Carbonate

    params = np.array([1.25, 2.00, 0.25, 0.005, 0.001, 1430, 30.0, 0.01,
                       1510, 30.0, 0.01, 0.100, 0.020, 0.01, 5e-4, 0.50])
    pmin = np.array([0.00, -3.000, -2.00, -0.600, -0.300,  1415,  22.5,  0.00,
                     1500,  22.50,  0.00,  0.000, -2.000, -2.00, -5e-1, -1.00])
    pmax = np.array([4.00, 3.00, 2.00, 0.600, 0.300, 1445, 40.0, 3.00,
                     1535, 40.0, 3.00, 3.000, 2.000, 2.00, 5e-1, 3.00])
    pstep = np.abs(pmin - pmax) * 0.01

    # Define prior limits for parameters
    prior = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 30.0, 0.0, 0.0,
                      30.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    priorlow = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 5.00, 0.0, 0.0,
                         5.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    priorup = np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.0, 5.00, 0.0, 0.0,
                        5.00, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    pnames = ['B_mean', 'B_PC1', 'B_PC2', 'B_PC3', 'B_PC4',
              'G1430_peak', 'G1430_std', 'G1430_amp',
              'G1515_peak', 'G1515_std', 'G1515_amp',
              'H1635_mean', 'H1635_PC1', 'H1635_PC2', 'm', 'b']

    texnames = ['$\\overline{B}$', '$\\overline{B}_{PC1}$',
                '$\\overline{B}_{PC2}$', '$\\overline{B}_{PC3}$',
                '$\\overline{B}_{PC4}$',
                '$\\mu_{1430}$', '$\\sigma_{1430}$', '$a_{1430}$',
                '$\\mu_{1515}$', '$\\sigma_{1515}$', '$a_{1515}$',
                '$\\overline{H_{1635}}$',
                '$\\overline{H_{1635}}_{PC1}$',
                '$\\overline{H_{1635}}_{PC2}$', '$m$', '$b$']

    mc3_output = mc3.sample(data=data, uncert=uncert, func=func, params=params,
                            indparams=indparams, pmin=pmin, pmax=pmax,
                            pstep=pstep, prior=prior, priorlow=priorlow,
                            priorup=priorup, pnames=pnames, texnames=texnames,
                            sampler='snooker', rms=False, nsamples=1e6,
                            nchains=9, ncpu=4, burnin=2e4, thinning=5,
                            leastsq='trf', chisqscale=False, grtest=True,
                            grbreak=1.01, grnmin=0.5, hsize=10,
                            kickoff='normal', wlike=False, plots=False,
                            log=log, savefile=savefile)

    return mc3_output


def Run_All_Spectra(dfs_dict, exportpath):

    """
    The Run_All_Spectra function inputs the dictionary of dataframes that were
    created by the Load_SampleCSV function and allows for all of the samples to
    be batched and run through the function. The function exports the best fit
    and standard deviations of peak locations, peak widths, and peak heights,
    as well as the principal component vectors used to fit the spectra. These
    values are exported in a csv file and figures are created for each
    individual sample.

    Parameters:
        dfs_dict (dictionary): dictionary containing FTIR data for each file
        exportpath (string or None): string of desired output directory name,
            None if no outputs are desired

    Returns:
        Volatile_PH (pd.DataFrame): dataframe containing all volatile peak
            heights
        failures (list): potential failures
    """

    path_beg = os.getcwd() + '/'

    # Load files with PC vectors for the baseline and H2Om,1635 peak.
    Wavenumber = Load_Wavenumber('BaselineAvgPC.npz')
    PCmatrix = Load_PC('BaselineAvgPC.npz')
    Peak_1635_PCmatrix = Load_PC('H2Om1635PC.npz')
    Nvectors = 5
    indparams = [Wavenumber, PCmatrix, Peak_1635_PCmatrix, Nvectors]

    # Create dataframes to store peak height data:
    # P_ = peak_, _BP = best parameter, #_STD = _stdev
    H2O_3550_PH = pd.DataFrame(columns=['PH_3550_M', 'PH_3550_STD',
                                        'H2OT_3550_MAX',
                                        'BL_H2OT_3550_MAX',
                                        'H2OT_3550_SAT?'])
    DF_Output = pd.DataFrame(columns=['PH_1635_BP',
                                      'PH_1635_STD',
                                      'H2Om_1635_MAX',
                                      'BL_H2Om_1635_MAX',
                                      'PH_1515_BP', 'PH_1515_STD',
                                      'P_1515_BP', 'P_1515_STD',
                                      'STD_1515_BP', 'STD_1515_STD',
                                      'MAX_1515_ABS', 'BL_MAX_1515_ABS',
                                      'PH_1430_BP', 'PH_1430_STD',
                                      'P_1430_BP', 'P_1430_STD',
                                      'STD_1430_BP', 'STD_1430_STD',
                                      'MAX_1430_ABS', 'BL_MAX_1430_ABS'])
    PC_Output = pd.DataFrame(columns=['AVG_BL_BP', 'AVG_BL_STD',
                                      'PC1_BP', 'PC1_STD',
                                      'PC2_BP', 'PC2_STD',
                                      'PC3_BP', 'PC3_STD',
                                      'PC4_BP', 'PC4_STD',
                                      'm_BP', 'm_STD', 'b_BP', 'b_STD',
                                      'PH_1635_PC1_BP', 'PH_1635_PC1_STD',
                                      'PH_1635_PC2_BP', 'PH_1635_PC2_STD'])
    NEAR_IR_PH = pd.DataFrame(columns=['PH_5200_M', 'PH_5200_STD',
                                       'PH_4500_M', 'PH_4500_STD',
                                       'S2N_P5200', 'ERR_5200',
                                       'S2N_P4500', 'ERR_4500'])

    # Initialize lists for failures and errors
    failures = []
    error = []
    error_4500 = []
    error_5200 = []

    # Determine best-fit baselines for all peaks with ALS (H2Om_{5200},
    # OH_{4500}, H2Ot_{3550}) and PyIRoGlass mc3 (H2Om_{1635}, CO3^{2-})
    for files, data in dfs_dict.items():
        try:
            # Three repeat baselines for the OH_{4500}
            H2O4500_wn_low_1, H2O4500_wn_high_1 = 4250, 4675
            (
                data_H2O4500_1, krige_output_4500_1,
                PH_4500_krige_1, STN_4500_1
            ) = NearIR_Process(data, H2O4500_wn_low_1, H2O4500_wn_high_1,
                               'OH')
            H2O4500_wn_low_2, H2O4500_wn_high_2 = 4225, 4650
            (
                data_H2O4500_2, krige_output_4500_2,
                PH_4500_krige_2, STN_4500_2
            ) = NearIR_Process(data, H2O4500_wn_low_2, H2O4500_wn_high_2,
                               'OH')
            H2O4500_wn_low_3, H2O4500_wn_high_3 = 4275, 4700
            (
                data_H2O4500_3, krige_output_4500_3,
                PH_4500_krige_3, STN_4500_3
            ) = NearIR_Process(data, H2O4500_wn_low_3, H2O4500_wn_high_3,
                               'OH')

            # Three repeat baselines for the H2Om_{5200}
            H2O5200_wn_low_1, H2O5200_wn_high_1 = 4875, 5400
            (
                data_H2O5200_1, krige_output_5200_1,
                PH_5200_krige_1, STN_5200_1
            ) = NearIR_Process(data, H2O5200_wn_low_1, H2O5200_wn_high_1,
                               'H2O')
            H2O5200_wn_low_2, H2O5200_wn_high_2 = 4850, 5375
            (
                data_H2O5200_2, krige_output_5200_2,
                PH_5200_krige_2, STN_5200_2
            ) = NearIR_Process(data, H2O5200_wn_low_2, H2O5200_wn_high_2,
                               'H2O')
            H2O5200_wn_low_3, H2O5200_wn_high_3 = 4900, 5425
            (
                data_H2O5200_3, krige_output_5200_3,
                PH_5200_krige_3, STN_5200_3
            ) = NearIR_Process(data, H2O5200_wn_low_3, H2O5200_wn_high_3,
                               'H2O')

            # Kriged peak heights
            PH_4500_krige = np.array([PH_4500_krige_1, PH_4500_krige_2,
                                      PH_4500_krige_3])
            PH_4500_krige_M, PH_4500_krige_STD = (np.mean(PH_4500_krige),
                                                  np.std(PH_4500_krige))
            PH_5200_krige = np.array([PH_5200_krige_1, PH_5200_krige_2,
                                      PH_5200_krige_3])
            PH_5200_krige_M, PH_5200_krige_STD = (np.mean(PH_5200_krige),
                                                  np.std(PH_5200_krige))

            # Calculate signal to noise ratio
            STN_4500 = np.array([STN_4500_1, STN_4500_2, STN_4500_3])
            STN_4500_M = np.average(STN_4500)
            STN_5200 = np.array([STN_5200_1, STN_5200_2, STN_5200_3])
            STN_5200_M = np.average(STN_5200)

            # Consider strength of signal
            if STN_4500_M >= 4.0:
                error_4500 = '-'
            elif STN_4500_M < 4.0:
                error_4500 = '*'

            if STN_5200_M >= 4.0:
                error_5200 = '-'
            elif STN_5200_M < 4.0:
                error_5200 = '*'

            # Save NIR peak heights
            NEAR_IR_PH.loc[files] = pd.Series({
                'PH_5200_M': PH_5200_krige_M,
                'PH_5200_STD': PH_5200_krige_STD,
                'PH_4500_M': PH_4500_krige_M,
                'PH_4500_STD': PH_4500_krige_STD,
                'S2N_P5200': STN_5200_M,
                'ERR_5200': error_5200,
                'S2N_P4500': STN_4500_M,
                'ERR_4500': error_4500
                })

            # Three repeat baselines for the H2Ot_{3550}
            H2O3550_wn_low_1, H2O3550_wn_high_1 = 1900, 4400
            (
                data_H2O3550_1, plot_output_3550_1,
                PH_3550_1, plotindex1
            ) = MidIR_Process(data, H2O3550_wn_low_1, H2O3550_wn_high_1)
            H2O3550_wn_low_2, H2O3550_wn_high_2 = 2100, 4200
            (
                data_H2O3550_2, plot_output_3550_2,
                PH_3550_2, plotindex2
            ) = MidIR_Process(data, H2O3550_wn_low_2, H2O3550_wn_high_2)
            H2O3550_wn_low_3, H2O3550_wn_high_3 = 2300, 4000
            (
                data_H2O3550_3, plot_output_3550_3,
                PH_3550_3, plotindex3
            ) = MidIR_Process(data, H2O3550_wn_low_3, H2O3550_wn_high_3)
            PH_3550_M = np.mean([PH_3550_1, PH_3550_2, PH_3550_3])
            PH_3550_STD = np.std([PH_3550_1, PH_3550_2, PH_3550_3])

            # Determine maximum absorbances for H2Ot_{3550} and H2Om_{1635}
            # to check for saturation
            MAX_3550_ABS = (
                data_H2O3550_1['Absorbance'].to_numpy()
                [np.argmax(data_H2O3550_1['Absorbance'].index.to_numpy()
                           > 3550)]
            )
            MAX_1635_ABS = (
                data_H2O3550_1['Absorbance'].to_numpy()
                [np.argmax((data_H2O3550_1['Absorbance'].index.to_numpy()
                            > 1600) &
                           (data_H2O3550_1['Absorbance'].index.to_numpy()
                            < 1800))]
            )
            BL_MAX_3550_ABS = (
                data_H2O3550_1['BL_MIR_3550'].to_numpy()
                [np.argmax(data_H2O3550_1['Absorbance'].index.to_numpy()
                           > 3550)]
            )

            # Assign * marker to saturated samples H2Ot_{3550} absorbance > 2
            if MAX_3550_ABS < 2:
                error = '-'
            elif MAX_3550_ABS >= 2:
                error = '*'

            # Save H2Om_{3550} peak heights and saturation information
            H2O_3550_PH.loc[files] = pd.Series({
                'PH_3550_M': PH_3550_M,
                'PH_3550_STD': PH_3550_STD,
                'H2OT_3550_MAX': MAX_3550_ABS,
                'BL_H2OT_3550_MAX': BL_MAX_3550_ABS,
                'H2OT_3550_SAT?': error
                })

            # Initialize PyIRoGlass mc3 fit for H2Om_{1635} and CO3^{2-}
            df_length = np.shape(Wavenumber)[0]
            CO2_wn_high, CO2_wn_low = 2400, 1250
            spec = data.loc[CO2_wn_low:CO2_wn_high]

            # Interpolate data to wavenumber spacing, to prepare for mc3
            if spec.shape[0] != df_length:
                interp_wn = np.linspace(spec.index[0], spec.index[-1],
                                        df_length)
                interp_abs = interpolate.interp1d(
                    spec.index, spec['Absorbance'])(interp_wn)
                spec = spec.reindex(index=interp_wn)
                spec['Absorbance'] = interp_abs
                spec_mc3 = spec['Absorbance'].to_numpy()
            elif spec.shape[0] == df_length:
                spec_mc3 = spec['Absorbance'].to_numpy()

            # Set uncertainty
            uncert = np.ones_like(spec_mc3) * 0.01

            # Run PyIRoGlass mc3!!!
            if exportpath is not None:
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                # Create output directories for resulting files
                output_dir = ["FIGURES", "PLOTFILES",
                              "NPZTXTFILES", "LOGFILES"]
                for ii in range(len(output_dir)):
                    if not os.path.exists(path_beg + output_dir[ii] +
                                          '/' + exportpath):
                        os.makedirs(path_beg + output_dir[ii] +
                                    '/' + exportpath, exist_ok=True)

                fpath = 'FIGURES/' + exportpath + '/'
                ppath = path_beg + 'PLOTFILES/' + exportpath + '/'
                sfpath = path_beg + 'NPZTXTFILES/' + exportpath + '/'
                lpath = path_beg + 'LOGFILES/' + exportpath + '/'

                add_dir = ["TRACE", "HISTOGRAM", "PAIRWISE", "MODELFIT"]
                for ii in range(len(add_dir)):
                    if not os.path.exists(ppath+add_dir[ii]):
                        os.makedirs(ppath+add_dir[ii],
                                    exist_ok=True)

                mc3_output = MCMC(data=spec_mc3, uncert=uncert,
                                  indparams=indparams,
                                  log=lpath+files+'.log',
                                  savefile=sfpath+files+'.npz')
            else:
                mc3_output = MCMC(data=spec_mc3, uncert=uncert,
                                  indparams=indparams, log=None, savefile=None)

            # Calculate baselines and peaks for plotting
            CO2P_BP = mc3_output['bestp'][-11:-5]
            H2OmP1635_BP = mc3_output['bestp'][-5:-2]
            CO2P_STD = mc3_output['stdp'][-11:-5]
            H2OmP1635_STD = mc3_output['stdp'][-5:-2]
            H2OmP1635_STD[0] = H2OmP1635_STD[0]

            PC_BP = mc3_output['bestp'][0:Nvectors]
            PC_STD = mc3_output['stdp'][0:Nvectors]
            Baseline_Solve_BP = PC_BP @ PCmatrix.T
            Baseline_Solve_BP = np.asarray(Baseline_Solve_BP).ravel()

            m_BP, b_BP = mc3_output['bestp'][-2:None]
            m_STD, b_STD = mc3_output['stdp'][-2:None]
            Line_BP = Linear(Wavenumber, m_BP, b_BP)
            Baseline_Solve_BP = Baseline_Solve_BP + Line_BP

            H1635_BP = H2OmP1635_BP @ Peak_1635_PCmatrix.T
            H1635_BP = np.asarray(H1635_BP).ravel()
            CO2P1430_BP = Gauss(Wavenumber, CO2P_BP[0], CO2P_BP[1],
                                A=CO2P_BP[2])
            CO2P1515_BP = Gauss(Wavenumber, CO2P_BP[3], CO2P_BP[4],
                                A=CO2P_BP[5])
            H1635_SOLVE = H1635_BP + Baseline_Solve_BP
            CO2P1515_SOLVE = CO2P1515_BP + Baseline_Solve_BP
            CO2P1430_SOLVE = CO2P1430_BP + Baseline_Solve_BP

            MAX_1515_ABS = (spec['Absorbance'].to_numpy()
                            [np.argmax((spec.index.to_numpy() > 1510)
                                       & (spec.index.to_numpy() < 1530))])
            MAX_1430_ABS = (spec['Absorbance'].to_numpy()
                            [np.argmax((spec.index.to_numpy() > 1410)
                                       & (spec.index.to_numpy() < 1440))])

            BL_MAX_1515_ABS = (Baseline_Solve_BP
                               [np.argmax((spec.index.to_numpy() > 1510)
                                          & (spec.index.to_numpy() < 1530))])
            BL_MAX_1430_ABS = (Baseline_Solve_BP
                               [np.argmax((spec.index.to_numpy() > 1410)
                                          & (spec.index.to_numpy() < 1450))])
            BL_MAX_1635_ABS = (Baseline_Solve_BP[np.argmax(H1635_BP)])

            warnings.filterwarnings("ignore", module="matplotlib\\..*")

            # Initialize plotting, create subplots of H2Om_{5200} and OH_{4500}
            # baselines and peak fits
            if exportpath is not None:
                plotmin = np.round(np.min(
                    data.loc[H2O4500_wn_low_1:H2O5200_wn_high_1]['Absorbance']
                    ), decimals=1)
                plotmax = np.round(np.max(
                    data.loc[H2O4500_wn_low_1:H2O5200_wn_high_1]['Absorbance']
                    ), decimals=1)
                fig, ax = plt.subplots(figsize=(26, 8))
                ax1 = plt.subplot2grid((2, 3), (0, 0))
                ax1.plot(data.index, data['Absorbance'], 'k', linewidth=1.5)
                ax1.plot(data_H2O5200_1.index,
                         data_H2O5200_1['Absorbance_Hat'],
                         data_H2O5200_2.index,
                         data_H2O5200_2['Absorbance_Hat'],
                         data_H2O5200_3.index,
                         data_H2O5200_3['Absorbance_Hat'])
                ax1.plot(data_H2O4500_1.index,
                         data_H2O4500_1['Absorbance_Hat'],
                         data_H2O4500_2.index,
                         data_H2O4500_2['Absorbance_Hat'],
                         data_H2O4500_3.index,
                         data_H2O4500_3['Absorbance_Hat'])
                ax1.plot(data_H2O5200_1.index,
                         data_H2O5200_1['BL_NIR_H2O'],
                         'lightsteelblue',
                         data_H2O5200_2.index,
                         data_H2O5200_2['BL_NIR_H2O'],
                         'lightsteelblue',
                         data_H2O5200_3.index,
                         data_H2O5200_3['BL_NIR_H2O'],
                         'lightsteelblue')
                ax1.plot(data_H2O4500_1.index,
                         data_H2O4500_1['BL_NIR_H2O'],
                         'silver',
                         data_H2O4500_2.index,
                         data_H2O4500_2['BL_NIR_H2O'],
                         'silver',
                         data_H2O4500_3.index,
                         data_H2O4500_3['BL_NIR_H2O'],
                         'silver')
                ax1.annotate(
                    r"$\mathregular{H_2O_{m, 5200}}$ Peak Height: "
                    + f"{PH_5200_krige_M:.4f} "
                    + r"± "
                    + f"{PH_5200_krige_STD:.4f}, S2N={STN_5200_M:.2f}",
                    (0.025, 0.9), xycoords='axes fraction'
                )
                ax1.annotate(
                    r"$\mathregular{OH^{-}_{4500}}$ Peak Height: "
                    + f"{PH_4500_krige_M:.4f} "
                    + r"± "
                    + f"{PH_4500_krige_STD:.4f}, S2N={STN_4500_M:.2f}",
                    (0.025, 0.8), xycoords='axes fraction'
                )
                ax1.set_ylabel('Absorbance')
                warnings.filterwarnings("ignore", category=UserWarning)
                ax1.legend(
                    [
                        'NIR Spectrum', '_', '_',
                        r'$\mathregular{H_2O_{m, 5200}}$ Median Filtered',
                        '_', '_',
                        r'$\mathregular{OH^{-}_{4500}}$ Median Filtered',
                        '_', '_',
                        r'$\mathregular{H_2O_{m, 5200}}$ Baseline',
                        '_', '_',
                        r'$\mathregular{OH^{-}_{4500}}$ Baseline'
                    ],
                    prop={'size': 10}
                )
                ax1.set_xlim([4200, 5400])
                ax1.set_ylim([plotmin-0.075, plotmax+0.075])
                ax1.invert_xaxis()

                plotmax = np.round(np.max(data_H2O4500_1['Subtracted_Peak']),
                                   decimals=1)
                ax2 = plt.subplot2grid((2, 3), (1, 0))
                ax2.plot(
                    data_H2O5200_1.index,
                    (data_H2O5200_1['Subtracted_Peak'] -
                     np.min(krige_output_5200_1['Absorbance'])), 'k',
                    label=r'$\mathregular{H_2O_{m, 5200}}$ Baseline Subtracted'
                )
                ax2.plot(data_H2O5200_2.index,
                         (data_H2O5200_2['Subtracted_Peak'] -
                          np.min(krige_output_5200_2['Absorbance'])), 'k')
                ax2.plot(data_H2O5200_3.index,
                         (data_H2O5200_3['Subtracted_Peak'] -
                          np.min(krige_output_5200_3['Absorbance'])), 'k')

                ax2.plot(
                    data_H2O4500_1.index,
                    (data_H2O4500_1['Subtracted_Peak'] -
                     np.min(krige_output_4500_1['Absorbance'])), 'k',
                    label=r'$\mathregular{OH^{-}_{4500}}$ Baseline Subtracted'
                )
                ax2.plot(data_H2O4500_2.index,
                         (data_H2O4500_2['Subtracted_Peak'] -
                          np.min(krige_output_4500_2['Absorbance'])), 'k')
                ax2.plot(data_H2O4500_3.index,
                         (data_H2O4500_3['Subtracted_Peak'] -
                          np.min(krige_output_4500_3['Absorbance'])), 'k')

                ax2.plot(
                    krige_output_5200_1.index,
                    (krige_output_5200_1['Absorbance'] -
                     np.min(krige_output_5200_1['Absorbance'])),
                    label=r'$\mathregular{H_2O_{m, 5200}}$ Kriged Peak'
                )
                ax2.plot(krige_output_5200_2.index,
                         (krige_output_5200_2['Absorbance'] -
                          np.min(krige_output_5200_2['Absorbance'])))
                ax2.plot(krige_output_5200_3.index,
                         (krige_output_5200_3['Absorbance'] -
                          np.min(krige_output_5200_3['Absorbance'])))

                ax2.plot(
                    krige_output_4500_1.index,
                    (krige_output_4500_1['Absorbance'] -
                     np.min(krige_output_4500_1['Absorbance'])),
                    label=r'$\mathregular{OH^{-}_{4500}}$ Kriged Peak'
                )
                ax2.plot(krige_output_4500_2.index,
                         (krige_output_4500_2['Absorbance'] -
                          np.min(krige_output_4500_2['Absorbance'])))
                ax2.plot(krige_output_4500_3.index,
                         (krige_output_4500_3['Absorbance'] -
                          np.min(krige_output_4500_3['Absorbance'])))
                ax2.set_xlabel(r'Wavenumber $(\mathregular{cm^{-1}})$')
                ax2.set_ylabel('Absorbance')
                warnings.filterwarnings("ignore", category=UserWarning)
                ax2.legend(
                    [
                        r'$\mathregular{H_2O_{m, 5200}}$ Baseline Subtracted',
                        '_', '_',
                        r'$\mathregular{OH^{-}_{4500}}$ Baseline Subtracted',
                        '_', '_', '_', '_',
                        r'$\mathregular{H_2O_{m, 5200}}$ Kriged',
                        '_', '_',
                        r'$\mathregular{OH^{-}_{4500}}$ Kriged'
                    ],
                    prop={'size': 10}
                )
                ax2.set_xlim([4200, 5400])
                ax2.set_ylim([0, plotmax+0.05])
                ax2.invert_xaxis()

                # Create subplot of H2Om_{3550} baselines and peak fits
                plotmax = np.round(np.max(
                    data.loc[1250:4000]['Absorbance'].to_numpy()
                    ), decimals=0)
                plotmin = np.round(np.min(
                    data.loc[1250:4000]['Absorbance'].to_numpy()
                    ), decimals=0)
                ax3 = plt.subplot2grid((2, 3), (0, 1), rowspan=2)
                ax3.plot(data.index, data['Absorbance'], 'k')
                ax3.plot(data_H2O3550_1['Absorbance'].index,
                         data_H2O3550_1['BL_MIR_3550'], 'silver',
                         label=r'$\mathregular{H_2O_{t, 3550}}$ Baseline')
                ax3.plot(data_H2O3550_2['Absorbance'].index,
                         data_H2O3550_2['BL_MIR_3550'], 'silver')
                ax3.plot(data_H2O3550_3['Absorbance'].index,
                         data_H2O3550_3['BL_MIR_3550'], 'silver')
                ax3.plot(plot_output_3550_1.index,
                         (plot_output_3550_1['Subtracted_Peak_Hat'] +
                          plot_output_3550_1['BL_MIR_3550']), 'r', linewidth=2)
                ax3.plot(plot_output_3550_2.index,
                         (plot_output_3550_2['Subtracted_Peak_Hat'] +
                          plot_output_3550_2['BL_MIR_3550']), 'r', linewidth=2)
                ax3.plot(plot_output_3550_3.index,
                         (plot_output_3550_3['Subtracted_Peak_Hat'] +
                          plot_output_3550_3['BL_MIR_3550']), 'r', linewidth=2)
                ax3.set_title(files)
                ax3.annotate(r"$\mathregular{H_2O_{t, 3550}}$  Peak Height: " +
                             f"{PH_3550_M:.4f} ± {PH_3550_STD:.4f}",
                             (0.025, 0.95), xycoords='axes fraction')
                ax3.set_xlabel(r'Wavenumber $(\mathregular{cm^{-1}})$')
                ax3.set_xlim([1250, 4000])
                ax3.set_ylabel('Absorbance')
                ax3.set_ylim([plotmin-0.25, plotmax+0.5])
                ax3.legend(loc='upper right', prop={'size': 10})
                ax3.invert_xaxis()

                # Load posterior/sampling errors to plot baseline ranges.
                mcmc_npz = np.load(sfpath+files+'.npz')
                posterior = mcmc_npz['posterior']
                mask = mcmc_npz['zmask']
                masked_posterior = posterior[mask]

                samplingerror = masked_posterior[:, 0:5]
                samplingerror = samplingerror[
                    0: np.shape(masked_posterior[:, :])[0]:
                    int(np.shape(masked_posterior[:, :])[0] / 100), :
                    ]
                lineerror = masked_posterior[:, -2:None]
                lineerror = lineerror[
                    0:np.shape(masked_posterior[:, :])[0]:
                    int(np.shape(masked_posterior[:, :])[0] / 100), :
                    ]

                Baseline_Array = np.array(samplingerror @ PCmatrix[:, :].T)
                Baseline_Array_Plot = Baseline_Array

                ax4 = plt.subplot2grid((2, 3), (0, 2), rowspan=2)
                for i in range(np.shape(Baseline_Array)[0]):
                    Linearray = Linear(Wavenumber, lineerror[i, 0],
                                       lineerror[i, 1])
                    Baseline_Array_Plot[i, :] = (Baseline_Array[i, :] +
                                                 Linearray)
                    plt.plot(Wavenumber, Baseline_Array_Plot[i, :],
                             'dimgray', linewidth=0.25)
                ax4.plot(Wavenumber, spec_mc3,
                         'tab:blue', linewidth=2.5,
                         label='FTIR Spectrum')
                ax4.plot(Wavenumber, H1635_SOLVE,
                         'tab:orange', linewidth=1.5,
                         label=r'$\mathregular{H_2O_{m, 1635}}$')
                ax4.plot(Wavenumber, CO2P1515_SOLVE,
                         'tab:green', linewidth=2.5,
                         label=r'$\mathregular{CO_{3, 1515}^{2-}}$')
                ax4.plot(Wavenumber, CO2P1430_SOLVE,
                         'tab:red', linewidth=2.5,
                         label=r'$\mathregular{CO_{3, 1430}^{2-}}$')
                ax4.plot(Wavenumber,
                         Carbonate(mc3_output['meanp'], Wavenumber, PCmatrix,
                                   Peak_1635_PCmatrix, Nvectors),
                         'tab:purple', linewidth=1.5,
                         label='PyIRoGlass Fit')
                ax4.plot(Wavenumber, Baseline_Solve_BP, 'k', linewidth=1.5,
                         label='Baseline')
                ax4.annotate(
                    r"$\mathregular{H_2O_{m, 1635}}$ Peak Height: "
                    + f"{H2OmP1635_BP[0]:.3f} ± {H2OmP1635_STD[0]:.3f}",
                    (0.025, 0.95), xycoords='axes fraction'
                )
                ax4.annotate(
                    r"$\mathregular{CO_{3, 1515}^{2-}}$ Peak Height: "
                    + f"{CO2P_BP[5]:.3f} ± {CO2P_STD[5]:.3f}",
                    (0.025, 0.90), xycoords='axes fraction'
                )
                ax4.annotate(
                    r"$\mathregular{CO_{3, 1430}^{2-}}$ Peak Height: "
                    + f"{CO2P_BP[2]:.3f} ± {CO2P_STD[2]:.3f}",
                    (0.025, 0.85), xycoords='axes fraction'
                )
                ax4.set_xlim([1250, 2400])
                ax4.set_xlabel(r'Wavenumber $(\mathregular{cm^{-1}})$')
                ax4.set_ylabel('Absorbance')
                ax4.legend(loc='upper right', prop={'size': 10})
                ax4.invert_xaxis()
                plt.tight_layout()
                plt.savefig(path_beg + fpath + files + '.pdf')
                plt.close('all')

                texnames = ['$\\overline{B}$', '$\\overline{B}_{PC1}$',
                            '$\\overline{B}_{PC2}$', '$\\overline{B}_{PC3}$',
                            '$\\overline{B}_{PC4}$',
                            '$\\mu_{1430}$', '$\\sigma_{1430}$', '$a_{1430}$',
                            '$\\mu_{1515}$', '$\\sigma_{1515}$', '$a_{1515}$',
                            '$\\overline{H_{1635}}$',
                            '$\\overline{H_{1635}}_{PC1}$',
                            '$\\overline{H_{1635}}_{PC2}$', '$m$', '$b$']

                posterior, zchain, zmask = mu.burn(mc3_output)
                post = mp.Posterior(posterior, texnames)

                post.plot()
                plt.suptitle(files)
                plt.savefig(ppath+'PAIRWISE/'+files+'_pairwise.pdf')
                plt.close('all')

                post.plot_histogram(nx=4, ny=4)
                plt.suptitle(files, y=1.015)
                plt.savefig(ppath+'HISTOGRAM/'+files+'_histogram.pdf',
                            bbox_inches='tight')
                plt.close('all')

                trace(mc3_output['posterior'], title=files,
                      zchain=mc3_output['zchain'],
                      burnin=mc3_output['burnin'],
                      pnames=texnames, savefile=(ppath + 'TRACE/' +
                                                 files + '_trace.pdf'))
                plt.close('all')

                modelfit(spec_mc3, uncert, indparams[0],
                         mc3_output['best_model'],
                         title=files, savefile=(ppath + 'MODELFIT/' +
                                                files + '_modelfit.pdf'))
                plt.close('all')

        except Exception as e:
            failures.append(files)
            print(f"{files} failed. Reason: {str(e)}")

        else:
            pass

            # Create dataframe of best fit parameters and standard deviations
            DF_Output.loc[files] = pd.Series({
                'PH_1635_BP': H2OmP1635_BP[0],
                'PH_1635_STD': H2OmP1635_STD[0],
                'H2Om_1635_MAX': MAX_1635_ABS,
                'BL_H2Om_1635_MAX': BL_MAX_1635_ABS,
                'PH_1515_BP': CO2P_BP[5],
                'PH_1515_STD': CO2P_STD[5],
                'P_1515_BP': CO2P_BP[3],
                'P_1515_STD': CO2P_STD[3],
                'STD_1515_BP': CO2P_BP[4],
                'STD_1515_STD': CO2P_STD[4],
                'PH_1430_BP': CO2P_BP[2],
                'PH_1430_STD': CO2P_STD[2],
                'P_1430_BP': CO2P_BP[0],
                'P_1430_STD': CO2P_STD[0],
                'STD_1430_BP': CO2P_BP[1],
                'STD_1430_STD': CO2P_STD[1],
                'MAX_1515_ABS': MAX_1515_ABS,
                'BL_MAX_1515_ABS': BL_MAX_1515_ABS,
                'MAX_1430_ABS': MAX_1430_ABS,
                'BL_MAX_1430_ABS': BL_MAX_1430_ABS
                })

            PC_Output.loc[files] = pd.Series({
                'AVG_BL_BP': PC_BP[0],
                'AVG_BL_STD': PC_STD[0],
                'PC1_BP': PC_BP[1],
                'PC1_STD': PC_STD[1],
                'PC2_BP': PC_BP[2],
                'PC2_STD': PC_STD[2],
                'PC3_BP': PC_BP[3],
                'PC3_STD': PC_STD[3],
                'PC4_BP': PC_BP[4],
                'PC4_STD': PC_STD[4],
                'm_BP': m_BP,
                'm_STD': m_STD,
                'b_BP': b_BP,
                'b_STD': b_STD,
                'PH_1635_PC1_BP': H2OmP1635_BP[1],
                'PH_1635_PC1_STD': H2OmP1635_STD[1],
                'PH_1635_PC2_BP': H2OmP1635_BP[2],
                'PH_1635_PC2_STD': H2OmP1635_STD[2]})

    Volatile_PH = pd.concat([H2O_3550_PH, DF_Output, NEAR_IR_PH, PC_Output],
                            axis=1)

    return Volatile_PH, failures


def Beer_Lambert(molar_mass, absorbance, density, thickness, epsilon):

    """
    Applies the Beer-Lambert Law to calculate concentration from given inputs.

    Parameters:
        molar_mass (float): The molar mass of the substance in grams per mole.
        absorbance (float): The absorbance of the substance, measured in
            optical density units.
        density (float): The density of the substance in grams per cubic
            centimeter.
        thickness (float): The thickness of the sample in centimeters.
        epsilon (float): The molar extinction coefficient, measured in liters
            per mole per centimeter.

    Returns:
        pd.Series: pandas series containing the concentration in wt.% or ppm.

    Notes:
        The Beer-Lambert Law states that the absorbance of a substance is
        proportional to its concentration. The formula for calculating
        concentration from absorbance is:
        concentration = (1e6*molar_mass*absorbance)/(density*thickness*epsilon)
        # https://sites.fas.harvard.edu/~scphys/nsta/error_propagation.pdf
    """

    concentration = pd.Series(dtype='float')
    concentration = ((1e6 * molar_mass * absorbance) /
                     (density * thickness * epsilon))

    return concentration


def Beer_Lambert_Error(N, molar_mass,
                       absorbance, sigma_absorbance,
                       density, sigma_density,
                       thickness, sigma_thickness,
                       epsilon, sigma_epsilon):

    """
    Applies a Monte Carlo simulation to estimate the uncertainty in
    concentration calculated using the Beer-Lambert Law.

    Parameters:
        N (int): The number of Monte Carlo samples to generate.
        molar_mass (float): The molar mass of the substance in grams
            per mole.
        absorbance (float): The absorbance of the substance, measured
            in optical density units.
        sigma_absorbance (float): The uncertainty associated with the
            absorbance measurement.
        density (float): The density of the substance in grams per
            cubic centimeter.
        sigma_density (float): The uncertainty associated with the
            density measurement.
        thickness (float): The thickness of the sample in centimeters.
        sigma_thickness (float): The uncertainty associated with the
            sample thickness measurement.
        epsilon (float): The molar extinction coefficient, measured
            in liters per mole per centimeter.
        sigma_epsilon (float): The uncertainty associated with the molar
            extinction coefficient measurement.

    Returns:
        float: The estimated uncertainty in concentration in wt.% or ppm.

    Notes:
        The Beer-Lambert Law states that the absorbance of a substance is
        proportional to its concentration. The formula for calculating
        concentration from absorbance is:
        concentration = (1e6*molar_mass*absorbance)/(density*thickness*epsilon)
        This function estimates the uncertainty in concentration using a
        Monte Carlo simulation. Absorbance, density, and thickness are assumed
        to follow Gaussian distributions with means from the input values
        and standard deviations from the input uncertainties. The absorbance
        coefficient is assumed to follow a uniform distribution with a mean
        given by the input value and a range from the input uncertainty.
        The function returns the standard deviation of the resulting
        concentration distribution.

        # https://astrofrog.github.io/py4sci/_static/Practice%20Problem%20-%20Monte-Carlo%20Error%20Propagation%20-%20Sample%20Solution.html
    """

    gaussian_concentration = (
        (1e6 * molar_mass * np.random.normal(absorbance, sigma_absorbance, N))
        / (
            np.random.normal(density, sigma_density, N) *
            np.random.normal(thickness, sigma_thickness, N) *
            np.random.uniform(epsilon-sigma_epsilon, epsilon+sigma_epsilon, N)
        )
    )

    concentration_std = np.std(gaussian_concentration)

    return concentration_std


def Density_Calculation(Composition, T, P):

    """
    The Density_Calculation function inputs the MI composition file and outputs
    the glass density at the temperature and pressure of analysis. The mole
    fraction is calculated. The total molar volume xivibari is determined from
    sum of the mole fractions of each oxide * partial molar volume at room
    temperature and pressure of analysis. The gram formula weight gfw is then
    determined by summing the mole fractions*molar masses. The density is
    finally determined by dividing gram formula weight by total molar volume.

    Parameters:
        Composition: dataframe containing oxide weight percentage for the glass
            composition
        T: temperature at which the density is calculated (in Celsius)
        P: pressure at which the density is calculated (in bars)

    Returns:
        mol: dataframe containing the oxide mole fraction for the glass
            composition
        density: glass density at room temperature and pressure (in kg/m^3)
    """

    # Define a dictionary of molar masses for each oxide
    molar_mass = {
        'SiO2': 60.08,
        'TiO2': 79.866,
        'Al2O3': 101.96,
        'Fe2O3': 159.69,
        'FeO': 71.844,
        'MnO': 70.9374,
        'MgO': 40.3044,
        'CaO': 56.0774,
        'Na2O': 61.9789,
        'K2O': 94.2,
        'P2O5': 141.9445,
        'H2O': 18.01528,
        'CO2': 44.01
        }

    # Convert room temperature from Celsius to Kelvin
    T_K = T + 273.15

    # Define a dictionary of partial molar volumes for each oxide, based on
    # data from Lesher and Spera (2015)
    par_molar_vol = {
        'SiO2': (26.86-1.89*P/1000),
        'TiO2': (23.16+7.24*(T_K-1673)/1000-2.31*P/1000),
        'Al2O3': (37.42-2.26*P/1000),
        'Fe2O3': (42.13+9.09*(T_K-1673)/1000-2.53*P/1000),
        'FeO': (13.65+2.92*(T_K-1673)/1000-0.45*P/1000),
        'MgO': (11.69+3.27*(T_K-1673)/1000+0.27*P/1000),
        'CaO': (16.53+3.74*(T_K-1673)/1000+0.34*P/1000),
        'Na2O': (28.88+7.68*(T_K-1673)/1000-2.4*P/1000),
        'K2O': (45.07+12.08*(T_K-1673)/1000-6.75*P/1000),
        'H2O': (26.27+9.46*(T_K-1673)/1000-3.15*P/1000)
        }

    # Create an dataframe to store oxide moles for composition
    mol = pd.DataFrame()
    # Calculate oxide moles by dividing weight percentage by molar mass
    for oxide in Composition:
        mol[oxide] = Composition[oxide]/molar_mass[oxide]

    # Calculate the total moles for the MI composition
    mol_tot = pd.DataFrame()
    mol_tot = mol.sum(axis=1)

    # Create empty dataframes to store the partial molar volume, gram formula
    # weight, oxide density, and the product of mole fraction and partial
    # molar volume for each oxide
    xivbari = pd.DataFrame()
    gfw = pd.DataFrame()
    for oxide in Composition:
        # If the oxide is included in the partial molar volume dictionary,
        # calculate its partial molar volume and gram formula weight
        if oxide in par_molar_vol:
            # partial molar volume
            xivbari[oxide] = mol[oxide]/mol_tot*par_molar_vol[oxide]
            # gram formula weight
            gfw[oxide] = mol[oxide]/mol_tot*molar_mass[oxide]

    xivbari_tot = xivbari.sum(axis=1)
    gfw_tot = gfw.sum(axis=1)

    # Calculate density of glass, convert by *1000 for values in kg/m^3
    density = 1000 * gfw_tot / xivbari_tot

    return mol, density


def Epsilon_Calculation(Composition, T, P):

    """
    The Epsilon_Calculation function computes the extinction coefficients
    and their uncertainties for various molecular species in a given MI
    or glass composition dataset.

    Parameters:
        Composition (dictionary): Dictionary containing the weight percentages
            of each oxide in the glass composition
        T (int): temperature at which the density is calculated (in Celsius)
        P (int): pressure at which the density is calculated (in bars)

    Returns:
        mol (pd.DataFrame): Dataframe of the mole fraction of each oxide in
            the glass composition
        density (pd.DataFrame): Dataframe of glass density at room temperature
            and pressure (in kg/m^3)
    """

    epsilon = pd.DataFrame(
        columns=[
            'Tau',
            'Eta',
            'eps_H2OT_3550',
            'sigma_eps_H2OT_3550',
            'eps_H2Om_1635',
            'sigma_eps_H2Om_1635',
            'eps_CO2',
            'sigma_eps_CO2',
            'eps_H2Om_5200',
            'sigma_eps_H2Om_5200',
            'eps_OH_4500',
            'sigma_eps_OH_4500'
            ]
            )

    mol, density = Density_Calculation(Composition, T, P)

    # Calculate extinction coefficient
    cation_tot = (mol.sum(axis=1) +
                  mol['Al2O3'] + mol['Na2O'] + mol['K2O'] + mol['P2O5'])
    Na_NaCa = (2*mol['Na2O']) / ((2*mol['Na2O']) + mol['CaO'])
    SiAl_tot = (mol['SiO2'] + (2*mol['Al2O3'])) / cation_tot

    # Set up extinction coefficient inversion best-fit parameters and
    # covariance matrices
    mest_3550 = np.array([15.725557, 71.368691])
    mest_1635 = np.array([-50.397564, 124.250534])
    mest_CO2 = np.array([426.66290034, -334.45444392])
    covm_est_3550 = np.diag([38.4640, 77.8597])
    covm_est_1635 = np.diag([20.8503, 39.3875])
    covm_est_CO2 = np.diag([93.85345732, 359.94988573])
    mest_4500 = np.array([-1.632730, 3.532522])
    mest_5200 = np.array([-2.291420, 4.675528])
    covm_est_4500 = np.diag([0.0329, 0.0708])
    covm_est_5200 = np.diag([0.0129, 0.0276])

    # Set up matrices for calculating uncertainties on extinction coefficients
    G_SiAl = np.ones((2, 1))
    G_NaCa = np.ones((2, 1))
    covz_error_SiAl = np.zeros((2, 2))
    covz_error_NaCa = np.zeros((2, 2))

    # Loop through and calculate for all MI or glass compositions.
    for i in Composition.index:

        # Calculate extinction coefficients with best-fit parameters
        eps_H2OT_3550 = mest_3550[0]+(mest_3550[1]*SiAl_tot[i])
        eps_H2Om_1635 = mest_1635[0]+(mest_1635[1]*SiAl_tot[i])
        eps_CO2 = mest_CO2[0]+(mest_CO2[1]*Na_NaCa[i])
        eps_H2Om_5200 = mest_5200[0]+(mest_5200[1]*SiAl_tot[i])
        eps_OH_4500 = mest_4500[0]+(mest_4500[1]*SiAl_tot[i])

        # Calculate extinction coefficient uncertainties
        G_SiAl[1, 0] = SiAl_tot[i]
        G_NaCa[1, 0] = Na_NaCa[i]
        covz_error_SiAl[1, 1] = SiAl_tot[i] * 0.01  # 1 sigma
        covz_error_NaCa[1, 1] = Na_NaCa[i] * 0.01

        CT_int_3550 = ((G_SiAl*covm_est_3550*np.transpose(G_SiAl)) +
                       (mest_3550*covz_error_SiAl*np.transpose(mest_3550)))
        CT68_3550 = (np.mean(np.diag(CT_int_3550)))**(1/2)

        CT_int_1635 = ((G_SiAl*covm_est_1635*np.transpose(G_SiAl)) +
                       (mest_1635*covz_error_SiAl*np.transpose(mest_1635)))
        CT68_1635 = (np.mean(np.diag(CT_int_1635)))**(1/2)

        CT_int_CO2 = ((G_NaCa*covm_est_CO2*np.transpose(G_NaCa)) +
                      (mest_CO2*covz_error_NaCa*np.transpose(mest_CO2)))
        CT68_CO2 = (np.mean(np.diag(CT_int_CO2)))**(1/2)

        CT_int_5200 = ((G_SiAl*covm_est_5200*np.transpose(G_SiAl)) +
                       (mest_5200*covz_error_SiAl*np.transpose(mest_5200)))
        CT68_5200 = (np.mean(np.diag(CT_int_5200)))**(1/2)
        CT_int_4500 = ((G_SiAl*covm_est_4500*np.transpose(G_SiAl)) +
                       (mest_4500*covz_error_SiAl*np.transpose(mest_4500)))
        CT68_4500 = (np.mean(np.diag(CT_int_4500)))**(1/2)

        # Save outputs of extinction coefficients to dataframe epsilon
        epsilon.loc[i] = pd.Series({
            'Tau': SiAl_tot[i],
            'Na/Na+Ca': Na_NaCa[i],
            'eps_H2OT_3550': eps_H2OT_3550,
            'sigma_eps_H2OT_3550': CT68_3550,
            'eps_H2Om_1635': eps_H2Om_1635,
            'sigma_eps_H2Om_1635': CT68_1635,
            'eps_CO2': eps_CO2,
            'sigma_eps_CO2': CT68_CO2,
            'eps_H2Om_5200': eps_H2Om_5200,
            'sigma_eps_H2Om_5200': CT68_5200,
            'eps_OH_4500': eps_OH_4500,
            'sigma_eps_OH_4500': CT68_4500
            })

    return epsilon


def Concentration_Output(Volatile_PH, thick, Composition, N=500000, T=25, P=1):

    """
    The Concentration_Output function inputs the peak heights for the total
    H2O peak (3550 cm^-1), molecular H2O peak (1635 cm^-1), and carbonate
    peaks (1515 and 1430 cm^-1), number of samples for the Monte Carlo,
    thickness information, and MI composition, and outputs the concentrations
    and uncertainties for each peak.

    Parameters:
        Volatile_PH: dataframe containing all volatile peak heights
        N: the number of Monte Carlo samples to generate
        thick: Wafer thickness (in µm)
        Composition: dictionary containing the weight percentages of each oxide
            in the glass composition
        T: temperature at which the density is calculated (in Celsius)
        P: pressure at which the density is calculated (in bars)

    Returns:
        density_epsilon: dataframe containing information on glass densities
            and extinction coefficient
        spreadsheet_f: volatile concentrations
    """

    # Define a dictionary of molar masses for each oxide
    molar_mass = {
        'SiO2': 60.08,
        'TiO2': 79.866,
        'Al2O3': 101.96,
        'Fe2O3': 159.69,
        'FeO': 71.844,
        'MnO': 70.9374,
        'MgO': 40.3044,
        'CaO': 56.0774,
        'Na2O': 61.9789,
        'K2O': 94.2,
        'P2O5': 141.9445,
        'H2O': 18.01528,
        'CO2': 44.01
        }

    # Create dataframes to store volatile data:
    spreadsheet = pd.DataFrame(
        columns=[
            'H2OT_MEAN',
            'H2OT_STD',
            'H2OT_3550_M',
            'H2OT_3550_STD',
            'H2OT_3550_SAT',
            'H2Om_1635_BP',
            'H2Om_1635_STD',
            'CO2_MEAN',
            'CO2_STD',
            'CO2_1515_BP',
            'CO2_1515_STD',
            'CO2_1430_BP',
            'CO2_1430_STD',
            'H2Om_5200_M',
            'H2Om_5200_STD',
            'OH_4500_M',
            'OH_4500_STD'
            ]
            )

    # Dataframe for saturated concentrations
    spreadsheet_sat = pd.DataFrame(columns=spreadsheet.columns)

    # Dataframe for storing glass density data
    density_df = pd.DataFrame(columns=['Density'])
    # Dataframe for storing saturated glass density data
    density_sat_df = pd.DataFrame(columns=['Density_Sat'])
    # Dataframe for storing mean volatile data
    mean_vol = pd.DataFrame(columns=['H2OT_MEAN', 'H2OT_STD',
                                     'CO2_MEAN', 'CO2_STD'])
    # Dataframe for storing signal-to-noise error data
    s2nerror = pd.DataFrame(columns=['PH_5200_S2N', 'ERR_5200',
                                     'PH_4500_S2N', 'ERR_4500'])

    # Initialize density calculation with 0 wt.% H2O.
    Composition['H2O'] = 0
    mol, density = Density_Calculation(Composition, T, P)
    epsilon = Epsilon_Calculation(Composition, T, P)

    # Doing density-H2O iterations:
    for jj in range(10):
        H2OT_3550_I = Beer_Lambert(
            molar_mass['H2O'],
            Volatile_PH['PH_3550_M'],
            density,
            thick['Thickness'],
            epsilon['eps_H2OT_3550'],
            )
        Composition['H2O'] = H2OT_3550_I
        mol, density = Density_Calculation(Composition, T, P)

    # Doing density-H2O iterations:
    for kk in Volatile_PH.index:
        # Calculate volatile species concentrations
        H2OT_3550_M = Beer_Lambert(
            molar_mass['H2O'],
            Volatile_PH['PH_3550_M'][kk],
            density[kk],
            thick['Thickness'][kk],
            epsilon['eps_H2OT_3550'][kk],
            )
        H2Om_1635_BP = Beer_Lambert(
            molar_mass['H2O'],
            Volatile_PH['PH_1635_BP'][kk],
            density[kk],
            thick['Thickness'][kk],
            epsilon['eps_H2Om_1635'][kk],
            )
        CO2_1515_BP = Beer_Lambert(
            molar_mass['CO2'],
            Volatile_PH['PH_1515_BP'][kk],
            density[kk],
            thick['Thickness'][kk],
            epsilon['eps_CO2'][kk],
            )
        CO2_1430_BP = Beer_Lambert(
            molar_mass['CO2'],
            Volatile_PH['PH_1430_BP'][kk],
            density[kk],
            thick['Thickness'][kk],
            epsilon['eps_CO2'][kk],
            )
        H2Om_5200_M = Beer_Lambert(
            molar_mass['H2O'],
            Volatile_PH['PH_5200_M'][kk],
            density[kk],
            thick['Thickness'][kk],
            epsilon['eps_H2Om_5200'][kk],
            )
        OH_4500_M = Beer_Lambert(
            molar_mass['H2O'],
            Volatile_PH['PH_4500_M'][kk],
            density[kk],
            thick['Thickness'][kk],
            epsilon['eps_OH_4500'][kk],
            )
        # Multiply by 1e4 to convert CO2 concentrations to ppm
        CO2_1515_BP *= 10000
        CO2_1430_BP *= 10000

        # Calculate volatile species concentration uncertainties
        H2OT_3550_M_STD = Beer_Lambert_Error(
            N, molar_mass['H2O'],
            Volatile_PH['PH_3550_M'][kk],
            Volatile_PH['PH_3550_STD'][kk],
            density[kk],
            density[kk] * 0.025,
            thick['Thickness'][kk],
            thick['Sigma_Thickness'][kk],
            epsilon['eps_H2OT_3550'][kk],
            epsilon['sigma_eps_H2OT_3550'][kk]
            )
        H2Om_1635_BP_STD = Beer_Lambert_Error(
            N, molar_mass['H2O'],
            Volatile_PH['PH_1635_BP'][kk],
            Volatile_PH['PH_1635_STD'][kk],
            density[kk],
            density[kk] * 0.025,
            thick['Thickness'][kk],
            thick['Sigma_Thickness'][kk],
            epsilon['eps_H2Om_1635'][kk],
            epsilon['sigma_eps_H2Om_1635'][kk]
            )
        CO2_1515_BP_STD = Beer_Lambert_Error(
            N, molar_mass['CO2'],
            Volatile_PH['PH_1515_BP'][kk],
            Volatile_PH['PH_1515_STD'][kk],
            density[kk],
            density[kk] * 0.025,
            thick['Thickness'][kk],
            thick['Sigma_Thickness'][kk],
            epsilon['eps_CO2'][kk],
            epsilon['sigma_eps_CO2'][kk]
            )
        CO2_1430_BP_STD = Beer_Lambert_Error(
            N, molar_mass['CO2'],
            Volatile_PH['PH_1430_BP'][kk],
            Volatile_PH['PH_1430_STD'][kk],
            density[kk],
            density[kk] * 0.025,
            thick['Thickness'][kk],
            thick['Sigma_Thickness'][kk],
            epsilon['eps_CO2'][kk],
            epsilon['sigma_eps_CO2'][kk]
            )
        H2Om_5200_M_STD = Beer_Lambert_Error(
            N, molar_mass['H2O'],
            Volatile_PH['PH_5200_M'][kk],
            Volatile_PH['PH_5200_STD'][kk],
            density[kk],
            density[kk] * 0.025,
            thick['Thickness'][kk],
            thick['Sigma_Thickness'][kk],
            epsilon['eps_H2Om_5200'][kk],
            epsilon['sigma_eps_H2Om_5200'][kk]
            )
        OH_4500_M_STD = Beer_Lambert_Error(
            N, molar_mass['H2O'],
            Volatile_PH['PH_4500_M'][kk],
            Volatile_PH['PH_4500_STD'][kk],
            density[kk],
            density[kk] * 0.025,
            thick['Thickness'][kk],
            thick['Sigma_Thickness'][kk],
            epsilon['eps_OH_4500'][kk],
            epsilon['sigma_eps_OH_4500'][kk]
            )
        # Multiply by 1e4 to convert CO2 uncertainties to ppm
        CO2_1515_BP_STD *= 10000
        CO2_1430_BP_STD *= 10000

        # Save volatile concentrations and uncertainties to dataframe
        density_df.loc[kk] = pd.Series({'Density': density[kk]})
        spreadsheet.loc[kk] = pd.Series({
            'H2OT_3550_M': H2OT_3550_M,
            'H2OT_3550_STD': H2OT_3550_M_STD,
            'H2OT_3550_SAT': Volatile_PH['H2OT_3550_SAT?'][kk],
            'H2Om_1635_BP': H2Om_1635_BP,
            'H2Om_1635_STD': H2Om_1635_BP_STD,
            'CO2_1515_BP': CO2_1515_BP,
            'CO2_1515_STD': CO2_1515_BP_STD,
            'CO2_1430_BP': CO2_1430_BP,
            'CO2_1430_STD': CO2_1430_BP_STD,
            'H2Om_5200_M': H2Om_5200_M,
            'H2Om_5200_STD': H2Om_5200_M_STD,
            'OH_4500_M': OH_4500_M,
            'OH_4500_STD': OH_4500_M_STD
            })

    # Loops through for samples to perform two operations depending on
    # saturation. For unsaturated samples, take concentrations and
    # uncertainties from above. For saturated samples, perform the
    # Beer-Lambert calculation again, and have total H2O = H2Om + OH-
    for ll in Volatile_PH.index:
        if Volatile_PH['H2OT_3550_SAT?'][ll] == '-':
            H2OT_3550_M = spreadsheet['H2OT_3550_M'][ll]
            H2Om_1635_BP = spreadsheet['H2Om_1635_BP'][ll]
            CO2_1515_BP = spreadsheet['CO2_1515_BP'][ll]
            CO2_1430_BP = spreadsheet['CO2_1430_BP'][ll]
            H2Om_5200_M = spreadsheet['H2Om_5200_M'][ll]
            OH_4500_M = spreadsheet['OH_4500_M'][ll]

            H2OT_3550_M_STD = spreadsheet['H2OT_3550_STD'][ll]
            H2Om_1635_BP_STD = spreadsheet['H2Om_1635_STD'][ll]
            CO2_1515_BP_STD = spreadsheet['CO2_1515_STD'][ll]
            CO2_1430_BP_STD = spreadsheet['CO2_1430_STD'][ll]
            H2Om_5200_M_STD = spreadsheet['H2Om_5200_STD'][ll]
            OH_4500_M_STD = spreadsheet['OH_4500_STD'][ll]
            density_sat = density_df['Density'][ll]

        elif Volatile_PH['H2OT_3550_SAT?'][ll] == '*':
            Sat_Composition = Composition.copy()
            for m in range(20):
                H2Om_1635_BP = Beer_Lambert(
                    molar_mass['H2O'],
                    Volatile_PH['PH_1635_BP'][ll],
                    density[ll],
                    thick['Thickness'][ll],
                    epsilon['eps_H2Om_1635'][ll],
                    )
                OH_4500_M = Beer_Lambert(
                    molar_mass['H2O'],
                    Volatile_PH['PH_4500_M'][ll],
                    density[ll],
                    thick['Thickness'][ll],
                    epsilon['eps_OH_4500'][ll],
                    )
                Sat_Composition.loc[ll, 'H2O'] = H2Om_1635_BP + OH_4500_M
                mol_sat, density_sat = Density_Calculation(Sat_Composition,
                                                           T, P)
            density_sat = density_sat[ll]

            H2OT_3550_M = Beer_Lambert(
                molar_mass['H2O'],
                Volatile_PH['PH_3550_M'][ll],
                density_sat,
                thick['Thickness'][ll],
                epsilon['eps_H2OT_3550'][ll],
                )
            H2Om_1635_BP = Beer_Lambert(
                molar_mass['H2O'],
                Volatile_PH['PH_1635_BP'][ll],
                density_sat,
                thick['Thickness'][ll],
                epsilon['eps_H2Om_1635'][ll],
                )
            CO2_1515_BP = Beer_Lambert(
                molar_mass['CO2'],
                Volatile_PH['PH_1515_BP'][ll],
                density_sat,
                thick['Thickness'][ll],
                epsilon['eps_CO2'][ll],
                )
            CO2_1430_BP = Beer_Lambert(
                molar_mass['CO2'],
                Volatile_PH['PH_1430_BP'][ll],
                density_sat,
                thick['Thickness'][ll],
                epsilon['eps_CO2'][ll],
                )
            H2Om_5200_M = Beer_Lambert(
                molar_mass['H2O'],
                Volatile_PH['PH_5200_M'][ll],
                density_sat,
                thick['Thickness'][ll],
                epsilon['eps_H2Om_5200'][ll],
                )
            OH_4500_M = Beer_Lambert(
                molar_mass['H2O'],
                Volatile_PH['PH_4500_M'][ll],
                density_sat,
                thick['Thickness'][ll],
                epsilon['eps_OH_4500'][ll],
                )
            CO2_1515_BP *= 10000
            CO2_1430_BP *= 10000

            H2OT_3550_M_STD = Beer_Lambert_Error(
                N, molar_mass['H2O'],
                Volatile_PH['PH_3550_M'][ll],
                Volatile_PH['PH_3550_STD'][ll],
                density_sat,
                density_sat * 0.025,
                thick['Thickness'][ll],
                thick['Sigma_Thickness'][ll],
                epsilon['eps_H2OT_3550'][ll],
                epsilon['sigma_eps_H2OT_3550'][ll]
            )
            H2Om_1635_BP_STD = Beer_Lambert_Error(
                N, molar_mass['H2O'],
                Volatile_PH['PH_1635_BP'][ll],
                Volatile_PH['PH_1635_STD'][ll],
                density_sat,
                density_sat * 0.025,
                thick['Thickness'][ll],
                thick['Sigma_Thickness'][ll],
                epsilon['eps_H2Om_1635'][ll],
                epsilon['sigma_eps_H2Om_1635'][ll]
            )
            CO2_1515_BP_STD = Beer_Lambert_Error(
                N, molar_mass['CO2'],
                Volatile_PH['PH_1515_BP'][ll],
                Volatile_PH['PH_1515_STD'][ll],
                density_sat,
                density_sat * 0.025,
                thick['Thickness'][ll],
                thick['Sigma_Thickness'][ll],
                epsilon['eps_CO2'][ll],
                epsilon['sigma_eps_CO2'][ll]
            )
            CO2_1430_BP_STD = Beer_Lambert_Error(
                N, molar_mass['CO2'],
                Volatile_PH['PH_1430_BP'][ll],
                Volatile_PH['PH_1430_STD'][ll],
                density_sat,
                density_sat * 0.025,
                thick['Thickness'][ll],
                thick['Sigma_Thickness'][ll],
                epsilon['eps_CO2'][ll],
                epsilon['sigma_eps_CO2'][ll]
            )
            H2Om_5200_M_STD = Beer_Lambert_Error(
                N, molar_mass['H2O'],
                Volatile_PH['PH_5200_M'][ll],
                Volatile_PH['PH_5200_STD'][ll],
                density_sat,
                density_sat * 0.025,
                thick['Thickness'][ll],
                thick['Sigma_Thickness'][ll],
                epsilon['eps_H2Om_5200'][ll],
                epsilon['sigma_eps_H2Om_5200'][ll]
            )
            OH_4500_M_STD = Beer_Lambert_Error(
                N, molar_mass['H2O'],
                Volatile_PH['PH_4500_M'][ll],
                Volatile_PH['PH_4500_STD'][ll],
                density_sat,
                density_sat * 0.025,
                thick['Thickness'][ll],
                thick['Sigma_Thickness'][ll],
                epsilon['eps_OH_4500'][ll],
                epsilon['sigma_eps_OH_4500'][ll]
            )
            CO2_1515_BP_STD *= 10000
            CO2_1430_BP_STD *= 10000

        density_sat_df.loc[ll] = pd.Series({'Density_Sat': density_sat})
        spreadsheet_sat.loc[ll] = pd.Series({
            'H2OT_3550_M': H2OT_3550_M,
            'H2OT_3550_SAT': Volatile_PH['H2OT_3550_SAT?'][ll],
            'H2OT_3550_STD': H2OT_3550_M_STD,
            'H2Om_1635_BP': H2Om_1635_BP,
            'H2Om_1635_STD': H2Om_1635_BP_STD,
            'CO2_1515_BP': CO2_1515_BP,
            'CO2_1515_STD': CO2_1515_BP_STD,
            'CO2_1430_BP': CO2_1430_BP,
            'CO2_1430_STD': CO2_1430_BP_STD,
            'H2Om_5200_M': H2Om_5200_M,
            'H2Om_5200_STD': H2Om_5200_M_STD,
            'OH_4500_M': OH_4500_M,
            'OH_4500_STD': OH_4500_M_STD
            })
        s2nerror.loc[ll] = pd.Series({
            'PH_5200_S2N': Volatile_PH['S2N_P5200'][ll],
            'PH_4500_S2N': Volatile_PH['S2N_P4500'][ll],
            'ERR_5200': Volatile_PH['ERR_5200'][ll],
            'ERR_4500': Volatile_PH['ERR_4500'][ll]
            })

    # Create final spreadsheet
    spreadsheet_f = pd.concat([spreadsheet_sat, s2nerror], axis=1)
    density_epsilon = pd.concat([density_df, density_sat_df, epsilon],
                                axis=1)

    # Output different values depending on saturation.
    for m in spreadsheet.index:
        if spreadsheet['H2OT_3550_SAT'][m] == '*':
            H2O_mean = (spreadsheet['H2Om_1635_BP'][m] +
                        spreadsheet['OH_4500_M'][m])
            H2O_std = ((spreadsheet['H2Om_1635_STD'][m]**2) +
                       (spreadsheet['OH_4500_STD'][m]**2))**(1/2) / 2

        elif spreadsheet['H2OT_3550_SAT'][m] == '-':
            H2O_mean = spreadsheet['H2OT_3550_M'][m]
            H2O_std = spreadsheet['H2OT_3550_STD'][m]
        mean_vol.loc[m] = pd.Series({
            'H2OT_MEAN': H2O_mean, 'H2OT_STD': H2O_std})
    mean_vol['CO2_MEAN'] = (spreadsheet['CO2_1515_BP'] +
                            spreadsheet['CO2_1430_BP']) / 2
    mean_vol['CO2_STD'] = ((spreadsheet['CO2_1515_STD']**2) +
                           (spreadsheet['CO2_1430_STD']**2))**(1/2) / 2

    spreadsheet_f['H2OT_MEAN'] = mean_vol['H2OT_MEAN']
    spreadsheet_f['H2OT_STD'] = mean_vol['H2OT_STD']
    spreadsheet_f['CO2_MEAN'] = mean_vol['CO2_MEAN']
    spreadsheet_f['CO2_STD'] = mean_vol['CO2_STD']

    return density_epsilon, spreadsheet_f

# %% Plotting Functions


def trace(posterior, title, zchain=None, pnames=None, thinning=50,
          burnin=0, fignum=1000, savefile=None, fmt=".", ms=2.5, fs=12):

    """
    Plot parameter trace MCMC sampling.

    Parameters:
        posterior (2D np.ndarray): MCMC posterior sampling with dimension:
            [nsamples, npars].
        zchain (1D np.ndarray): Chain index for each posterior sample.
        pnames (str): Label names for parameters.
        thinning (int): Thinning factor for plotting (plot every thinning-th
            value).
        burnin (int): Thinned burn-in number of iteration (only used when
            zchain is not None).
        fignum (int): The figure number.
        savefile (bool): Name of file to save the plot if not none
        fmt (string): The format string for the line and marker.
        ms (float): Marker size.
        fs (float): Fontsize of texts.

    Returns:
        axes (1D list of matplotlib.axes.Axes): List of axes containing
            the marginal posterior distributions.
    """

    # Get indices for samples considered in final analysis:
    if zchain is not None:
        nchains = np.amax(zchain) + 1
        good = np.zeros(len(zchain), bool)
        for c in range(nchains):
            good[np.where(zchain == c)[0][burnin:]] = True
        # Values accepted for posterior stats:
        posterior = posterior[good]
        zchain = zchain[good]
        # Sort the posterior by chain:
        zsort = np.lexsort([zchain])
        posterior = posterior[zsort]
        zchain = zchain[zsort]
        # Get location for chains separations:
        xsep = np.where(np.ediff1d(zchain[0::thinning]))[0]

    # Get number of parameters and length of chain:
    nsamples, npars = np.shape(posterior)
    # Number of samples (thinned):
    xmax = len(posterior[0::thinning])

    # Set default parameter names:
    if pnames is None:
        pnames = mu.default_parnames(npars)

    npanels = 16  # Max number of panels per page
    npages = int(1 + (npars-1)/npanels)

    # Make the trace plot:
    axes = []
    ipar = 0
    for page in range(npages):
        fig = plt.figure(fignum+page, figsize=(8.5, 11.0))
        plt.clf()
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05,
                            top=0.95, hspace=0.15)
        while ipar < npars:
            ax = plt.subplot(npanels, 1, ipar % npanels+1)
            axes.append(ax)
            ax.plot(posterior[0::thinning, ipar], fmt, ms=ms,
                    c='#46A4F5', rasterized=True)
            yran = ax.get_ylim()
            if zchain is not None:
                ax.vlines(xsep, yran[0], yran[1], "0.5")
            # Y-axis adjustments:
            ax.set_ylim(yran)
            ax.locator_params(axis='y', nbins=5, tight=True)
            ax.tick_params(labelsize=fs-1, direction='in',
                           top=True, right=True)
            ax.set_ylabel(pnames[ipar], size=fs, multialignment='center')
            # X-axis adjustments:
            ax.set_xlim(0, xmax)
            ax.get_xaxis().set_visible(False)
            ipar += 1
            if ipar % npanels == 0:
                break
        ax.set_xlabel('Thinned MCMC Sample', size=fs)
        ax.get_xaxis().set_visible(True)

        if savefile is not None:
            if npages > 1:
                sf = os.path.splitext(savefile)
                try:
                    bbox = fig.get_tightbbox(fig._cachedRenderer).padded(0.1)
                    bbox_points = bbox.get_points()
                    bbox_points[:, 0] = 0.0, 8.5
                    bbox.set_points(bbox_points)
                except Exception:
                    ylow = 9.479-0.862*np.amin([npanels-1,
                                                npars-npanels*page-1])
                    bbox = mpl.transforms.Bbox([[0.0, ylow], [8.5, 11]])

                fig.savefig(f"{sf[0]}_page{page:02d}{sf[1]}",
                            bbox_inches=bbox)
            else:
                fig.suptitle(title, fontsize=fs+1)
                plt.ioff()
                fig.savefig(savefile, bbox_inches='tight')

    return axes


def modelfit(data, uncert, indparams, model, title, nbins=75,
             fignum=1400, savefile=None):

    """
    Plot the binned dataset with given uncertainties and model curves as
    a function of indparams. In a lower panel, plot the residuals between
    the data and model.

    Parameters:
        data (1D np.ndarray): Input data set.
        uncert (1D np.ndarray): One-sigma uncertainties of the data points.
        indparams (1D np.ndarray): Independent variable (X axis) of the
            data points.
        model (1D np.ndarray): Model of data.
        nbins (int): Number of bins in the output plot.
        fignum (int): Figure number.
        savefile (bool): Name of file to save the plot, if not none.

    Returns:
        ax (matplotlib.axes.Axes): Axes instance containing the marginal
        posterior distributions.
    """

    # Bin down array:
    binsize = int((np.size(data)-1) / nbins + 1)
    binindp = ms.bin_array(indparams, binsize)
    binmodel = ms.bin_array(model, binsize)
    bindata, binuncert = ms.bin_array(data, binsize, uncert)
    fs = 12

    plt.figure(fignum, figsize=(8, 6))
    plt.clf()

    # Residuals:
    rax = plt.axes([0.15, 0.1, 0.8, 0.2])
    rax.errorbar(binindp, bindata-binmodel, binuncert, fmt='ko', ms=4)
    rax.plot([indparams[0], indparams[-1]], [0, 0], 'k:', lw=1.5)
    rax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
    rax.set_xlabel("Wavenumber (cm^-1)", fontsize=fs)
    rax.set_ylabel('Residual', fontsize=fs)
    rax.invert_xaxis()

    # Data and Model:
    ax = plt.axes([0.15, 0.35, 0.8, 0.575])
    ax.errorbar(binindp, bindata, binuncert, fmt='ko', ms=4,
                label='Binned data')
    ax.plot(indparams, data, 'tab:blue', lw=3, label='Data')
    ax.plot(indparams, model, 'tab:purple', linestyle='-.', lw=3,
            label='Best Fit')
    ax.set_xticklabels([])
    ax.invert_xaxis()
    ax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
    ax.set_ylabel('Absorbance', fontsize=fs)
    ax.legend(loc='best')

    if savefile is not None:
        plt.suptitle(title, fontsize=fs+1)
        plt.ioff()
        plt.savefig(savefile, bbox_inches='tight')

    return ax, rax
