# %% -*- coding: utf-8 -*-
""" Created on June 12, 2021 // @author: Sarah Shi and Henry Towbin """

# %% Import packages

import os
import copy
import warnings

import mc3
import numpy as np
import pandas as pd

import scipy.signal as signal
import scipy.interpolate as interpolate
from scipy.linalg import solveh_banded

from pykrige import OrdinaryKriging

import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    mpl.use('Agg')
from matplotlib import pyplot as plt
import scipy.interpolate as si
import mc3.utils as mu
import mc3.stats as ms

__all__ = ['trace', 'histogram', 'pairwise', 'rms', 'modelfit', 'subplotter', 'themes',]
themes = {
    'blue':{'edgecolor':'navy','facecolor':'royalblue','color':'navy'},
    'red': {'edgecolor':'crimson','facecolor':'orangered','color':'darkred'},
    'black':{'edgecolor':'0.3','facecolor':'0.3','color':'black'},
    'green':{'edgecolor':'forestgreen','facecolor':'limegreen','color':'darkgreen'},
    'orange':{'edgecolor':'darkorange','facecolor':'gold','color':'darkgoldenrod'},}

__version__ = "0.1.0"

# %%


def Load_SampleCSV(paths, H2O_wn_high, H2O_wn_low): 

    """The Load_SampleCSV function takes the inputs of the path to a directory with all sample CSVs, 
    wavenumber high, wavenumber low values. The function outputs a dictionary of each sample's associated 
    wavenumbers and absorbances."""

    CO2_dfs = []
    H2O_dfs = []
    files = []

    for path in paths:
        head_tail = os.path.split(path)
        file = head_tail[1][0:-4]

        df = pd.read_csv(path, names= ['Wavenumber', 'Absorbance'])
        df.set_index('Wavenumber', inplace = True)
        H2O_spec = df.loc[H2O_wn_low:H2O_wn_high]
        H2O_dfs.append(H2O_spec)
        files.append(file)

    H2O_zipobj = zip(files, H2O_dfs)
    H2O_dfs_dict = dict(H2O_zipobj)

    return files, H2O_dfs_dict


def Load_PCA(PCA_Path):

    """The Load_PCA function takes the input of a path to a CSV of predetermined PCA components. 
    The function returns a dataframe with the PCA components."""

    wn_high = 2200
    wn_low = 1275

    PCA_DF = pd.read_csv(PCA_Path, index_col = "Wavenumber")
    PCA_DF = PCA_DF[wn_low:wn_high]
    PCA_matrix = np.matrix(PCA_DF.to_numpy())

    return PCA_matrix


def Load_Wavenumber(Wavenumber_Path):

    """The Load_PCA function takes the input of a path to a CSV of the wavenumbers associated with PCA components. 
    The function returns a dataframe with the wavenumbers."""

    wn_high = 2200
    wn_low = 1275

    Wavenumber_DF = pd.read_csv(Wavenumber_Path, index_col = "Wavenumber")
    Wavenumber_DF = Wavenumber_DF[wn_low:wn_high]
    Wavenumber = np.array(Wavenumber_DF.index)

    return Wavenumber


def Load_ChemistryThickness(ChemistryThickness_Path):

    """The Load_ChemistryThickness function takes the input of a path to a CSV with MI chemistry or thickness. 
    The function returns dataframes with these data."""

    ChemistryThickness = pd.read_csv(ChemistryThickness_Path)
    ChemistryThickness.set_index('Sample', inplace = True)

    Chemistry = ChemistryThickness.loc[:, ['SiO2', 'TiO2', 'Al2O3', 'Fe2O3', 'FeO', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O', 'P2O5']]
    Thickness = ChemistryThickness.loc[:, ['Thickness']]

    return Chemistry, Thickness

def Gauss(x, mu, sd, A=1):

    """The Gauss function takes the inputs of the wavenumbers of interest, center of peak, 
    standard deviation (or width of peak), and amplitude. The function outputs a Gaussian fit 
    for the CO3^{2-} doublet peaks at 1515 and 1430 cm^-1 peaks."""

    G = A * np.exp(-(x - mu) ** 2 / (2 * sd ** 2))

    return G


def Linear(x, m, b):

    """The Linear function takes the inputs of wavenumbers of interest, tilt, and offset for 
    adjusting the model data. The function returns a linear offset taking the form of y = mx+b."""

    b = np.ones_like(x) * b
    m = np.arange(0, max(x.shape)) * m 
    tilt_offset = m + b 

    return tilt_offset


def Carbonate(P, x, PCAmatrix, Peak_1635_PCAmatrix, Nvectors = 5): 

    """The Carbonate function takes in the inputs of fitting parameters P, wavenumbers x, PCA matrix, 
    number of PCA vectors of interest. The function calculates the molecular H2O_{1635} peak, 
    CO3^{2-} Gaussian peaks, linear offset, and baseline. The function then returns the model data."""

    PCA_Weights = np.array([P[0:Nvectors]])
    Peak_Weights = np.array([P[-5:-2]])

    peak_G1430, std_G1430, G1430_amplitude, peak_G1515, std_G1515, G1515_amplitude = P[Nvectors:-5]
    m, b = P[-2:None]

    Peak_1635 = Peak_Weights * Peak_1635_PCAmatrix.T
    G1515 = Gauss(x, peak_G1515, std_G1515, A=G1515_amplitude)
    G1430 = Gauss(x, peak_G1430, std_G1430, A=G1430_amplitude)

    linear_offset = Linear(x, m, b) 

    baseline = PCA_Weights * PCAmatrix.T
    model_data = baseline + linear_offset + Peak_1635 + G1515 + G1430
    model_data = np.array(model_data)[0,:]

    return model_data


def als_baseline(intensities,
    asymmetry_param=0.05, smoothness_param=5e5,
    max_iters=10, conv_thresh=1e-5, verbose=False):
    """ Computes the asymmetric least squares baseline.
    http://www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
    Smoothness_param: Relative importance of smoothness of the predicted response.
    Asymmetry_param (p): if y > z, w = p, otherwise w = 1-p.
    Setting p=1 is effectively a hinge loss. """

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
            upper_bands[i, -i - 1 :] = ds[::-1][: i + 1]
        self.upper_bands = upper_bands

    def smooth(self, w):
        foo = self.upper_bands.copy()
        foo[-1] += w  # last row is the diagonal
        return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)


def NearIR_Process(data, H2O_wn_low, H2O_wn_high, peak): 

    """The NearIR_Process function inputs the dictionary data, the Near-IR H2O (5200 peak) or OH (4500 peak) wavenumbers of interest, 
    and the H2O or OH peak of interest and returns a dataframe of the absorbance data in the region of interest, median filtered data, 
    baseline subtracted absorbance, the kriged data output, the peak height, and the signal to noise ratio. 
    The function median filters the peak absorbance (given the noise inherent to these Near-IR peaks for MI), 
    fits an asymmetric least squares determined baseline to the peak, and subtracts the baseline to determine the 
    peak absorbance. The baseline-subtracted peak is then kriged, to further reduce noise and to obtain peak height. 
    The signal to noise ratio is then determined and if the ratio is high, a warning is outputted and the 
    user can consider the usefulness of these peaks. This function is used three times with slightly different 
    H2O wavenumber ranges, so that uncertainty can be assessed."""

    data_H2O = data[H2O_wn_low:H2O_wn_high]
    data_output = pd.DataFrame(columns = ['Absorbance', 'Absorbance_Hat', 'BL_NIR_H2O', 'Subtracted_Peak'], index = data_H2O.index)
    data_output['Absorbance'] = data_H2O
    data_output['Absorbance_Hat'] = signal.medfilt(data_H2O['Absorbance'], 5)
    data_output['BL_NIR_H2O'] = als_baseline(data_output['Absorbance_Hat'], asymmetry_param=0.001, smoothness_param=1e9, max_iters=10, conv_thresh=1e-5)
    data_output['Subtracted_Peak'] = data_output['Absorbance_Hat'] - data_output['BL_NIR_H2O']
    
    krige_wn_range = np.linspace(H2O_wn_low-5, H2O_wn_high+5, H2O_wn_high-H2O_wn_low+11)
    krige_peak = OrdinaryKriging(data_H2O.index, np.zeros(data_output['Subtracted_Peak'].shape), data_output['Subtracted_Peak'], variogram_model = 'gaussian')
    krige_abs, krige_std = krige_peak.execute("grid", krige_wn_range, np.array([0.0]))
    krige_output = pd.DataFrame(columns = ['Absorbance', 'STD'], index = krige_wn_range)
    krige_output['Absorbance'] = np.asarray(np.squeeze(krige_abs))
    krige_output['STD'] = np.asarray(np.squeeze(krige_std))

    if peak == 'OH': # 4500 peak
        PR_4500_low, PR_4500_high = 4400, 4600
        PH_max = np.max(data_output['Subtracted_Peak'][PR_4500_low:PR_4500_high])
        PH = np.max(data_output['Subtracted_Peak'][PR_4500_low:PR_4500_high])
        PH_krige = np.max(krige_output['Absorbance'][PR_4500_low:PR_4500_high]) - np.min(krige_output['Absorbance'])
        PH_krige_index = int(data_output['Subtracted_Peak'][data_output['Subtracted_Peak'] == PH_max].index.to_numpy())
        PH_std = np.std(data_output['Subtracted_Peak'][PH_krige_index-50:PH_krige_index+50])
        STN = PH_krige / PH_std
    elif peak == 'H2O': # 5200 peak
        PR_5200_low, PR_5200_high = 5100, 5300
        PH_max = np.max(data_output['Subtracted_Peak'][PR_5200_low:PR_5200_high])
        PH = np.max(data_output['Subtracted_Peak'][PR_5200_low:PR_5200_high])
        PH_krige = np.max(krige_output['Absorbance'][PR_5200_low:PR_5200_high]) - np.min(krige_output['Absorbance'])
        PH_krige_index = int(data_output['Subtracted_Peak'][data_output['Subtracted_Peak'] == PH_max].index.to_numpy())
        PH_std = np.std(data_output['Subtracted_Peak'][PH_krige_index-50:PH_krige_index+50])
        STN = PH_krige / PH_std

    return data_output, krige_output, PH_krige, STN


def MidIR_Process(data, H2O_wn_low, H2O_wn_high):

    """The MidIR_Process function with the inputs of dictionary data and the Mid-IR total H2O wavenumbers of interest and returns 
    a dataframe of the absorbance data in the region of interest, the kriged data output, and the peak height. 
    This function is used three times with slightly different H2O wavenumber ranges, so that uncertainty in peak height can be assessed."""

    data_H2O3550 = data[H2O_wn_low:H2O_wn_high]
    data_output = pd.DataFrame(columns = ['Absorbance', 'BL_MIR_3550', 'Subtracted_Peak', 'Subtracted_Peak_Hat'], index = data_H2O3550.index)
    data_output['Absorbance'] = data_H2O3550['Absorbance']
    data_output['BL_MIR_3550'] = als_baseline(data_H2O3550['Absorbance'], asymmetry_param=0.0010, smoothness_param=1e11, max_iters=20, conv_thresh=1e-7)
    data_output['Subtracted_Peak'] = data_H2O3550['Absorbance'] - data_output['BL_MIR_3550']
    data_output['Subtracted_Peak_Hat'] = signal.medfilt(data_output['Subtracted_Peak'], 21)

    peak_wn_low, peak_wn_high = 3300, 3600
    plot_output = data_output[peak_wn_low:peak_wn_high]
    plotindex = np.argmax(plot_output['Absorbance'].index.to_numpy() > 3400)
    PH_3550 = np.max(plot_output['Subtracted_Peak_Hat'])

    return data_output, plot_output, PH_3550, plotindex


# %% 

# %% 


# %% Plotting functions 


def trace(posterior, title, zchain=None, pnames=None, thinning=25,
    burnin=0, fignum=1000, savefile=None, fmt=".", ms=2.5, fs=11):
    """
    Plot parameter trace MCMC sampling.
    Parameters
    ----------
    posterior: 2D float ndarray
        An MCMC posterior sampling with dimension: [nsamples, npars].
    zchain: 1D integer ndarray
        the chain index for each posterior sample.
    pnames: Iterable (strings)
        Label names for parameters.
    thinning: Integer
        Thinning factor for plotting (plot every thinning-th value).
    burnin: Integer
        Thinned burn-in number of iteration (only used when zchain is not None).
    fignum: Integer
        The figure number.
    savefile: Boolean
        If not None, name of file to save the plot.
    fmt: String
        The format string for the line and marker.
    ms: Float
        Marker size.
    fs: Float
        Fontsize of texts.
    Returns
    -------
    axes: 1D list of matplotlib.axes.Axes
        List of axes containing the marginal posterior distributions.
    """
    # Get indices for samples considered in final analysis:
    if zchain is not None:
        nchains = np.amax(zchain) + 1
        good = np.zeros(len(zchain), bool)
        for c in range(nchains):
            good[np.where(zchain == c)[0][burnin:]] = True
        # Values accepted for posterior stats:
        posterior = posterior[good]
        zchain    = zchain   [good]
        # Sort the posterior by chain:
        zsort = np.lexsort([zchain])
        posterior = posterior[zsort]
        zchain    = zchain   [zsort]
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
        fig = plt.figure(fignum+page, figsize=(8.5,11.0))
        plt.clf()
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.05, top=0.95, hspace=0.15)
        while ipar < npars:
            ax = plt.subplot(npanels, 1, ipar%npanels+1)
            axes.append(ax)
            ax.plot(posterior[0::thinning,ipar], fmt, ms=ms, rasterized = True)
            yran = ax.get_ylim()
            if zchain is not None:
                ax.vlines(xsep, yran[0], yran[1], "0.5")
            # Y-axis adjustments:
            ax.set_ylim(yran)
            ax.locator_params(axis='y', nbins=5, tight=True)
            ax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
            ax.set_ylabel(pnames[ipar], size=fs, multialignment='center')
            # X-axis adjustments:
            ax.set_xlim(0, xmax)
            ax.get_xaxis().set_visible(False)
            ipar += 1
            if ipar%npanels == 0:
                break
        ax.set_xlabel('MCMC sample', size=fs)
        ax.get_xaxis().set_visible(True)

        if savefile is not None:
            if npages > 1:
                sf = os.path.splitext(savefile)
                try:
                    bbox = fig.get_tightbbox(fig._cachedRenderer).padded(0.1)
                    bbox_points = bbox.get_points()
                    bbox_points[:,0] = 0.0, 8.5
                    bbox.set_points(bbox_points)
                except:  # May fail for ssh connection without X display
                    ylow = 9.479-0.862*np.amin([npanels-1,npars-npanels*page-1])
                    bbox = mpl.transforms.Bbox([[0.0, ylow], [8.5, 11]])

                fig.savefig(f"{sf[0]}_page{page:02d}{sf[1]}", bbox_inches=bbox, backend='pgf')
            else:
                fig.suptitle(title)
                plt.ioff()
                fig.savefig(savefile, bbox_inches='tight', backend='pgf') # dpi = 100)

    return axes

def histogram(posterior, title, pnames=None, thinning=1, fignum=1100,
    savefile=None, bestp=None, quantile=None, pdf=None,
    xpdf=None, ranges=None, axes=None, lw=2.0, fs=11,
    theme='blue', yscale=False, orientation='vertical'):
    """
    Plot parameter marginal posterior distributions
    Parameters
    ----------
    posterior: 1D or 2D float ndarray
        An MCMC posterior sampling with dimension [nsamples] or [nsamples, nparameters].
    pnames: Iterable (strings)
        Label names for parameters.
    thinning: Integer
        Thinning factor for plotting (plot every thinning-th value).
    fignum: Integer
        The figure number.
    savefile: Boolean
        If not None, name of file to save the plot.
    bestp: 1D float ndarray
        If not None, plot the best-fitting values for each parameter given by bestp.
    quantile: Float
        If not None, plot the quantile- highest posterior density region of the distribution. 
        For example, set quantile=0.68 for a 68% HPD.
    pdf: 1D float ndarray or list of ndarrays
        A smoothed PDF of the distribution for each parameter.
    xpdf: 1D float ndarray or list of ndarrays
        The X coordinates of the PDFs.
    ranges: List of 2-element arrays
        List with custom (lower,upper) x-ranges for each parameter.
        Leave None for default, e.g., ranges=[(1.0,2.0), None, (0, 1000)].
    axes: List of matplotlib.axes
        If not None, plot histograms in the currently existing axes.
    lw: Float
        Linewidth of the histogram contour.
    fs: Float
        Font size for texts.
    theme: String or dict
        The histograms' color theme.  If string must be one of mc3.plots.themes.
        If dict, must define edgecolor, facecolor, color (with valid matplotlib
        colors) for the histogram edge and face colors, and the best-fit color, respectively.
    yscale: Bool
        If True, set an absolute Y-axis scaling among all posteriors. False is default.
    orientation: String
        Orientation of the histograms.  If 'horizontal', the bottom of the
        histogram will be at the left (might require some adjusting of the
        axes location, e.g., a plt.tight_layout() call).
    Returns
    -------
    axes: 1D list of matplotlib.axes.Axes
        List of axes containing the marginal posterior distributions.
    """
    if isinstance(theme, str):
        theme = themes[theme]

    if np.ndim(posterior) == 1:
        posterior = np.expand_dims(posterior, axis=1)
    nsamples, npars = np.shape(posterior)

    if pdf is None:
        pdf, xpdf = [None]*npars, [None]*npars
    if not isinstance(pdf, list):  # Put single arrays into list
        pdf, xpdf  = [pdf], [xpdf]
    # Histogram keywords:
    if int(np.__version__.split('.')[1]) >= 15:
        hkw = {'density':not yscale}
    else:
        hkw = {'normed':not yscale}

    # Set default parameter names:
    if pnames is None:
        pnames = mu.default_parnames(npars)

    # Xranges:
    if ranges is None:
        ranges = np.repeat(None, npars)

    # Set number of rows:
    nrows, ncolumns, npanels = 4, 4, 16
    npages = int(1 + (npars-1)/npanels)

    ylabel = "$N$ samples" if yscale else "Posterior density"
    if axes is None:
        figs, axes = [], []
        for j in range(npages):
            fig = plt.figure(fignum+j, figsize=(8.5, 11.0))
            figs.append(fig)
            fig.clf()
            fig.subplots_adjust(left=0.1, right=0.97, bottom=0.08, 
                top=0.95, hspace=0.5, wspace=0.1)
            for ipar in range(np.amin([npanels, npars-npanels*j])):
                ax = fig.add_subplot(nrows, ncolumns, ipar+1)
                axes.append(ax)
                yax = ax.yaxis if orientation == 'vertical' else ax.xaxis
                if ipar%ncolumns == 0 or orientation == 'horizontal':
                    yax.set_label_text(ylabel, fontsize=fs)
                if ipar%ncolumns != 0 or yscale is False:
                    yax.set_ticklabels([])
    else:
        npages = 1  # Assume there's only one page
        figs = [axes[0].get_figure()]
        for ax in axes:
            ax.set_yticklabels([])

    maxylim = 0
    for ipar in range(npars):
        ax = axes[ipar]
        if orientation == 'vertical':
            xax = ax.xaxis
            get_xlim, set_xlim = ax.get_xlim, ax.set_xlim
            get_ylim, set_ylim = ax.get_ylim, ax.set_ylim
            fill_between = ax.fill_between
            axline = ax.axvline
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
        else:
            xax = ax.yaxis
            get_xlim, set_xlim = ax.get_ylim, ax.set_ylim
            get_ylim, set_ylim = ax.get_xlim, ax.set_xlim
            fill_between = ax.fill_between
            axline = ax.axhline

        ax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
        xax.set_label_text(pnames[ipar], fontsize=fs)
        vals, bins, h = ax.hist(posterior[0::thinning,ipar],
            bins=25, histtype='step', lw=lw, zorder=0,
            range=ranges[ipar], ec=theme['edgecolor'],
            orientation=orientation, **hkw)
        # Plot HPD region if needed:
        if quantile is None:
            ax.hist(posterior[0::thinning,ipar],
                bins=25, lw=lw, zorder=-2, alpha=0.4,
                range=ranges[ipar], facecolor=theme['facecolor'], ec='none',
                orientation=orientation, **hkw)
        if quantile is not None:
            PDF, Xpdf, HPDmin = ms.cred_region(
                posterior[:,ipar], quantile, pdf[ipar], xpdf[ipar])
            vals = np.r_[0, vals, 0]
            bins = np.r_[bins[0] - (bins[1]-bins[0]), bins]
            # Interpolate xpdf into the histogram:
            f = si.interp1d(bins+0.5*(bins[1]-bins[0]), vals, kind='nearest')
            # Plot the HPD region as shaded areas:
            if ranges[ipar] is not None:
                xran = np.argwhere((Xpdf>ranges[ipar][0]) & (Xpdf<ranges[ipar][1]))
                Xpdf = Xpdf[np.amin(xran):np.amax(xran)]
                PDF  = PDF [np.amin(xran):np.amax(xran)]
            fill_between(
                Xpdf, 0, f(Xpdf), where=PDF>=HPDmin,
                facecolor=theme['facecolor'], edgecolor='none',
                interpolate=True, zorder=-2, alpha=0.4)
        if bestp is not None:
            axline(bestp[ipar], dashes=(7,4), lw=1.25, color=theme['color'])
        maxylim = np.amax((maxylim, get_ylim()[1]))
        if ranges[ipar] is not None:
            set_xlim(np.clip(get_xlim(), ranges[ipar][0], ranges[ipar][1]))

    if yscale:
        for ax in axes:
            set_ylim = ax.get_ylim if orientation == 'vertical' else ax.set_xlim
            set_ylim(0, maxylim)
    
    if savefile is not None:
        for page, fig in enumerate(figs):
            if npages > 1:
                sf = os.path.splitext(savefile)
                fig.savefig(f"{sf[0]}_page{page:02d}{sf[1]}", bbox_inches='tight', backend='pgf')
            else:
                fig.suptitle(title)
                plt.ioff()
                fig.savefig(savefile, bbox_inches='tight', backend='pgf')
    
    return axes

def pairwise(posterior, title, pnames=None, thinning=25, fignum=1200,
    savefile=None, bestp=None, nbins=25, nlevels=20,
    absolute_dens=False, ranges=None, fs=11, rect=None, margin=0.01):
    """
    Plot parameter pairwise posterior distributions.
    Parameters
    ----------
    posterior: 2D ndarray
        An MCMC posterior sampling with dimension: [nsamples, nparameters].
    pnames: Iterable (strings)
        Label names for parameters.
    thinning: Integer
        Thinning factor for plotting (plot every thinning-th value).
    fignum: Integer
        The figure number.
    savefile: Boolean
        If not None, name of file to save the plot.
    bestp: 1D float ndarray
        If not None, plot the best-fitting values for each parameter given by bestp.
    nbins: Integer
        The number of grid bins for the 2D histograms.
    nlevels: Integer
        The number of contour color levels.
    ranges: List of 2-element arrays
        List with custom (lower,upper) x-ranges for each parameter.
        Leave None for default, e.g., ranges=[(1.0,2.0), None, (0, 1000)].
    fs: Float
        Fontsize of texts.
    rect: 1D list/ndarray
        If not None, plot the pairwise plots in current figure, within the
        ranges defined by rect (xleft, ybottom, xright, ytop).
    margin: Float
        Margins between panels (when rect is not None).
    Returns
    -------
    axes: 2D ndarray of matplotlib.axes.Axes
        Array of axes containing the marginal posterior distributions.
    cb: matplotlib.axes.Axes
        The colorbar axes.
    Notes
    -----
    rect delimits the boundaries of the panels. The labels and
    ticklabels will appear outside rect, so the user needs to leave
    some wiggle room for them.
    """
    # Get number of parameters and length of chain:
    nsamples, npars = np.shape(posterior)

    # Don't plot if there are no pairs:
    if npars == 1:
        return None, None

    if ranges is None:
        ranges = np.repeat(None, npars)
    else: # Set default ranges if necessary:
        for i in range(npars):
            if ranges[i] is None:
                ranges[i] = (np.nanmin(posterior[0::thinning,i]), np.nanmax(posterior[0::thinning,i]))

    # Set default parameter names:
    if pnames is None:
        pnames = mu.default_parnames(npars)

    # Set palette color:
    palette = copy.copy(plt.cm.viridis_r)
    palette.set_under(color='w')
    palette.set_bad(color='w')

    # Gather 2D histograms:
    hist = []
    xran, yran, lmax = [], [], []
    for irow in range(1, npars):
        for icol in range(irow):
            ran = None
            if ranges[icol] is not None:
                ran = [ranges[icol], ranges[irow]]
            h, x, y = np.histogram2d(
                posterior[0::thinning,icol], posterior[0::thinning,irow],
                bins=nbins, range=ran, density=False)
            hist.append(h.T)
            xran.append(x)
            yran.append(y)
            lmax.append(np.amax(h)+1)
    # Reset upper boundary to absolute maximum value if requested:
    if absolute_dens:
        lmax = npars*(npars+1)*2 * [np.amax(lmax)]

    if rect is None:
        rect = (0.10, 0.10, 0.95, 0.95)
        plt.figure(fignum, figsize=(12, 12))
        plt.clf()

    axes = np.tile(None, (npars-1, npars-1))
    # Plot:
    k = 0 # Histogram index
    for irow in range(1, npars):
        for icol in range(irow):
            h = (npars-1)*(irow-1) + icol + 1  # Subplot index
            ax = axes[icol,irow-1] = subplotter(rect, margin, h, npars-1)
            # Labels:
            ax.tick_params(labelsize=fs-1, direction='in')
            if icol == 0:
                ax.set_ylabel(pnames[irow], size=fs)
            else:
                ax.set_yticklabels([])
            if irow == npars-1:
                ax.set_xlabel(pnames[icol], size=fs)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
            else:
                ax.set_xticklabels([])
            # The plot:
            cont = ax.contourf(hist[k], cmap=palette, rasterized = True, vmin=1, origin='lower',
                levels=[0]+list(np.linspace(1,lmax[k], nlevels)),
                extent=(xran[k][0], xran[k][-1], yran[k][0], yran[k][-1]))
            for c in cont.collections:
                c.set_edgecolor("face")
            if bestp is not None:
                ax.axvline(bestp[icol], dashes=(6,4), color="0.5", lw=1.0)
                ax.axhline(bestp[irow], dashes=(6,4), color="0.5", lw=1.0)
            if ranges[icol] is not None:
                ax.set_xlim(ranges[icol])
            if ranges[icol] is not None:
                ax.set_ylim(ranges[irow])
            k += 1

    # The colorbar:
    bounds = np.linspace(0, 1.0, nlevels)
    norm = mpl.colors.BoundaryNorm(bounds, palette.N)
    if rect is not None:
        dx = (rect[2]-rect[0])*0.05
        dy = (rect[3]-rect[1])*0.45
        ax2 = plt.axes([rect[2]-dx, rect[3]-dy, dx, dy])
    else:
        ax2 = plt.axes([0.95, 0.57, 0.025, 0.36])
    cb = mpl.colorbar.ColorbarBase(
        ax2, cmap=palette, norm=norm,
        spacing='proportional', boundaries=bounds, format='%.1f')
    cb.set_label("Posterior density", fontsize=fs)
    cb.ax.yaxis.set_ticks_position('left')
    cb.ax.yaxis.set_label_position('left')
    cb.ax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
    cb.set_ticks(np.linspace(0, 1, 5))
    for c in ax2.collections:
        c.set_edgecolor("face")
    plt.draw()

    # Save file:
    if savefile is not None:
        plt.suptitle(title)
        plt.ioff()
        plt.savefig(savefile, dpi = 50)
    return axes, cb

def modelfit(data, uncert, indparams, model, title, nbins=75,
    fignum=1400, savefile=None, fmt="."):
    """
    Plot the binned dataset with given uncertainties and model curves
    as a function of indparams.
    In a lower panel, plot the residuals bewteen the data and model.
    Parameters
    ----------
    data: 1D float ndarray
        Input data set.
    uncert: 1D float ndarray
        One-sigma uncertainties of the data points.
    indparams: 1D float ndarray
        Independent variable (X axis) of the data points.
    model: 1D float ndarray
        Model of data.
    nbins: Integer
        Number of bins in the output plot.
    fignum: Integer
        The figure number.
    savefile: Boolean
        If not None, name of file to save the plot.
    fmt: String
        Format of the plotted markers.
    Returns
    -------
    ax: matplotlib.axes.Axes
        Axes instance containing the marginal posterior distributions.
    """
    # Bin down array:
    binsize = int((np.size(data)-1)/nbins + 1)
    binindp  = ms.bin_array(indparams, binsize)
    binmodel = ms.bin_array(model,     binsize)
    bindata, binuncert = ms.bin_array(data, binsize, uncert)
    fs = 12 # Font-size

    plt.figure(fignum, figsize=(8,6))
    plt.clf()

    # Residuals:
    rax = plt.axes([0.15, 0.1, 0.8, 0.2])
    rax.errorbar(binindp, bindata-binmodel, binuncert, fmt='ko', ms=4)
    rax.plot([indparams[0], indparams[-1]], [0,0], 'k:', lw=1.5)
    rax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
    rax.set_xlabel("Wavenumber $(cm^{-1})$", fontsize=fs)
    rax.set_ylabel('Absorbance Residual', fontsize=fs)
    rax.invert_xaxis()

    # Data and Model:
    ax = plt.axes([0.15, 0.35, 0.8, 0.575])
    ax.errorbar(binindp, bindata, binuncert, fmt = 'ko', ms=4, label='Binned data')
    ax.plot(indparams, data, 'tab:blue', lw=3, label = 'Data')
    ax.plot(indparams, model, 'tab:purple', linestyle = '-.', lw=3, label='Best Fit')
    ax.set_xticklabels([])
    ax.invert_xaxis()
    ax.tick_params(labelsize=fs-1, direction='in', top=True, right=True)
    ax.set_ylabel('Absorbance', fontsize=fs)
    ax.legend(loc='best')

    if savefile is not None:
        plt.suptitle(title)
        plt.ioff()
        plt.savefig(savefile, backend='pgf')
    return ax, rax

def subplotter(rect, margin, ipan, nx, ny=None, ymargin=None):
    """
    Create an axis instance for one panel (with index ipan) of a grid
    of npanels, where the grid located inside rect (xleft, ybottom,
    xright, ytop).
    Parameters
    ----------
    rect: 1D List/ndarray
        Rectangle with xlo, ylo, xhi, yhi positions of the grid boundaries.
    margin: Float
        Width of margin between panels.
    ipan: Integer
        Index of panel to create (as in plt.subplots).
    nx: Integer
        Number of panels along the x axis.
    ny: Integer
        Number of panels along the y axis. If None, assume ny=nx.
    ymargin: Float
        Width of margin between panels along y axes (if None, adopt margin).
    Returns
    -------
    axes: Matplotlib.axes.Axes
        An Axes instance at the specified position.
    """
    if ny is None:
        ny = nx
    if ymargin is None:
        ymargin = margin

    # Size of a panel:
    Dx = rect[2] - rect[0]
    Dy = rect[3] - rect[1]
    dx = Dx/nx - (nx-1.0)* margin/nx
    dy = Dy/ny - (ny-1.0)*ymargin/ny
    # Position of panel ipan:
    # Follow plt's scheme, where panel 1 is at the top left panel,
    # panel 2 is to the right of panel 1, and so on:
    xloc = (ipan-1) % nx
    yloc = (ny-1) - ((ipan-1) // nx)
    # Bottom-left corner of panel:
    xpanel = rect[0] + xloc*(dx+ margin)
    ypanel = rect[1] + yloc*(dy+ymargin)

    return plt.axes([xpanel, ypanel, dx, dy])


# %% 

def MCMC(data, uncert, indparams, log, savefile):
    
    """The MCMC function takes the required inputs and runs the Monte Carlo-Markov Chain. The function ouputs the 
    mc3_output which contains all of the best fit parameters and standard deviations."""

    func = Carbonate
    params   =  np.array([1.25,  2.00,  0.25,  0.01,  0.01, 1430, 25.0, 0.0100, 1510, 25.0, 0.0100, 0.10,  0.02,  0.01,  5e-4,  0.70])
    pmin     =  np.array([0.00, -5.00, -1.00, -0.75, -0.75, 1415, 22.5, 0.0000, 1500, 22.5, 0.0000, 0.00, -0.50, -0.50, -5e-2, -1.00])
    pmax     =  np.array([5.00,  8.00,  1.00,  0.75,  0.75, 1445, 40.0, 2.0000, 1535, 40.0, 2.0000, 3.00,  0.50,  0.50,  5e-2,  3.00])
    pstep    =  np.array([0.30,  0.50,  0.20,  0.20,  0.20, 3.25, 2.25, 0.0005, 6.0, 2.25, 0.0005, 0.25,  0.75,  0.75,  0.002, 0.20])


    priorlow =  np.array([0.00,  0.00,  0.00,  0.00,  0.00, 0.0, 2.5, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    priorup  =  np.array([0.00,  0.00,  0.00,  0.00,  0.00, 0.0, 2.5, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    pnames   = ['Avg_BL',"PCA1","PCA2","PCA3","PCA4",'peak_G1430','std_G1430','G1430_amplitude','peak_G1515','std_G1515','G1515_amplitude','Average_1635Peak','1635PeakPCA1','1635PeakPCA2','m','b']
    texnames = ['$\overline{BL}$',"$PCA_1$","$PCA_2$","$PCA_3$",'$PCA_4$','$P_{1430}$','$S_{1430}$','$PH_{1430}$','$P_{1515}$','$S_{1515}$','$PH_{1515}$','$\overline{H_{1635}}$','${H_{1635,PCA_1}}$','${H_{1635,PCA_2}}$','m','b']

    mc3_output = mc3.sample(data, uncert, func=func, params=params, indparams=indparams, 
        pmin=pmin, pmax=pmax, priorlow=priorlow, priorup=priorup, 
        pnames=pnames, texnames=texnames, sampler='snooker', rms=False,
        nsamples=1e6, nchains=9, ncpu=3, burnin=5000, thinning=1, # 1e6, 5000
        leastsq='trf', chisqscale=False, grtest=True, grbreak=1.01, grnmin=0.5,
        hsize=10, kickoff='normal', wlike=False, plots=False, log=log, savefile=savefile)

    return mc3_output

# %% 


def Run_All_Spectra(dfs_dict, paths):

    """The Run_All_Spectra function inputs the dictionary of dataframes that were created by the Load_SampleCSV function and allows 
    for all of the samples to be batched and run through the function. The function exports the best fit and standard deviations 
    of peak locations, peak widths, and peak heights, as well as the PCA vectors used to fit the spectra. These values are 
    exported in a csv file and figures are created for each individual sample."""

    path_parent = os.path.dirname(os.getcwd())

    PCAmatrix = Load_PCA(paths[0])
    Peak_1635_PCAmatrix = Load_PCA(paths[1])
    Wavenumber = Load_Wavenumber(paths[0])
    path_beg, exportpath = paths[-2], paths[-1]
    Nvectors = 5
    indparams = [Wavenumber, PCAmatrix, Peak_1635_PCAmatrix, Nvectors]

    output_dir = ["FIGURES", "PLOTFILES", "NPZFILES", "LOGFILES"] 
    for ii in range(len(output_dir)):
        if not os.path.exists(path_beg + output_dir[ii] + '/' + exportpath):
            os.makedirs(path_beg + output_dir[ii] + '/' + exportpath, exist_ok=True)

    plotpath = 'PLOTFILES/' + exportpath + '/'
    logpath = 'LOGFILES/' + exportpath + '/'
    savefilepath = 'NPZFILES/' + exportpath + '/'
    figurepath = 'FIGURES/' + exportpath + '/'

    additional_dir = ["TRACE", "HISTOGRAM", "PAIRWISE", "MODELFIT"]
    for ii in range(len(additional_dir)): 
        if not os.path.exists(path_beg+plotpath+additional_dir[ii]): 
            os.makedirs(path_beg+plotpath+additional_dir[ii], exist_ok=True)

    # P_ = peak_, _BP = best parameter, #_STD = _stdev
    H2O_3550_PH = pd.DataFrame(columns=['PH_3550_M', 'PH_3550_STD', 'H2OT_3550_MAX', 'BL_H2OT_3550_MAX', 'H2OT_3550_SAT?'])
    DF_Output = pd.DataFrame(columns = ['PH_1635_BP','PH_1635_STD','H2Om_1635_MAX', 'BL_H2Om_1635_MAX', 
    'PH_1515_BP','PH_1515_STD','P_1515_BP','P_1515_STD','STD_1515_BP','STD_1515_STD','MAX_1515_ABS', 'BL_MAX_1515_ABS',
    'PH_1430_BP','PH_1430_STD','P_1430_BP','P_1430_STD','STD_1430_BP','STD_1430_STD','MAX_1430_ABS', 'BL_MAX_1430_ABS'])
    PCA_Output = pd.DataFrame(columns = ['AVG_BL_BP','AVG_BL_STD','PCA1_BP','PCA1_STD','PCA2_BP','PCA2_STD','PCA3_BP','PCA3_STD','PCA4_BP','PCA4_STD','m_BP','m_STD','b_BP','b_STD', 
    'PH_1635_PCA1_BP','PH_1635_PCA1_STD','PH_1635_PCA2_BP','PH_1635_PCA2_STD'])
    NEAR_IR_PH = pd.DataFrame(columns=['PH_5200_M', 'PH_4500_M', 'PH_5200_STD', 'PH_4500_STD', 'S2N_P5200', 'S2N_P4500', 'ERR_5200', 'ERR_4500'])

    # Run the MCMC:
    failures = []
    error = []
    error_4500 = []
    error_5200 = []

    try: 
        for files, data in dfs_dict.items(): 

            H2O4500_wn_low_1, H2O4500_wn_high_1 = 4250, 4675
            data_H2O4500_1, krige_output_4500_1, PH_4500_krige_1, STN_4500_1 = NearIR_Process(data, H2O4500_wn_low_1, H2O4500_wn_high_1, 'OH')
            H2O4500_wn_low_2, H2O4500_wn_high_2 = 4225, 4650
            data_H2O4500_2, krige_output_4500_2, PH_4500_krige_2, STN_4500_2 = NearIR_Process(data, H2O4500_wn_low_2, H2O4500_wn_high_2, 'OH')
            H2O4500_wn_low_3, H2O4500_wn_high_3 = 4275, 4700
            data_H2O4500_3, krige_output_4500_3, PH_4500_krige_3, STN_4500_3 = NearIR_Process(data, H2O4500_wn_low_3, H2O4500_wn_high_3, 'OH')

            # Three repeat baselines for the H2Om_{5200}
            H2O5200_wn_low_1, H2O5200_wn_high_1 = 4875, 5400
            data_H2O5200_1, krige_output_5200_1, PH_5200_krige_1, STN_5200_1 = NearIR_Process(data, H2O5200_wn_low_1, H2O5200_wn_high_1, 'H2O')
            H2O5200_wn_low_2, H2O5200_wn_high_2  = 4850, 5375
            data_H2O5200_2, krige_output_5200_2, PH_5200_krige_2, STN_5200_2 = NearIR_Process(data, H2O5200_wn_low_2, H2O5200_wn_high_2, 'H2O')
            H2O5200_wn_low_3, H2O5200_wn_high_3 = 4900, 5425
            data_H2O5200_3, krige_output_5200_3, PH_5200_krige_3, STN_5200_3 = NearIR_Process(data, H2O5200_wn_low_3, H2O5200_wn_high_3, 'H2O')

            PH_4500_krige = np.array([PH_4500_krige_1, PH_4500_krige_2, PH_4500_krige_3])
            PH_4500_krige_M, PH_4500_krige_STD = np.mean(PH_4500_krige), np.std(PH_4500_krige)
            PH_5200_krige = np.array([PH_5200_krige_1, PH_5200_krige_2, PH_5200_krige_3])
            PH_5200_krige_M, PH_5200_krige_STD = np.mean(PH_5200_krige), np.std(PH_5200_krige)

            STN_4500 = np.array([STN_4500_1, STN_4500_2, STN_4500_3])
            STN_4500_M = np.average(STN_4500)
            STN_5200 = np.array([STN_5200_1, STN_5200_2, STN_5200_3])
            STN_5200_M = np.average(STN_5200)

            if STN_4500_M >= 4.0: 
                error_4500 = '-'
            elif STN_4500_M < 4.0: 
                error_4500 = '*'

            if STN_5200_M >= 4.0: 
                error_5200 = '-'
            elif STN_5200_M < 4.0: 
                error_5200 = '*'

            plotmin = np.round(np.min(data[H2O4500_wn_low_1:H2O5200_wn_high_1]['Absorbance']), decimals = 1)
            plotmax = np.round(np.max(data[H2O4500_wn_low_1:H2O5200_wn_high_1]['Absorbance']), decimals = 1)
            fig, ax = plt.subplots(figsize = (26, 8))
            ax1 = plt.subplot2grid((2, 3), (0, 0))
            ax1.plot(data.index, data['Absorbance'], 'k', linewidth = 1.5)
            ax1.plot(data_H2O5200_1.index, data_H2O5200_1['Absorbance_Hat'], data_H2O5200_2.index, data_H2O5200_2['Absorbance_Hat'], data_H2O5200_3.index, data_H2O5200_3['Absorbance_Hat'])
            ax1.plot(data_H2O4500_1.index, data_H2O4500_1['Absorbance_Hat'], data_H2O4500_2.index, data_H2O4500_2['Absorbance_Hat'], data_H2O4500_3.index, data_H2O4500_3['Absorbance_Hat'])
            ax1.plot(data_H2O5200_1.index, data_H2O5200_1['BL_NIR_H2O'], data_H2O5200_2.index, data_H2O5200_2['BL_NIR_H2O'], data_H2O5200_3.index, data_H2O5200_3['BL_NIR_H2O'])
            ax1.plot(data_H2O4500_1.index, data_H2O4500_1['BL_NIR_H2O'], data_H2O4500_2.index, data_H2O4500_2['BL_NIR_H2O'], data_H2O4500_3.index, data_H2O4500_3['BL_NIR_H2O'])
            ax1.annotate(f"5200 cm $^{{- 1}}$  Peak Height Gaussian: {PH_5200_krige_M:.4f} ± {PH_5200_krige_STD:.4f} $\ cm ^{{- 1}} $, S2N = {STN_5200_M:.2f}", (0.025, 0.90), xycoords = 'axes fraction')
            ax1.annotate(f"4500 cm $^{{- 1}}$  Peak Height Gaussian: {PH_4500_krige_M:.4f} ± {PH_4500_krige_STD:.4f} $\ cm ^{{- 1}} $, S2N = {STN_4500_M:.2f}", (0.025, 0.80), xycoords = 'axes fraction')
            ax1.set_ylabel('Absorbance')
            warnings.filterwarnings("ignore", category = UserWarning)
            ax1.legend(['Near IR Data','_','_','Median Filtered 5200','_','_','Median Filtered 4500','_','_','Baseline 5200','_','_','Baseline 4500'], prop={'size': 12})
            ax1.set_xlim([4200, 5400])
            ax1.set_ylim([plotmin-0.075, plotmax+0.075])
            ax1.invert_xaxis()

            plotmax = np.round(np.max(data_H2O4500_1['Subtracted_Peak']), decimals = 1)
            ax2 = plt.subplot2grid((2, 3), (1, 0))
            ax2.plot(data_H2O5200_1.index, data_H2O5200_1['Subtracted_Peak'] - np.min(krige_output_5200_1['Absorbance']), 'k', data_H2O5200_2.index, data_H2O5200_2['Subtracted_Peak'] - np.min(krige_output_5200_2['Absorbance']), 'k', 
                data_H2O5200_3.index, data_H2O5200_3['Subtracted_Peak'] - np.min(krige_output_5200_3['Absorbance']), 'k', label = 'Subtracted Baseline 5200')
            ax2.plot(data_H2O4500_1.index, data_H2O4500_1['Subtracted_Peak'] - np.min(krige_output_4500_1['Absorbance']), 'k', data_H2O4500_2.index, data_H2O4500_2['Subtracted_Peak'] - np.min(krige_output_4500_2['Absorbance']), 'k', 
                data_H2O4500_3.index, data_H2O4500_3['Subtracted_Peak'] - np.min(krige_output_4500_3['Absorbance']), 'k', label = 'Subtracted Baseline 4500')
            ax2.plot(krige_output_5200_1.index, krige_output_5200_1['Absorbance'] - np.min(krige_output_5200_1['Absorbance']), krige_output_5200_2.index, krige_output_5200_2['Absorbance'] - np.min(krige_output_5200_2['Absorbance']),
                krige_output_5200_3.index, krige_output_5200_3['Absorbance'] - np.min(krige_output_5200_3['Absorbance']), label = 'Gaussian 5200')
            ax2.plot(krige_output_4500_1.index, krige_output_4500_1['Absorbance'] - np.min(krige_output_4500_1['Absorbance']), krige_output_4500_2.index, krige_output_4500_2['Absorbance'] - np.min(krige_output_4500_2['Absorbance']), 
                krige_output_4500_3.index, krige_output_4500_3['Absorbance'] - np.min(krige_output_4500_3['Absorbance']), label = 'Gaussian 4500')
            ax2.set_xlabel('Wavenumber $cm^{-1}$')    
            ax2.set_ylabel('Absorbance')
            warnings.filterwarnings("ignore", category = UserWarning)
            ax2.legend(['Subtracted 5200','_','_','Subtracted 4500','_','_','_','_','Gaussian 5200','_','_','Gaussian 4500'])
            ax2.set_xlim([4200, 5400])
            ax2.set_ylim([0, plotmax+0.05])
            ax2.invert_xaxis()

            NEAR_IR_PH.loc[files] = pd.Series({'PH_5200_M': PH_5200_krige_M, 'PH_4500_M': PH_4500_krige_M, 
            'PH_5200_STD': PH_5200_krige_STD, 'PH_4500_STD': PH_4500_krige_STD, 
            'S2N_P5200': STN_5200_M, 'S2N_P4500': STN_4500_M, 'ERR_5200': error_5200, 'ERR_4500': error_4500})

            H2O3550_wn_low_1  = 1900
            H2O3550_wn_high_1 = 4400
            data_H2O3550_1, plot_output_3550_1, PH_3550_1, plotindex1 = MidIR_Process(data, H2O3550_wn_low_1, H2O3550_wn_high_1)
            H2O3550_wn_low_2  = 2100
            H2O3550_wn_high_2 = 4200
            data_H2O3550_2, plot_output_3550_2, PH_3550_2, plotindex2 = MidIR_Process(data, H2O3550_wn_low_2, H2O3550_wn_high_2)
            H2O3550_wn_low_3  = 2300
            H2O3550_wn_high_3 = 4000
            data_H2O3550_3, plot_output_3550_3, PH_3550_3, plotindex3 = MidIR_Process(data, H2O3550_wn_low_3, H2O3550_wn_high_3)
            PH_3550_M = np.mean([PH_3550_1, PH_3550_2, PH_3550_3])
            PH_3550_STD = np.std([PH_3550_1, PH_3550_2, PH_3550_3])

            MAX_3550_ABS = data_H2O3550_1['Absorbance'].to_numpy()[np.argmax(data_H2O3550_1['Absorbance'].index.to_numpy() > 3550)]
            MAX_1635_ABS = data_H2O3550_1['Absorbance'].to_numpy()[np.argmax((data_H2O3550_1['Absorbance'].index.to_numpy() > 1600) & (data_H2O3550_1['Absorbance'].index.to_numpy() < 1800))]

            BL_MAX_3550_ABS = data_H2O3550_1['BL_MIR_3550'].to_numpy()[np.argmax(data_H2O3550_1['Absorbance'].index.to_numpy() > 3550)]

            if MAX_3550_ABS < 2: 
                error = '-'
            elif MAX_3550_ABS >= 2: 
                error = '*'
            
            plotmax = np.round(np.max(data_H2O3550_1['Absorbance'].to_numpy()), decimals = 0)
            plotmin = np.round(np.min(data_H2O3550_1['Absorbance'].to_numpy()), decimals = 0)
            ax3 = plt.subplot2grid((2, 3), (0, 1), rowspan = 2)
            ax3.plot(data.index, data['Absorbance'], 'k')
            ax3.plot(data_H2O3550_1['Absorbance'].index, data_H2O3550_1['BL_MIR_3550'], data_H2O3550_2['Absorbance'].index, 
                data_H2O3550_2['BL_MIR_3550'], data_H2O3550_3['Absorbance'].index, data_H2O3550_3['BL_MIR_3550'])
            ax3.plot(plot_output_3550_1.index, (plot_output_3550_1['Subtracted_Peak_Hat']+plot_output_3550_1['BL_MIR_3550']), 'r', linewidth = 2)
            ax3.plot(plot_output_3550_2.index, (plot_output_3550_2['Subtracted_Peak_Hat']+plot_output_3550_2['BL_MIR_3550']), 'r', linewidth = 2)
            ax3.plot(plot_output_3550_3.index, (plot_output_3550_3['Subtracted_Peak_Hat']+plot_output_3550_3['BL_MIR_3550']), 'r', linewidth = 2)
            ax3.set_title(files)
            ax3.annotate(f"3550 cm $^{{- 1}}$  Peak Height: {PH_3550_M:.4f} ± {PH_3550_STD:.4f} $\ cm ^{{- 1}}$", (0.025, 0.95), xycoords = 'axes fraction')
            ax3.set_xlabel('Wavenumber $cm^{-1}$')
            ax3.set_xlim([1275, 4000])
            ax3.set_ylabel('Absorbance')
            ax3.set_ylim([plotmin-0.25, plotmax+0.5])
            ax3.invert_xaxis()

            H2O_3550_PH.loc[files] = pd.Series({'PH_3550_M': PH_3550_M, 'PH_3550_STD': PH_3550_STD, 'H2OT_3550_MAX': MAX_3550_ABS, 
            'BL_H2OT_3550_MAX': BL_MAX_3550_ABS, 'H2OT_3550_SAT?': error})

            df_length = np.shape(Wavenumber)[0]
            
            CO2_wn_high = 2200
            CO2_wn_low  = 1275 

            spec = data[CO2_wn_low:CO2_wn_high]

            if spec.shape[0] != df_length:              
                interp_wn = np.linspace(spec.index[0], spec.index[-1], df_length)
                interp_abs = interpolate.interp1d(spec.index, spec['Absorbance'])(interp_wn)
                spec = spec.reindex(index = interp_wn)
                spec['Absorbance'] = interp_abs
                spec_mc3 = spec['Absorbance'].to_numpy()
            elif spec.shape[0] == df_length: 
                spec_mc3 = spec['Absorbance'].to_numpy()

            uncert = np.ones_like(spec_mc3) * 0.01
            mc3_output = MCMC(data = spec_mc3, uncert = uncert, indparams = indparams, log = path_beg+logpath+files+'.log', savefile=path_beg+savefilepath+files+'.npz')

            PCA_BP = mc3_output['bestp'][0:Nvectors]
            CO2P_BP = mc3_output['bestp'][-11:-5]
            H2OmP1635_BP = mc3_output['bestp'][-5:-2]
            m_BP, b_BP = mc3_output['bestp'][-2:None]
            Baseline_Solve_BP = PCA_BP * PCAmatrix.T
            Baseline_Solve_BP = np.asarray(Baseline_Solve_BP).ravel()

            PCA_STD = mc3_output['stdp'][0:Nvectors]
            CO2P_STD = mc3_output['stdp'][-11:-5]
            H2OmP1635_STD = mc3_output['stdp'][-5:-2]
            H2OmP1635_STD[0] = H2OmP1635_STD[0]
            m_STD, b_STD = mc3_output['stdp'][-2:None]

            Line_BP = Linear(Wavenumber, m_BP, b_BP) 
            Baseline_Solve_BP = Baseline_Solve_BP + Line_BP

            H1635_BP = H2OmP1635_BP * Peak_1635_PCAmatrix.T
            H1635_BP = np.asarray(H1635_BP).ravel()

            CO2P1430_BP = Gauss(Wavenumber, CO2P_BP[0], CO2P_BP[1], A=CO2P_BP[2])
            CO2P1515_BP = Gauss(Wavenumber, CO2P_BP[3], CO2P_BP[4], A=CO2P_BP[5])

            H1635_SOLVE = H1635_BP + Baseline_Solve_BP
            CO2P1515_SOLVE = CO2P1515_BP + Baseline_Solve_BP
            CO2P1430_SOLVE = CO2P1430_BP + Baseline_Solve_BP

            MAX_1515_ABS = spec['Absorbance'].to_numpy()[np.argmax((spec.index.to_numpy() > 1510) & (spec.index.to_numpy() < 1530))]
            MAX_1430_ABS = spec['Absorbance'].to_numpy()[np.argmax((spec.index.to_numpy() > 1410) & (spec.index.to_numpy() < 1440))]

            BL_MAX_1515_ABS = Baseline_Solve_BP[np.argmax((spec.index.to_numpy() > 1510) & (spec.index.to_numpy() < 1530))]
            BL_MAX_1430_ABS = Baseline_Solve_BP[np.argmax((spec.index.to_numpy() > 1410) & (spec.index.to_numpy() < 1450))]

            posteriorerror = np.load(path_beg+savefilepath+files+'.npz')
            samplingerror = posteriorerror['posterior'][:, 0:5]
            samplingerror = samplingerror[0: np.shape(posteriorerror['posterior'][:, :])[0] :int(np.shape(posteriorerror['posterior'][:, :])[0] / 200), :]
            lineerror = posteriorerror['posterior'][:, -2:None]
            lineerror = lineerror[0:np.shape(posteriorerror['posterior'][:, :])[0]:int(np.shape(posteriorerror['posterior'][:, :])[0] / 200), :]
            Baseline_Array = np.array(samplingerror * PCAmatrix[:, :].T)
            Baseline_Array_Plot = Baseline_Array

            ax4 = plt.subplot2grid((2, 3), (0, 2), rowspan = 2)
            for i in range(np.shape(Baseline_Array)[0]):
                Linearray = Linear(Wavenumber, lineerror[i, 0], lineerror[i, 1])
                Baseline_Array_Plot[i, :] = Baseline_Array[i, :] + Linearray
                plt.plot(Wavenumber, Baseline_Array_Plot[i, :], 'dimgray', linewidth = 0.25)
            ax4.plot(Wavenumber, spec_mc3, 'tab:blue', linewidth = 2.5, label = 'FTIR Spectrum')
            ax4.plot(Wavenumber, H1635_SOLVE, 'tab:orange', linewidth = 1.5, label = '1635')
            ax4.plot(Wavenumber, CO2P1515_SOLVE, 'tab:green', linewidth = 2.5, label = '1515')
            ax4.plot(Wavenumber, CO2P1430_SOLVE, 'tab:red', linewidth = 2.5, label = '1430')
            ax4.plot(Wavenumber, Carbonate(mc3_output['meanp'], Wavenumber, PCAmatrix, Peak_1635_PCAmatrix, Nvectors), 'tab:purple', linewidth = 1.5, label = 'MC3 Fit')
            ax4.plot(Wavenumber, Baseline_Solve_BP, 'k', linewidth = 1.5, label = 'Baseline')
            ax4.annotate(f"1635 cm $^{{- 1}}$  Peak Height: {H2OmP1635_BP[0]:.3f} ± {H2OmP1635_STD[0]:.3f} $\ cm ^{{- 1}} $ ", (0.025, 0.95), xycoords = 'axes fraction')
            ax4.annotate(f"1515 cm $^{{- 1}}$  Peak Height: {CO2P_BP[5]:.3f} ± {CO2P_STD[5]:.3f} $\ cm ^{{- 1}} $ ", (0.025, 0.90), xycoords = 'axes fraction')
            ax4.annotate(f"1430 cm $^{{- 1}}$  Peak Height: {CO2P_BP[2]:.3f} ± {CO2P_STD[2]:.3f} $\ cm ^{{- 1}} $ ", (0.025, 0.85), xycoords = 'axes fraction')
            ax4.set_title(files)
            ax4.set_xlim([1275, 2000])
            ax4.set_xlabel('Wavenumber $cm^{-1}$')
            ax4.set_ylabel('Absorbance')
            ax4.legend(loc = 'upper right', prop={'size': 12})
            ax4.invert_xaxis()
            plt.tight_layout()
            plt.savefig(path_beg + figurepath + files + '.pdf', backend='pgf')
            plt.close('all')

            BL_MAX_1635_ABS = Baseline_Solve_BP[np.argmax(H1635_BP)]

            texnames = ['$\overline{BL}$',"$PCA_1$","$PCA_2$","$PCA_3$",'$PCA_4$','$P_{1430}$','$S_{1430}$','$A_{1430}$','$P_{1515}$','$S_{1515}$','$A_{1515}$','$\overline{H_{1635}}$','${H_{1635,PCA_1}}$','${H_{1635,PCA_2}}$','m','b']

            fig1 = trace(mc3_output['posterior'], title = files, zchain=mc3_output['zchain'], burnin=mc3_output['burnin'], pnames=texnames, savefile=path_beg+plotpath+'TRACE/'+files+'_trace.pdf')
            plt.close('all')
            fig2 = histogram(mc3_output['posterior'], title = files, pnames=texnames, bestp=mc3_output['bestp'], savefile=path_beg+plotpath+'HISTOGRAM/'+files+'_histogram.pdf', quantile=0.683)
            plt.close('all')
            fig3 = pairwise(mc3_output['posterior'], title = files, pnames=texnames, bestp=mc3_output['bestp'], savefile=path_beg+plotpath+'PAIRWISE/'+files+'_pairwise.pdf')
            plt.close('all')
            fig4 = modelfit(spec_mc3, uncert, indparams[0], mc3_output['best_model'], title = files, savefile=path_beg+plotpath+'MODELFIT/'+files+'_modelfit.pdf')
            plt.close('all')

            # Create dataframe of best fit parameters and their standard deviations
            DF_Output.loc[files] = pd.Series({'PH_1635_BP':H2OmP1635_BP[0],'PH_1635_STD':H2OmP1635_STD[0],'H2Om_1635_MAX': MAX_1635_ABS, 'BL_H2Om_1635_MAX': BL_MAX_1635_ABS,
            'PH_1515_BP':CO2P_BP[5],'PH_1515_STD':CO2P_STD[5],'P_1515_BP':CO2P_BP[3],'P_1515_STD':CO2P_STD[3],'STD_1515_BP':CO2P_BP[4],'STD_1515_STD':CO2P_STD[4], 
            'PH_1430_BP':CO2P_BP[2],'PH_1430_STD':CO2P_STD[2],'P_1430_BP':CO2P_BP[0],'P_1430_STD':CO2P_STD[0],'STD_1430_BP':CO2P_BP[1],'STD_1430_STD':CO2P_STD[1], 
            'MAX_1515_ABS': MAX_1515_ABS, 'BL_MAX_1515_ABS': BL_MAX_1515_ABS,
            'MAX_1430_ABS': MAX_1430_ABS, 'BL_MAX_1430_ABS': BL_MAX_1430_ABS})

            PCA_Output.loc[files] = pd.Series({'AVG_BL_BP':PCA_BP[0],'AVG_BL_STD':PCA_STD[0],'PCA1_BP':PCA_BP[1],'PCA1_STD':PCA_STD[1],'PCA2_BP':PCA_BP[2],'PCA2_STD':PCA_STD[2], 
            'PCA3_BP':PCA_BP[3],'PCA3_STD':PCA_STD[3],'PCA4_BP':PCA_BP[4],'PCA4_STD':PCA_STD[4],'m_BP':m_BP,'m_STD':m_STD,'b_BP':b_BP,'b_STD':b_STD, 
            'PH_1635_PCA1_BP':H2OmP1635_BP[1],'PH_1635_PCA1_STD':H2OmP1635_STD[1],'PH_1635_PCA2_BP':H2OmP1635_BP[2],'PH_1635_PCA2_STD':H2OmP1635_STD[2]})

    except:
        failures.append(files)
        print(files + ' failed.')

    Volatiles_DF = pd.concat([H2O_3550_PH, DF_Output, NEAR_IR_PH, PCA_Output], axis = 1)

    return Volatiles_DF, failures


# %% 

def Beer_Lambert(molar_mass, absorbance, sigma_absorbance, density, sigma_density, thickness, sigma_thickness, epsilon, sigma_epsilon):

    """The Beer_Lambert function applies the Beer-Lambert Law with the inputs of molar mass, absorbance, density, thickness, and epsilon (absorbance coefficient), 
    as well as the uncertainty associated with each term aside from molar mass and returns the concentration with a multiplier of concentration uncertainty,
    calculated by standard error propagation techniques."""

    # https://sites.fas.harvard.edu/~scphys/nsta/error_propagation.pdf

    concentration = pd.Series()
    concentration = (1e6 * molar_mass * absorbance) / (density * thickness * epsilon)

    return concentration


def Beer_Lambert_Error(N, molar_mass, absorbance, sigma_absorbance, density, sigma_density, thickness, sigma_thickness, epsilon, sigma_epsilon): 

    """The Beer_Lambert_Error function applies a simple Monte Carlo with the same inputs as the Beer_Lambert function, with the addition 
    of N (number of samples), and outputs the uncertainty in concentration. Absorbance, density, and thickness is treated with a Gaussian distribution, 
    and absorbance coefficient is treated with a uniform distribution."""

    # https://astrofrog.github.io/py4sci/_static/Practice%20Problem%20-%20Monte-Carlo%20Error%20Propagation%20-%20Sample%20Solution.html
    gaussian_concentration = (1e6 * molar_mass * np.random.normal(absorbance, sigma_absorbance, N) / (np.random.normal(density, sigma_density, N) 
        * np.random.normal(thickness, sigma_thickness, N) * np.random.uniform(epsilon-sigma_epsilon, epsilon+sigma_epsilon, N)))
    concentration_std = np.std(gaussian_concentration)

    return concentration_std


def Density_Calculation(MI_Composition):

    """The Density_Calculation function inputs the MI composition file and outputs the glass density at room temperature and pressure. 
    The mole fraction is calculated. The total molar volume xivibari is determined from sum of the mole fractions of each oxide * partial molar volume 
    at room temperature and pressure of analysis. The gram formula weight gfw is then determined by summing the mole fractions*molar masses. 
    The density is finally determined by dividing gram formula weight by total molar volume."""

    molar_mass = {'SiO2': 60.08, 'TiO2': 79.866, 'Al2O3': 101.96, 'Fe2O3': 159.69, 'FeO': 71.844, 'MnO': 70.9374, 
        'MgO': 40.3044, 'CaO': 56.0774, 'Na2O': 61.9789, 'K2O': 94.2, 'P2O5': 141.9445, 'H2O': 18.01528, 'CO2': 44.01}

    T_room = 25 # during analysis (deg. C)
    P_room = 1  # during analysis (bars)

    # Partial Molar Volumes from Lesher and Spera, 2015
    par_molar_vol = {'SiO2': (26.86-1.89*P_room/1000), 'TiO2': (23.16+7.24*(T_room+273-1673)/1000-2.31*P_room/1000), 'Al2O3': (37.42-2.26*P_room/1000), 
        'Fe2O3': (42.13+9.09*(T_room+273-1673)/1000-2.53*P_room/1000), 'FeO': (13.65+2.92*(T_room+273-1673)/1000-0.45*P_room/1000),
        'MgO': (11.69+3.27*(T_room+273-1673)/1000+0.27*P_room/1000), 'CaO': (16.53+3.74*(T_room+273-1673)/1000+0.34*P_room/1000), 
        'Na2O': (28.88+7.68*(T_room+273-1673)/1000-2.4*P_room/1000), 'K2O': (45.07+12.08*(T_room+273-1673)/1000-6.75*P_room/1000), 
        'H2O': (26.27+9.46*(T_room+273-1673)/1000-3.15*P_room/1000)}

    mol = pd.DataFrame()
    for oxide in MI_Composition:
        mol[oxide] = MI_Composition[oxide]/molar_mass[oxide]

    mol_tot = pd.DataFrame()
    mol_tot = mol.sum(axis = 1)

    oxide_density = pd.DataFrame()
    xivbari = pd.DataFrame()
    gfw = pd.DataFrame() # gram formula weight
    for oxide in MI_Composition:
        if oxide in par_molar_vol:
            xivbari[oxide] = mol[oxide]/mol_tot*par_molar_vol[oxide]
            gfw[oxide] = mol[oxide]/mol_tot*molar_mass[oxide]

    xivbari_tot = xivbari.sum(axis = 1)
    gfw_tot = gfw.sum(axis = 1)
    
    density = 1000 * gfw_tot / xivbari_tot

    return mol, density


# %% 


def Concentration_Output(Volatiles_DF, N, thickness, MI_Composition, sigma_thickness = 3):

    """The Concentration_Output function inputs a dictionary with the peak heights for the total H2O peak (3550 cm^-1), molecular H2O peak (1635 cm^-1), 
    and carbonate peaks (1515 and 1430 cm^-1), number of samples for the Monte Carlo, thickness information, and MI composition, and 
    outputs the concentrations and uncertainties for each peak. Both the best fit parameter and mean from the MC3 code are used to calculate 
    concentration."""

    mega_spreadsheet = pd.DataFrame(columns = ['H2OT_MEAN', 'H2OT_STD','H2OT_3550_M', 'H2OT_3550_STD', 'H2OT_3550_SAT', 'H2Om_1635_BP', 'H2Om_1635_STD', 'CO2_MEAN', 'CO2_STD',
        'CO2_1515_BP', 'CO2_1515_STD', 'CO2_1430_BP', 'CO2_1430_STD', 'H2Om_5200_M', 'H2Om_5200_STD', 'OH_4500_M', 'OH_4500_STD'])
    mega_spreadsheet_sat = pd.DataFrame(columns = ['H2OT_MEAN', 'H2OT_STD','H2OT_3550_M', 'H2OT_3550_STD', 'H2OT_3550_SAT', 'H2Om_1635_BP', 'H2Om_1635_STD', 'CO2_MEAN', 'CO2_STD',
        'CO2_1515_BP', 'CO2_1515_STD', 'CO2_1430_BP', 'CO2_1430_STD', 'H2Om_5200_M', 'H2Om_5200_STD', 'OH_4500_M', 'OH_4500_STD'])
    epsilon = pd.DataFrame(columns=['Tau', 'Na/Na+Ca', 'epsilon_H2OT_3550', 'sigma_epsilon_H2OT_3550', 'epsilon_H2Om_1635', 'sigma_epsilon_H2Om_1635', 
        'epsilon_CO2', 'sigma_epsilon_CO2', 'epsilon_H2Om_5200', 'sigma_epsilon_H2Om_5200', 'epsilon_OH_4500', 'sigma_epsilon_OH_4500'])
    density_df = pd.DataFrame(columns=['Density'])
    density_sat_df = pd.DataFrame(columns=['Density_Sat'])
    mean_vol = pd.DataFrame(columns = ['H2OT_MEAN', 'H2OT_STD', 'CO2_MEAN', 'CO2_STD'])
    s2nerror = pd.DataFrame(columns = ['PH_5200_S2N', 'PH_4500_S2N', 'ERR_5200', 'ERR_4500'])

    molar_mass = {'SiO2': 60.08, 'TiO2': 79.866, 'Al2O3': 101.96, 'Fe2O3': 159.69, 'FeO': 71.844, 'MnO': 70.9374, 
        'MgO': 40.3044, 'CaO': 56.0774, 'Na2O': 61.9789, 'K2O': 94.2, 'P2O5': 141.9445, 'H2O': 18.01528, 'CO2': 44.01}

    MI_Composition['H2O'] = 0
    mol, density = Density_Calculation(MI_Composition)

    cation_tot = mol.sum(axis = 1) + mol['Al2O3'] + mol['Na2O'] + mol['K2O'] + mol['P2O5']
    Na_NaCa = (2*mol['Na2O']) / ((2*mol['Na2O']) + mol['CaO'])
    SiAl_tot = (mol['SiO2'] + (2*mol['Al2O3'])) / cation_tot

    mest_3550, mest_1635, mest_CO2 = np.array([15.725557, 71.368691]), np.array([-50.397564, 124.250534]), np.array([440.6964, -355.2053])
    covm_est_3550, covm_est_1635, covm_est_CO2 = np.diag([38.4640, 77.8597]), np.diag([20.8503, 39.3875]), np.diag([103.7645, 379.9891])
    mest_4500, mest_5200 = np.array([-1.632730, 3.532522]), np.array([-2.291420, 4.675528])
    covm_est_4500, covm_est_5200 = np.diag([0.0329, 0.0708]), np.diag([0.0129, 0.0276])
    sigma_thickness = sigma_thickness

    G_SiAl, G_NaCa = np.ones((2, 1)),  np.ones((2, 1))
    covz_error_SiAl, covz_error_NaCa = np.zeros((2, 2)), np.zeros((2, 2))

    for i in MI_Composition.index:

        epsilon_H2OT_3550 = mest_3550[0]+(mest_3550[1]*SiAl_tot[i])
        epsilon_H2Om_1635 = mest_1635[0]+(mest_1635[1]*SiAl_tot[i])
        epsilon_CO2 = mest_CO2[0]+(mest_CO2[1]*Na_NaCa[i])
        epsilon_H2Om_5200 = mest_5200[0]+(mest_5200[1]*SiAl_tot[i])
        epsilon_OH_4500 = mest_4500[0]+(mest_4500[1]*SiAl_tot[i])

        G_SiAl[1, 0] = SiAl_tot[i]
        G_NaCa[1, 0] = Na_NaCa[i]
        covz_error_SiAl[1, 1] = SiAl_tot[i] * 0.01 # 1 sigma
        covz_error_NaCa[1, 1] = Na_NaCa[i] * 0.01

        CT_int_3550 = (G_SiAl*covm_est_3550*np.transpose(G_SiAl)) + (mest_3550*covz_error_SiAl*np.transpose(mest_3550))
        CT68_3550 = np.sqrt(np.mean(np.diag(CT_int_3550)))

        CT_int_1635 = (G_SiAl*covm_est_1635*np.transpose(G_SiAl)) + (mest_1635*covz_error_SiAl*np.transpose(mest_1635))
        CT68_1635 = np.sqrt(np.mean(np.diag(CT_int_1635)))

        CT_int_CO2 = (G_NaCa*covm_est_CO2*np.transpose(G_NaCa)) + (mest_CO2*covz_error_NaCa*np.transpose(mest_CO2))
        CT68_CO2 = np.sqrt(np.mean(np.diag(CT_int_CO2)))

        CT_int_5200 = (G_SiAl*covm_est_5200*np.transpose(G_SiAl)) + (mest_5200*covz_error_SiAl*np.transpose(mest_5200))
        CT68_5200 = np.sqrt(np.mean(np.diag(CT_int_5200)))
        CT_int_4500 = (G_SiAl*covm_est_4500*np.transpose(G_SiAl)) + (mest_4500*covz_error_SiAl*np.transpose(mest_4500))
        CT68_4500 = np.sqrt(np.mean(np.diag(CT_int_4500)))

        epsilon.loc[i] = pd.Series({'Tau': SiAl_tot[i], 'Na/Na+Ca': Na_NaCa[i], 'epsilon_H2OT_3550': epsilon_H2OT_3550, 'sigma_epsilon_H2OT_3550': CT68_3550, 
        'epsilon_H2Om_1635': epsilon_H2Om_1635, 'sigma_epsilon_H2Om_1635': CT68_1635, 'epsilon_CO2': epsilon_CO2, 'sigma_epsilon_CO2': CT68_CO2, 
        'epsilon_H2Om_5200': epsilon_H2Om_5200, 'sigma_epsilon_H2Om_5200': CT68_5200, 'epsilon_OH_4500': epsilon_OH_4500, 'sigma_epsilon_OH_4500': CT68_4500})

    # Doing density-H2O iterations:
    for j in range(10):
        H2OT_3550_I = Beer_Lambert(molar_mass['H2O'], Volatiles_DF['PH_3550_M'], Volatiles_DF['PH_3550_STD'],
            density, density * 0.025, thickness['Thickness'], sigma_thickness, epsilon['epsilon_H2OT_3550'], epsilon['sigma_epsilon_H2OT_3550'])
        MI_Composition['H2O'] = H2OT_3550_I
        mol, density = Density_Calculation(MI_Composition)

    # Doing density-H2O iterations:
    for k in Volatiles_DF.index: 
        # if Volatiles_DF['H2OT_3550_SAT?'][k] == '-': 
        H2OT_3550_M = Beer_Lambert(molar_mass['H2O'], Volatiles_DF['PH_3550_M'][k], Volatiles_DF['PH_3550_STD'][k],
            density[k], density[k] * 0.025, thickness['Thickness'][k], sigma_thickness, epsilon['epsilon_H2OT_3550'][k], epsilon['sigma_epsilon_H2OT_3550'][k])
        H2Om_1635_BP = Beer_Lambert(molar_mass['H2O'], Volatiles_DF['PH_1635_BP'][k], Volatiles_DF['PH_1635_STD'][k],
            density[k], density[k] * 0.025, thickness['Thickness'][k], sigma_thickness, epsilon['epsilon_H2Om_1635'][k], epsilon['sigma_epsilon_H2Om_1635'][k])
        CO2_1515_BP = Beer_Lambert(molar_mass['CO2'], Volatiles_DF['PH_1515_BP'][k], Volatiles_DF['PH_1515_STD'][k],
            density[k], density[k] * 0.025, thickness['Thickness'][k], sigma_thickness, epsilon['epsilon_CO2'][k], epsilon['sigma_epsilon_CO2'][k])
        CO2_1430_BP = Beer_Lambert(molar_mass['CO2'], Volatiles_DF['PH_1430_BP'][k], Volatiles_DF['PH_1430_STD'][k],
            density[k], density[k] * 0.025, thickness['Thickness'][k], sigma_thickness, epsilon['epsilon_CO2'][k], epsilon['sigma_epsilon_CO2'][k])
        H2Om_5200_M = Beer_Lambert(molar_mass['H2O'], Volatiles_DF['PH_5200_M'][k], Volatiles_DF['PH_5200_STD'][k],
            density[k], density[k] * 0.025, thickness['Thickness'][k], sigma_thickness, epsilon['epsilon_H2Om_5200'][k], epsilon['sigma_epsilon_H2Om_5200'][k])
        OH_4500_M = Beer_Lambert(molar_mass['H2O'], Volatiles_DF['PH_4500_M'][k], Volatiles_DF['PH_4500_STD'][k],
            density[k], density[k] * 0.025, thickness['Thickness'][k], sigma_thickness, epsilon['epsilon_OH_4500'][k], epsilon['sigma_epsilon_OH_4500'][k])
        CO2_1515_BP *= 10000
        CO2_1430_BP *= 10000
        
        H2OT_3550_M_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_DF['PH_3550_M'][k], Volatiles_DF['PH_3550_STD'][k],
            density[k], density[k] * 0.025, thickness['Thickness'][k], sigma_thickness, epsilon['epsilon_H2OT_3550'][k], epsilon['sigma_epsilon_H2OT_3550'][k])
        H2Om_1635_BP_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_DF['PH_1635_BP'][k], Volatiles_DF['PH_1635_STD'][k],
            density[k], density[k] * 0.025, thickness['Thickness'][k], sigma_thickness, epsilon['epsilon_H2Om_1635'][k], epsilon['sigma_epsilon_H2Om_1635'][k])
        CO2_1515_BP_STD = Beer_Lambert_Error(N, molar_mass['CO2'], Volatiles_DF['PH_1515_BP'][k], Volatiles_DF['PH_1515_STD'][k],
            density[k], density[k] * 0.025, thickness['Thickness'][k], sigma_thickness, epsilon['epsilon_CO2'][k], epsilon['sigma_epsilon_CO2'][k])
        CO2_1430_BP_STD = Beer_Lambert_Error(N, molar_mass['CO2'], Volatiles_DF['PH_1430_BP'][k], Volatiles_DF['PH_1430_STD'][k],
            density[k], density[k] * 0.025, thickness['Thickness'][k], sigma_thickness, epsilon['epsilon_CO2'][k], epsilon['sigma_epsilon_CO2'][k])
        H2Om_5200_M_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_DF['PH_5200_M'][k], Volatiles_DF['PH_5200_STD'][k],
            density[k], density[k] * 0.025, thickness['Thickness'][k], sigma_thickness, epsilon['epsilon_H2Om_5200'][k], epsilon['sigma_epsilon_H2Om_5200'][k])
        OH_4500_M_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_DF['PH_4500_M'][k], Volatiles_DF['PH_4500_STD'][k],
            density[k], density[k] * 0.025, thickness['Thickness'][k], sigma_thickness, epsilon['epsilon_OH_4500'][k], epsilon['sigma_epsilon_OH_4500'][k])
        CO2_1515_BP_STD *= 10000
        CO2_1430_BP_STD *= 10000

        density_df.loc[k] = pd.Series({'Density': density[k]})
        mega_spreadsheet.loc[k] = pd.Series({'H2OT_3550_M': H2OT_3550_M, 'H2OT_3550_STD': H2OT_3550_M_STD, 'H2OT_3550_SAT': Volatiles_DF['H2OT_3550_SAT?'][k], 
            'H2Om_1635_BP': H2Om_1635_BP, 'H2Om_1635_STD': H2Om_1635_BP_STD, 
            'CO2_1515_BP': CO2_1515_BP, 'CO2_1515_STD': CO2_1515_BP_STD, 
            'CO2_1430_BP': CO2_1430_BP, 'CO2_1430_STD': CO2_1430_BP_STD, 
            'H2Om_5200_M': H2Om_5200_M, 'H2Om_5200_STD': H2Om_5200_M_STD, 
            'OH_4500_M': OH_4500_M, 'OH_4500_STD': OH_4500_M_STD})

    for l in Volatiles_DF.index: 
        if Volatiles_DF['H2OT_3550_SAT?'][l] == '-': 
            H2OT_3550_M = mega_spreadsheet['H2OT_3550_M'][l]
            H2Om_1635_BP = mega_spreadsheet['H2Om_1635_BP'][l]
            CO2_1515_BP = mega_spreadsheet['CO2_1515_BP'][l]
            CO2_1430_BP = mega_spreadsheet['CO2_1430_BP'][l]
            H2Om_5200_M = mega_spreadsheet['H2Om_5200_M'][l]
            OH_4500_M = mega_spreadsheet['OH_4500_M'][l]

            H2OT_3550_M_STD = mega_spreadsheet['H2OT_3550_STD'][l]
            H2Om_1635_BP_STD = mega_spreadsheet['H2Om_1635_STD'][l]
            CO2_1515_BP_STD = mega_spreadsheet['CO2_1515_STD'][l]
            CO2_1430_BP_STD = mega_spreadsheet['CO2_1430_STD'][l]
            H2Om_5200_M_STD = mega_spreadsheet['H2Om_5200_STD'][l]
            OH_4500_M_STD = mega_spreadsheet['OH_4500_STD'][l]
            density_sat = density_df['Density'][l]

        elif Volatiles_DF['H2OT_3550_SAT?'][l] == '*':
            for m in range(20):
                H2Om_1635_BP = Beer_Lambert(molar_mass['H2O'], Volatiles_DF['PH_1635_BP'][l], Volatiles_DF['PH_1635_STD'][l],
                    density[l], density[l] * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_H2Om_1635'][l], epsilon['sigma_epsilon_H2Om_1635'][l])
                OH_4500_M = Beer_Lambert(molar_mass['H2O'], Volatiles_DF['PH_4500_M'][l], Volatiles_DF['PH_4500_STD'][l],
                    density[l], density[l] * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_OH_4500'][l], epsilon['sigma_epsilon_OH_4500'][l])
                MI_Composition['H2O'][l] = H2Om_1635_BP + OH_4500_M
                mol_sat, density_sat = Density_Calculation(MI_Composition)
            density_sat = density_sat[l]

            H2OT_3550_M = Beer_Lambert(molar_mass['H2O'], Volatiles_DF['PH_3550_M'][l], Volatiles_DF['PH_3550_STD'][l],
                density_sat, density_sat * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_H2OT_3550'][l], epsilon['sigma_epsilon_H2OT_3550'][l])
            H2Om_1635_BP = Beer_Lambert(molar_mass['H2O'], Volatiles_DF['PH_1635_BP'][l], Volatiles_DF['PH_1635_STD'][l],
                density_sat, density_sat * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_H2Om_1635'][l], epsilon['sigma_epsilon_H2Om_1635'][l])
            CO2_1515_BP = Beer_Lambert(molar_mass['CO2'], Volatiles_DF['PH_1515_BP'][l], Volatiles_DF['PH_1515_STD'][l],
                density_sat, density_sat * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_CO2'][l], epsilon['sigma_epsilon_CO2'][l])
            CO2_1430_BP = Beer_Lambert(molar_mass['CO2'], Volatiles_DF['PH_1430_BP'][l], Volatiles_DF['PH_1430_STD'][l],
                density_sat, density_sat * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_CO2'][l], epsilon['sigma_epsilon_CO2'][l])
            H2Om_5200_M = Beer_Lambert(molar_mass['H2O'], Volatiles_DF['PH_5200_M'][l], Volatiles_DF['PH_5200_STD'][l],
                density_sat, density_sat * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_H2Om_5200'][l], epsilon['sigma_epsilon_H2Om_5200'][l])
            OH_4500_M = Beer_Lambert(molar_mass['H2O'], Volatiles_DF['PH_4500_M'][l], Volatiles_DF['PH_4500_STD'][l],
                density_sat, density_sat * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_OH_4500'][l], epsilon['sigma_epsilon_OH_4500'][l])
            CO2_1515_BP *= 10000
            CO2_1430_BP *= 10000
            
            H2OT_3550_M_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_DF['PH_3550_M'][l], Volatiles_DF['PH_3550_STD'][l],
                density_sat, density_sat * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_H2OT_3550'][l], epsilon['sigma_epsilon_H2OT_3550'][l])
            H2Om_1635_BP_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_DF['PH_1635_BP'][l], Volatiles_DF['PH_1635_STD'][l],
                density_sat, density_sat * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_H2Om_1635'][l], epsilon['sigma_epsilon_H2Om_1635'][l])
            CO2_1515_BP_STD = Beer_Lambert_Error(N, molar_mass['CO2'], Volatiles_DF['PH_1515_BP'][l], Volatiles_DF['PH_1515_STD'][l],
                density_sat, density_sat * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_CO2'][l], epsilon['sigma_epsilon_CO2'][l])
            CO2_1430_BP_STD = Beer_Lambert_Error(N, molar_mass['CO2'], Volatiles_DF['PH_1430_BP'][l], Volatiles_DF['PH_1430_STD'][l],
                density_sat, density_sat * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_CO2'][l], epsilon['sigma_epsilon_CO2'][l])
            H2Om_5200_M_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_DF['PH_5200_M'][l], Volatiles_DF['PH_5200_STD'][l],
                density_sat, density_sat * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_H2Om_5200'][l], epsilon['sigma_epsilon_H2Om_5200'][l])
            OH_4500_M_STD = Beer_Lambert_Error(N, molar_mass['H2O'], Volatiles_DF['PH_4500_M'][l], Volatiles_DF['PH_4500_STD'][l],
               density_sat, density_sat * 0.025, thickness['Thickness'][l], sigma_thickness, epsilon['epsilon_OH_4500'][l], epsilon['sigma_epsilon_OH_4500'][l])
            CO2_1515_BP_STD *= 10000
            CO2_1430_BP_STD *= 10000

        density_sat_df.loc[l] = pd.Series({'Density_Sat': density_sat})
        mega_spreadsheet_sat.loc[l] = pd.Series({'H2OT_3550_M': H2OT_3550_M, 'H2OT_3550_SAT': Volatiles_DF['H2OT_3550_SAT?'][l], 'H2OT_3550_STD': H2OT_3550_M_STD, 
            'H2Om_1635_BP': H2Om_1635_BP, 'H2Om_1635_STD': H2Om_1635_BP_STD, 
            'CO2_1515_BP': CO2_1515_BP, 'CO2_1515_STD': CO2_1515_BP_STD, 'CO2_1430_BP': CO2_1430_BP, 'CO2_1430_STD': CO2_1430_BP_STD, 
            'H2Om_5200_M': H2Om_5200_M, 'H2Om_5200_STD': H2Om_5200_M_STD, 'OH_4500_M': OH_4500_M, 'OH_4500_STD': OH_4500_M_STD})
        s2nerror.loc[l] = pd.Series({'PH_5200_S2N': Volatiles_DF['S2N_P5200'][l], 'PH_4500_S2N': Volatiles_DF['S2N_P4500'][l], 'ERR_5200': Volatiles_DF['ERR_5200'][l], 'ERR_4500': Volatiles_DF['ERR_4500'][l]})

    mega_spreadsheet_f = pd.concat([mega_spreadsheet_sat, s2nerror], axis = 1)
    density_epsilon = pd.concat([density_df, density_sat_df, epsilon], axis = 1)

    for m in mega_spreadsheet.index: 
        if mega_spreadsheet['H2OT_3550_SAT'][m] == '*': 
            H2O_mean = mega_spreadsheet['H2Om_1635_BP'][m] + mega_spreadsheet['OH_4500_M'][m]
            H2O_std = np.sqrt((mega_spreadsheet['H2Om_1635_STD'][m]**2) + (mega_spreadsheet['OH_4500_STD'][m]**2)) / 2

        elif mega_spreadsheet['H2OT_3550_SAT'][m] == '-': 
            H2O_mean = mega_spreadsheet['H2OT_3550_M'][m] 
            H2O_std = mega_spreadsheet['H2OT_3550_STD'][m] 
        mean_vol.loc[m] = pd.Series({'H2OT_MEAN': H2O_mean, 'H2OT_STD': H2O_std})
    mean_vol['CO2_MEAN'] = (mega_spreadsheet['CO2_1515_BP'] + mega_spreadsheet['CO2_1430_BP']) / 2
    mean_vol['CO2_STD'] = np.sqrt((mega_spreadsheet['CO2_1515_STD']**2) + (mega_spreadsheet['CO2_1430_STD']**2)) / 2

    mega_spreadsheet_f['H2OT_MEAN'] = mean_vol['H2OT_MEAN']
    mega_spreadsheet_f['H2OT_STD'] = mean_vol['H2OT_STD']
    mega_spreadsheet_f['CO2_MEAN'] = mean_vol['CO2_MEAN']
    mega_spreadsheet_f['CO2_STD'] = mean_vol['CO2_STD']

    return density_epsilon, mega_spreadsheet_f
