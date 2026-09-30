""" PyIRoGlass web app worker: MCMC baselines and H2O-CO2 concentrations for a few spectra. // @author: Sarah Shi """

# Run as a separate process by streamlit_app.py, inside a temporary directory, because
# calculate_baselines writes its outputs to the current working directory and uses pyplot.
# Prints one "DONE <sample>" line per spectrum so the app can show progress.
#
# Usage: python baseline_worker.py SPECTRA_DIR CHEMTHICK_CSV T P IGNORE_NIR

import os
import sys
import pickle
import inspect
import warnings

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PyIRoGlass as pig

EXPORT = "RESULTS"
SUBTRACTED_FIGURES = os.path.join("FIGURES", f"{EXPORT}_SUBTRACTED")
# plot_subtracted is new in PyIRoGlass 0.6.8.
HAS_PLOT_SUBTRACTED = "plot_subtracted" in inspect.signature(pig.calculate_baselines).parameters


def summary_figure(name, spectrum, als_bls, mc3_output, ignore_nir, plot_subtracted):
    """Draws a spectrum's summary figure with the same layout calculate_baselines uses. Returns (fig, panel axes)."""
    if ignore_nir:
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        pig.plot_H2Ot_3550(spectrum, name, als_bls, ax=ax[0])
        axes = list(ax)
    else:
        fig = plt.figure(figsize=(26, 8))
        axes = [plt.subplot2grid((2, 3), (0, 0), fig=fig), plt.subplot2grid((2, 3), (1, 0), fig=fig),
                plt.subplot2grid((2, 3), (0, 1), rowspan=2, fig=fig),
                plt.subplot2grid((2, 3), (0, 2), rowspan=2, fig=fig)]
        pig.plot_H2Om_OH(spectrum, name, als_bls, ax_top=axes[0], ax_bot=axes[1])
        pig.plot_H2Ot_3550(spectrum, name, als_bls, ax=axes[2])
    kwargs = {"plot_subtracted": True} if plot_subtracted else {}
    pig.plot_carbonate(spectrum, name, mc3_output, None, ax=axes[-1], **kwargs)
    return fig, axes


def plot_subtracted_figure(name, spectrum, ignore_nir):
    """
    Redraws a spectrum's summary figure with and without the baseline-subtracted carbonate peaks below the
    carbonate fit, from the fit calculate_baselines just saved (no new MCMC), so the app can switch between them.
    Both use the panel positions of the version with the peaks, which needs room for the longer axis label, so
    switching doesn't shift the panels. The version without replaces the figure calculate_baselines saved.
    """
    with open(os.path.join("PKLFILES", EXPORT, f"{name}.pkl"), "rb") as handle:
        als_bls = pickle.load(handle)
    mc3_output = dict(np.load(os.path.join("NPZTXTFILES", EXPORT, f"{name}.npz")))

    fig, axes = summary_figure(name, spectrum, als_bls, mc3_output, ignore_nir, plot_subtracted=True)
    fig.tight_layout()
    positions = [a.get_position(original=True) for a in axes]  # carbonate: the area before splitting off the peaks
    os.makedirs(SUBTRACTED_FIGURES, exist_ok=True)
    fig.savefig(os.path.join(SUBTRACTED_FIGURES, f"{name}.pdf"))
    plt.close(fig)

    fig, axes = summary_figure(name, spectrum, als_bls, mc3_output, ignore_nir, plot_subtracted=False)
    for a, position in zip(axes, positions):
        a.set_position(position)
    fig.savefig(os.path.join("FIGURES", EXPORT, f"{name}.pdf"))
    plt.close("all")


def main(spectra_dir, chemthick_csv, T, P, ignore_nir):
    warnings.filterwarnings("ignore")
    loader = pig.SampleDataLoader(spectra_dir, chemthick_csv)
    dfs_dict, chemistry, thickness = loader.load_all_data()

    peak_heights = []
    for name, spectrum in dfs_dict.items():
        out, failures = pig.calculate_baselines({name: spectrum}, EXPORT, ignore_NIR=ignore_nir)
        peak_heights.append(out)
        if HAS_PLOT_SUBTRACTED and not failures:
            try:
                plot_subtracted_figure(name, spectrum, ignore_nir)
            except Exception as e:  # the main figure and results are already saved; keep going
                print(f"{name}: subtracted carbonate figure failed. Reason: {e}", flush=True)
        print(f"DONE\t{name}", flush=True)

    # Each call above overwrote FINALDATA/RESULTS_DF.csv with one spectrum; write them all.
    volatile_ph = pd.concat(peak_heights)
    volatile_ph.to_csv(f"FINALDATA/{EXPORT}_DF.csv")
    if volatile_ph.empty:
        print("FAILED\tall spectra", flush=True)
        return

    pig.calculate_concentrations(
        volatile_ph, chemistry.loc[volatile_ph.index], thickness.loc[volatile_ph.index],
        EXPORT, T=T, P=P, ignore_NIR=ignore_nir,
    )
    print("CONCENTRATIONS", flush=True)


if __name__ == "__main__":
    spectra_dir, chemthick_csv, T, P, ignore_nir = sys.argv[1:6]
    main(spectra_dir, chemthick_csv, float(T), float(P), ignore_nir == "1")
