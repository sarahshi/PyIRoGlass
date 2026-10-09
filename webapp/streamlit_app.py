""" PyIRoGlass web app: wafer thickness from reflectance FTIR, and H2O-CO2 from a few transmission spectra. // @author: Sarah Shi """

import io
import sys
import inspect
import shutil
import warnings
import zipfile
import tempfile
import threading
import subprocess
from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import streamlit as st
import PyIRoGlass as pig

st.set_page_config(page_title="PyIRoGlass", page_icon="🌋", initial_sidebar_state=350,  # sidebar width, px
                   layout="wide")  # full browser width, so spectra, figures and tables render large
st.set_page_config(initial_sidebar_state="expanded")  # additive: keeps the width, and opens the sidebar

HERE = Path(__file__).resolve().parent
WORKER = HERE / "baseline_worker.py"
EXAMPLES = HERE.parent / "Inputs" / "COLAB_BINDER"
OXIDES = ["SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MnO", "MgO", "CaO", "Na2O", "K2O", "P2O5"]
MAX_SPECTRA = 5
N_EXAMPLES = 2  # example transmission spectra the H2O-CO2 tool runs
TOOLS = ["H₂O and CO₂ (transmission FTIR)", "Wafer thickness (reflectance FTIR)",
         "Glass density (composition)", "Molar absorptivity (composition)"]
SKIP_IN_ZIP = {"NPZTXTFILES", "PKLFILES"}  # MCMC chains and pickles: ~18 MB per spectrum
# plot_subtracted is new in PyIRoGlass 0.6.8; hide the option on older pinned versions.
HAS_PLOT_SUBTRACTED = "plot_subtracted" in inspect.signature(pig.calculate_baselines).parameters
# The template generators are also new in 0.6.8.
HAS_TEMPLATES = hasattr(pig, "create_transmission_template") and hasattr(pig, "create_reflectance_template")

# Refractive index and default wavenumber window (cm^-1) per phase.
PHASES = {
    "Olivine": dict(index=pig.reflectance_index_ol, comp="Forsterite content, XFo", default=0.80, wn=(2100, 2700)),
    "Orthopyroxene": dict(index=pig.reflectance_index_opx, comp="Mg/(Mg+Fe+Mn), XMg", default=0.80, wn=(2100, 2700)),
    "Clinopyroxene": dict(index=pig.reflectance_index_cpx, comp="Mg/(Mg+Fe+Mn), XMg", default=0.80, wn=(2100, 2700)),
    "Basaltic glass": dict(n=1.546, wn=(1700, 2850)),
    "Other (enter refractive index)": dict(n=1.60, wn=(2100, 2700)),
}

# Density calibrations (oxide partial molar volumes) in calculate_density; the H2O-CO2 tool uses LS.
DENSITY_MODELS = {"Lesher and Spera (2015)": "LS", "Iacovino and Till (2019)": "IT"}
DENSITY_SIGMA = 0.025  # relative 1σ on density that calculate_concentrations propagates
# Display labels for calculate_epsilon's columns; downloads keep PyIRoGlass's names.
EPSILON_LABELS = {
    "H2Ot_3550": "H₂Oₜ 3550", "H2Om_1635": "H₂Oₘ 1635", "carbonate": "CO₃²⁻ 1515, 1430",
    "CO2": "CO₂ 2350", "H2Om_5200": "H₂Oₘ 5200", "OH_4500": "OH⁻ 4500",
}

CITATION = (
    "Shi, S., Towbin, W. H., Plank, T., Barth, A., Rasmussen, D., Moussallam, Y., Lee, H. J. and Menke, W. (2024) "
    "PyIRoGlass: An open-source, Bayesian MCMC algorithm for fitting baselines to FTIR spectra of "
    "basaltic-andesitic glasses. Volcanica, 7(2), pp. 471-501. doi: 10.30909/vol.07.02.471501."
)
BIBTEX = (
    "@article{Shietal2024,\n"
    "  doi     = {10.30909/vol.07.02.471501},\n"
    "  url     = {https://doi.org/10.30909/vol.07.02.471501},\n"
    "  year    = {2024},\n"
    "  volume  = {7},\n"
    "  number  = {2},\n"
    "  pages   = {471-501},\n"
    "  author  = {Shi, Sarah C. and Towbin, W. Henry and Plank, Terry and Barth, Anna and "
    "Rasmussen, Daniel and Moussallam, Yves and Lee, Hyun Joo and Menke, William},\n"
    "  title   = {PyIRoGlass: An open-source, Bayesian MCMC algorithm for fitting baselines to "
    "FTIR spectra of basaltic-andesitic glasses},\n"
    "  journal = {Volcanica}\n"
    "}"
)


# %% ----------------------------------------------------------------
# helpers

@st.cache_resource
def plot_lock():
    """pyplot is global state; serialize the (fast) thickness plotting across sessions."""
    return threading.Lock()


@st.cache_resource
def job_lock():
    """One MCMC job at a time: each uses several CPU cores for about a minute per spectrum."""
    return threading.Lock()


def uploaded_to_files(uploaded):
    return tuple((f.name, f.getvalue()) for f in uploaded)


def example_files(folder):
    return tuple((p.name, p.read_bytes()) for p in sorted(folder.glob("*")) if p.is_file())


def write_files(files, folder):
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files:
        (folder / name).write_bytes(data)


def sample_name(filename):
    return filename[:-4]  # SampleDataLoader drops the 3-letter extension


def switch_tool(name):
    """Button callback: selects another tool in the sidebar menu (callbacks run before the rerun draws it)."""
    st.session_state["tool"] = name


def template_csv(create, names, **kwargs):
    """Runs a PyIRoGlass template generator for these sample names in a temporary directory; returns CSV bytes."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "template.csv"
        create(dict.fromkeys(names), export_path=str(path), **kwargs)  # only the keys (sample names) are used
        return path.read_bytes()


def example_chemthick_csv():
    """The example ChemThick rows for the example spectra the H2O-CO2 tool runs, as CSV bytes."""
    names = [sample_name(p.name) for p in sorted((EXAMPLES / "TransmissionSpectra").glob("*")) if p.is_file()]
    chemthick = pd.read_csv(EXAMPLES / "Colab_Binder_ChemThick.csv", encoding="utf-8-sig").set_index("Sample")
    return chemthick.loc[names[:N_EXAMPLES]].to_csv().encode()


def fig_to_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    return buf.getvalue()


def pdf_to_png(path):
    import pymupdf
    with pymupdf.open(path) as doc:
        return doc[0].get_pixmap(dpi=90).tobytes("png")


@st.cache_data(show_spinner=False, max_entries=16)
def run_thickness(files, n, wn_low, wn_high):
    with tempfile.TemporaryDirectory() as d:
        write_files(files, Path(d))
        dfs = pig.SampleDataLoader(spectrum_path=d).load_spectrum_directory(wn_high=wn_high, wn_low=wn_low)
    with plot_lock():
        before = set(plt.get_fignums())
        thickness = pig.calculate_mean_thickness(dfs, n, wn_high, wn_low, plotting=True)
        figs = [plt.figure(k) for k in plt.get_fignums() if k not in before]
        pngs = [(fig.axes[0].get_title(), fig_to_png(fig)) for fig in figs]
        for fig in figs:
            plt.close(fig)
    return thickness, pngs


def run_baselines(files, chemthick, T, P, ignore_nir):
    """Runs baseline_worker.py in a temporary directory, with a progress bar. Returns a results dict."""
    tmp = Path(tempfile.mkdtemp(prefix="pyiroglass_"))
    proc = None
    try:
        write_files(files, tmp / "spectra")
        chemthick.to_csv(tmp / "ChemThick.csv", index=False)
        names = [sample_name(n) for n, _ in files]

        progress = st.progress(0.0, text=f"Fitting baselines: 0 of {len(names)} spectra done")
        log = []
        proc = subprocess.Popen(
            [sys.executable, str(WORKER), "spectra", "ChemThick.csv", str(T), str(P), "1" if ignore_nir else "0"],
            cwd=tmp, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        done = 0
        for line in proc.stdout:
            log.append(line.rstrip())
            if line.startswith("DONE\t"):
                done += 1
                progress.progress(done / len(names), text=f"Fitting baselines: {done} of {len(names)} spectra done")
        proc.wait()
        progress.empty()

        final = tmp / "FINALDATA"
        conc_csv, peaks_csv = final / "RESULTS_H2OCO2.csv", final / "RESULTS_DF.csv"
        results = {
            "names": names,
            "log": "\n".join(log),
            "conc": pd.read_csv(conc_csv, index_col=0) if conc_csv.exists() else None,
            "peaks": pd.read_csv(peaks_csv, index_col=0) if peaks_csv.exists() else None,
            "figures": [(p.stem, pdf_to_png(p)) for p in sorted((tmp / "FIGURES" / "RESULTS").glob("*.pdf"))],
            # Same figures with the baseline-subtracted carbonate peaks (PyIRoGlass 0.6.8+), for the display toggle
            "figures_subtracted": [(p.stem, pdf_to_png(p))
                                   for p in sorted((tmp / "FIGURES" / "RESULTS_SUBTRACTED").glob("*.pdf"))],
        }
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(tmp.rglob("*")):
                rel = p.relative_to(tmp)
                if p.is_file() and rel.parts[0] not in SKIP_IN_ZIP | {"spectra"}:
                    z.write(p, str(rel))
        results["zip"] = buf.getvalue()
        return results
    finally:
        # If the visitor leaves mid-fit, Streamlit stops this script; don't leave the worker running.
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        shutil.rmtree(tmp, ignore_errors=True)


def chemthick_table(names, source, report_unmatched=True):
    """Editable chemistry and thickness table, one row per spectrum, prefilled from a ChemThick CSV if given."""
    base = pd.DataFrame({"Sample": names}).set_index("Sample")
    for col in OXIDES + ["Thickness", "Sigma_Thickness"]:
        base[col] = float("nan")
    if source is not None:
        src = pd.read_csv(source, encoding="utf-8-sig").set_index("Sample")
        common = base.index.intersection(src.index)
        cols = [c for c in base.columns if c in src.columns]
        base.loc[common, cols] = src.loc[common, cols].astype(float)
        missing = base.index.difference(src.index).tolist()
        unmatched = src.index.difference(base.index).tolist()
        if missing:
            st.warning(f"No matching `Sample` in your ChemThick file for these spectra (fill in below): {missing}")
        if unmatched and report_unmatched:
            st.caption(f"Rows in your ChemThick file with no matching spectrum (ignored; check for typos if a "
                       f"spectrum above is blank): {unmatched}")
    return st.data_editor(
        base.reset_index(), hide_index=True, key=f"chemthick_{hash(tuple(names))}",
        column_config={"Sample": st.column_config.TextColumn(disabled=True),
                       "Thickness": st.column_config.NumberColumn("Thickness (µm)"),
                       "Sigma_Thickness": st.column_config.NumberColumn("σ Thickness (µm)")},
    )


def composition_table(source, key, extra=()):
    """Editable glass composition table, one row per glass, prefilled from a CSV if given. Rows can be added."""
    cols = ["Sample"] + OXIDES + list(extra)
    if source is None:
        base = pd.DataFrame({"Sample": [f"Glass_{i}" for i in range(1, 4)]})
    else:
        base = pd.read_csv(source, encoding="utf-8-sig")
        if "Sample" not in base.columns:
            st.error("Your composition file needs a `Sample` column.")
            st.stop()
    base = base.reindex(columns=cols)
    base[cols[1:]] = base[cols[1:]].apply(pd.to_numeric, errors="coerce")
    return st.data_editor(
        base, hide_index=True, num_rows="dynamic", key=key,
        column_config={"Sample": st.column_config.TextColumn(),
                       "H2O": st.column_config.NumberColumn("H2O (wt%)") if "H2O" in extra else None,},
    )


def compositions(table):
    """The filled-in rows of a composition table, indexed by Sample, blanks as 0. None if there are none."""
    comp = table.dropna(subset=table.columns.drop("Sample"), how="all").copy()
    if comp.empty:
        return None
    blank = comp["Sample"].isna() | (comp["Sample"].astype(str).str.strip() == "")
    comp.loc[blank, "Sample"] = [f"Row_{i + 1}" for i in comp.index[blank]]
    comp = comp.set_index("Sample")
    dupes = comp.index[comp.index.duplicated()].unique().tolist()
    if dupes:
        st.error(f"Sample names must be unique: {dupes}")
        st.stop()
    return comp.fillna(0.0)


def calculate_densities(comp, T, P, models):
    """Density (kg/m³) at T and P for each glass and its 1σ, two columns per calibration, e.g. Density_LS."""
    density = pd.DataFrame()
    for m in models:
        code = DENSITY_MODELS[m]
        density[f"Density_{code}"] = pig.calculate_density(comp, T, P, model=code)[1]
        density[f"Sigma_Density_{code}"] = density[f"Density_{code}"] * DENSITY_SIGMA
    density.index.name = "Sample"
    return density


def calculate_epsilons(comp):
    """Molar absorptivities (L/mol·cm) for each glass. T and P don't enter calculate_epsilon's result."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # out-of-range compositions are listed in the Notes column
        epsilon = pig.calculate_epsilon(comp[OXIDES], 25, 1)
    epsilon.index.name = "Sample"
    return epsilon


def composition_source(use_example, uploaded, tool_key):
    """CSV source and data editor key for the composition tools. The examples are one row per glass."""
    if use_example:
        examples = pd.read_csv(EXAMPLES / "Colab_Binder_ChemThick.csv", encoding="utf-8-sig")
        return io.StringIO(examples.drop_duplicates(subset=OXIDES).to_csv(index=False)), f"{tool_key}_example"
    return uploaded, f"{tool_key}_{uploaded.file_id}" if uploaded else f"{tool_key}_blank"


# %% ----------------------------------------------------------------
# sidebar

with st.sidebar:
    st.header("Tool")
    tool = st.selectbox("What do you want to calculate?", TOOLS, key="tool")

    if tool.startswith("Wafer"):
        st.header("Spectra")
        use_example = st.checkbox("Use example spectra", value=False)
        uploaded = [] if use_example else st.file_uploader(
            "Reflectance spectra (CSV)", type=["csv", "txt"], accept_multiple_files=True,
            help="Two columns, wavenumber and absorbance, one spectrum per file. File names become sample names.",
        )
        st.header("Phase")
        phase = st.selectbox("Phase", list(PHASES))
        spec = PHASES[phase]
        if "index" in spec:
            x = st.number_input(spec["comp"], 0.0, 1.0, spec["default"], step=0.01, format="%.2f")
            n = spec["index"](x)
            st.caption(f"Refractive index n = {n:.4f} (Deer, Howie, and Zussman)")
        elif phase == "Basaltic glass":
            n = spec["n"]
            st.caption(f"Refractive index n = {n} (Nichols and Wysoczanski, 2007)")
        else:
            n = st.number_input("Refractive index, n", 1.0, 3.0, spec["n"], step=0.001, format="%.3f")
        wn_low, wn_high = st.slider("Wavenumber window (cm⁻¹)", 1000, 5000, spec["wn"], step=50)
    elif tool in TOOLS[2:]:
        st.header("Compositions")
        use_example = st.checkbox("Use example compositions", value=False)
        composition_file = None if use_example else st.file_uploader(
            "Composition file (optional CSV)", type=["csv"],
            help="Columns: Sample and oxides (wt%)"
                 + (", and optionally H2O (wt%)" if tool == TOOLS[2] else "")
                 + ". A ChemThick file works too; other columns are ignored. Or type or paste the values into "
                   "the table on the page.",
        )
        if tool == TOOLS[2]:
            st.header("Options")
            T = st.number_input("Temperature of analysis (°C)", value=25)
            P = st.number_input("Pressure of analysis (bar)", value=1)
            density_models = st.multiselect(
                "Density calibrations", list(DENSITY_MODELS), default=list(DENSITY_MODELS),
                help="Oxide partial molar volumes. Each calibration you pick gets its own column.",
            )
    else:
        st.header("Spectra")
        use_example = st.checkbox(f"Use example spectra ({N_EXAMPLES})", value=False)
        uploaded = [] if use_example else st.file_uploader(
            f"Transmission spectra (CSV, up to {MAX_SPECTRA})", type=["csv", "txt"], accept_multiple_files=True,
            help="Two columns, wavenumber and absorbance, spanning 1000-5500 cm⁻¹. File names become sample names.",
        )
        st.header("Chemistry and thickness")
        if use_example:
            chemthick_file = None
            st.download_button("Download example ChemThick file", example_chemthick_csv(),
                               file_name="Example_ChemThick.csv", mime="text/csv", key="chemthick_example")
            st.caption("The example spectra come with their chemistry and thickness. Download the file to see "
                       "what a filled-in ChemThick file looks like.")
        else:
            chemthick_file = st.file_uploader(
                "ChemThick file (optional CSV)", type=["csv"],
                help="Columns: Sample, oxides (wt%), Thickness and Sigma_Thickness (µm). Each Sample must match a "
                     "spectrum's file name without the extension (AC4_OL49_a.CSV → AC4_OL49_a). "
                     "Or fill in the table on the page.",
            )
            if HAS_TEMPLATES:
                # Always shown, disabled until spectra are uploaded, so the layout doesn't jump.
                st.download_button(
                    "Download ChemThick template",
                    template_csv(pig.create_transmission_template, [sample_name(f.name) for f in uploaded])
                    if uploaded else b"",
                    file_name="ChemThick_Template.csv", mime="text/csv", key="chemthick_template_sidebar",
                    disabled=not uploaded,
                )
                st.caption("No file yet? Download a template with the sample names filled in from your uploaded "
                           "spectra. Fill it in and upload it above.")
        st.header("Options")
        T = st.number_input("Temperature of analysis (°C)", value=25)
        P = st.number_input("Pressure of analysis (bar)", value=1)
        ignore_nir = st.checkbox("Ignore near-IR peaks (5200, 4500 cm⁻¹)", value=False,
                                 help="For spectra that do not reach 5500 cm⁻¹.")
        plot_subtracted = HAS_PLOT_SUBTRACTED and st.checkbox(
            "Show baseline-subtracted carbonate peaks", value=False,
            help="Plots the baseline-subtracted and modeled CO₃²⁻ 1515 and 1430 cm⁻¹ carbonate fits with posterior draws. "
                 "Switch it on or off any time after fitting.")

    st.header("About")
    st.markdown(
        "[Documentation](https://pyiroglass.readthedocs.io) · "
        "[GitHub](https://github.com/sarahshi/PyIRoGlass) · "
        "[Paper](https://doi.org/10.30909/vol.07.02.471501)"
    )
    st.caption(
        f"PyIRoGlass v{pig.__version__}. Questions, bugs, ideas, or devolatilized spectra to help shape "
        "the baseline? Email [sarahshi@berkeley.edu](mailto:sarahshi@berkeley.edu) or open a "
        "[GitHub issue](https://github.com/sarahshi/PyIRoGlass/issues)."
    )

# %% ----------------------------------------------------------------
# main

st.title("PyIRoGlass")
st.markdown(
    "Bayesian MCMC baselines for determining H₂O and CO₂ species concentrations in transmission FTIR "
    "spectra of basaltic to andesitic glasses, plus wafer thicknesses from reflectance FTIR spectra and "
    "glass densities and molar absorptivities from composition."
)

with st.expander("How to cite"):
    st.markdown("If you use PyIRoGlass in your work, please cite:")
    st.code(CITATION, language=None, wrap_lines=True)
    st.markdown("BibTeX:")
    st.code(BIBTEX, language="bibtex")

if tool.startswith("Wafer"):
    st.header("Wafer thickness")
    st.markdown(
        "Thickness from the interference fringes in reflectance FTIR spectra, using the refractive index "
        "of the phase and the spacing of fringe peaks and troughs. Choose the phase and wavenumber window "
        "in the menu at left. The defaults are 2100-2700 cm⁻¹ for minerals and 1700-2850 cm⁻¹ for glass."
    )
    files = example_files(EXAMPLES / "ReflectanceSpectra" / ("GL" if phase == "Basaltic glass" else "OL")) \
        if use_example else uploaded_to_files(uploaded)
    if not files:
        st.info("Upload reflectance spectra in the menu at left, or tick **Use example spectra**.")
        st.stop()

    with st.spinner("Calculating thicknesses..."):
        thickness, pngs = run_thickness(files, n, wn_low, wn_high)

    st.subheader("Results")
    st.dataframe(thickness, column_config={c: st.column_config.NumberColumn(format="%.2f") for c in thickness.columns})
    st.caption("Thickness_M and Thickness_STD (µm) use all fringe peaks and troughs, and are the recommended values.")
    d1, d2, _ = st.columns([1, 1.6, 1])
    d1.download_button("Download CSV", thickness.to_csv().encode(), file_name="PyIRoGlass_thickness.csv", mime="text/csv")
    if HAS_TEMPLATES:
        d2.download_button(
            "Download refractive index template",
            template_csv(pig.create_reflectance_template, [sample_name(name) for name, _ in files], default_n=n),
            file_name="RefractiveIndex_Template.csv", mime="text/csv",
        )
        n_rows = f"{len(files)} row{'' if len(files) == 1 else 's'}"
        st.caption(
            f"The refractive index template has {n_rows}, one for each spectrum, with the sample names filled in "
            f"and n prefilled as {n:.4f} (the value used above). For batches that mix phases, edit n for each spectrum and pass the file to "
            "`pig.calculate_mean_thickness` when running PyIRoGlass on your own computer; this page uses one n "
            "for all spectra."
        )

    st.subheader("Fringes")
    st.markdown("Baseline-subtracted spectra with the peaks (▲) and troughs (▼) used. Check that every fringe is picked.")
    cols = st.columns(2)
    for i, (title, png) in enumerate(pngs):
        cols[i % 2].image(png, caption=title)

elif tool == TOOLS[2]:
    st.header("Glass density")
    st.markdown(
        "Glass density at the temperature and pressure of analysis, from the oxide partial molar volumes. Pick one "
        "or more calibrations in the menu at left to compare them. The H₂O and CO₂ tool uses Lesher and Spera (2015)."
    )
    st.markdown(
        "Enter one glass per row in wt%, or upload a CSV in the menu at left. Add rows at the bottom of the table, "
        "or paste from a spreadsheet. Blank oxides count as 0."
    )
    st.button("Need molar absorptivities? Use the Molar absorptivity tool", type="tertiary",
              icon=":material/arrow_forward:", on_click=switch_tool, args=(TOOLS[3],))
    comp = compositions(composition_table(*composition_source(use_example, composition_file, "density"),
                                          extra=["H2O"]))
    if comp is None:
        st.info("Enter a composition in the table above, upload a composition file, or tick **Use example compositions**.")
        st.stop()
    if not density_models:
        st.info("Pick at least one density calibration in the menu at left.")
        st.stop()

    density = calculate_densities(comp, T, P, density_models)
    st.subheader("Results")
    density_cols = {}
    for m in density_models:
        density_cols[f"Density_{DENSITY_MODELS[m]}"] = st.column_config.NumberColumn(f"{m} (kg/m³)", format="%.1f")
        density_cols[f"Sigma_Density_{DENSITY_MODELS[m]}"] = st.column_config.NumberColumn(f"σ {m}", format="%.1f")
    st.dataframe(density, column_config=density_cols)
    st.caption(
        f"Density in kg/m³ at {T} °C and {P} bar. σ is the {DENSITY_SIGMA:.1%} (1σ) density uncertainty that the H₂O "
        "and CO₂ tool propagates into concentrations, not an error calculated from the composition. For basaltic "
        "glasses the two calibrations differ by about 2%, a larger effect than analytical errors in the composition."
    )
    st.download_button("Download CSV", density.to_csv().encode(), file_name="PyIRoGlass_density.csv",
                       mime="text/csv")

elif tool == TOOLS[3]:
    st.header("Molar absorptivity")
    st.markdown(
        "Molar absorptivities (ε) of each H₂O and CO₂ species from the glass composition, using the compositional "
        "regressions of Shi et al. (2024). These are the values the H₂O and CO₂ tool uses in the Beer-Lambert law, "
        "so you can check them."
    )
    st.markdown(
        "Enter one glass per row in wt% (anhydrous), or upload a CSV in the menu at left. Add rows at the bottom of "
        "the table, or paste from a spreadsheet. Blank oxides count as 0."
    )
    st.button("Need glass densities? Use the Glass density tool", type="tertiary",
              icon=":material/arrow_forward:", on_click=switch_tool, args=(TOOLS[2],))
    comp = compositions(composition_table(*composition_source(use_example, composition_file, "epsilon")))
    if comp is None:
        st.info("Enter a composition in the table above, upload a composition file, or tick **Use example compositions**.")
        st.stop()

    epsilon = calculate_epsilons(comp)
    st.subheader("Results")
    eps_cols = {}
    for name, label in EPSILON_LABELS.items():
        eps_cols[f"epsilon_{name}"] = st.column_config.NumberColumn(f"ε {label}", format="%.2f")
        eps_cols[f"sigma_epsilon_{name}"] = st.column_config.NumberColumn(f"σ ε {label}", format="%.2f")
    st.dataframe(epsilon, column_config={
        "Tau": st.column_config.NumberColumn("τ", format="%.4f"),
        "Eta": st.column_config.NumberColumn("η", format="%.4f"),
        **eps_cols,
    })
    st.caption(
        "Molar absorptivities in L/mol·cm with 1σ uncertainties, at the peak wavenumbers in cm⁻¹. "
        "τ = (Si⁴⁺ + Al³⁺)/total cations sets ε for the H₂O peaks; η = Na⁺/(Na⁺ + Ca²⁺) sets ε for the carbonate "
        "peaks. Molecular CO₂ at 2350 cm⁻¹ uses a constant 830 ± 5%."
    )
    flagged = epsilon.loc[epsilon["Notes"].astype(bool), "Notes"]
    if not flagged.empty:
        st.warning(
            "**Outside the calibration range; use these molar absorptivities with caution:**\n\n"
            + "\n".join(f"- `{sample}`: {note}" for sample, note in flagged.items())
        )
    st.download_button("Download CSV", epsilon.to_csv().encode(), file_name="PyIRoGlass_epsilon.csv",
                       mime="text/csv")

else:
    st.header("H₂O and CO₂ from transmission FTIR")
    st.warning(
        f"**Use this for a few spectra only (up to {MAX_SPECTRA}).** Each spectrum runs a full MCMC fit of the "
        "baseline, about 20 s on a laptop and roughly a minute on this web server, and only one fit runs at "
        "a time for all visitors. For larger datasets, run PyIRoGlass on "
        "[VICTOR](https://hub.victorproject.org/hub/login?next=%2Fhub%2F), "
        "[Google Colab](https://colab.research.google.com/github/SarahShi/PyIRoGlass/blob/main/PyIRoGlass_RUN_colab.ipynb), "
        "or your own computer (`pip install PyIRoGlass`)."
    )

    if use_example:
        files = example_files(EXAMPLES / "TransmissionSpectra")[:N_EXAMPLES]
        chemthick_source = EXAMPLES / "Colab_Binder_ChemThick.csv"
    else:
        files, chemthick_source = uploaded_to_files(uploaded), chemthick_file
    if not files:
        st.info("Upload transmission spectra in the menu at left, or tick **Use example spectra**.")
        st.stop()
    if len(files) > MAX_SPECTRA:
        st.error(f"{len(files)} spectra uploaded. The web version runs up to {MAX_SPECTRA} at a time; "
                 "use VICTOR, Colab, or your own computer for more.")
        st.stop()

    names = [sample_name(name) for name, _ in files]
    st.subheader("Chemistry and thickness")
    st.markdown("Glass composition (wt%) and wafer thickness (µm) for each spectrum, used for density and "
                "molar absorptivities.")
    st.button("Have reflectance spectra? Get thicknesses with the Wafer thickness tool", type="tertiary",
              icon=":material/arrow_forward:", on_click=switch_tool, args=(TOOLS[1],))
    st.info(
        "**Sample names must match.** Each row is matched to a spectrum by name: the `Sample` column must equal "
        "the spectrum's file name without the extension, e.g. `AC4_OL49_021920_30x30_H2O_a.CSV` → "
        "`AC4_OL49_021920_30x30_H2O_a`. If you upload a ChemThick CSV, rows are filled in by this name; any "
        "spectrum without a match is left blank below for you to fill in."
    )
    if HAS_TEMPLATES:
        st.markdown("**No ChemThick file yet?** Download this template: the sample names are already filled in.")
        st.download_button(
            "Download ChemThick template",
            template_csv(pig.create_transmission_template, names),
            file_name="ChemThick_Template.csv", mime="text/csv", key="chemthick_template_page",
        )
        n_rows = f"{len(names)} row{'' if len(names) == 1 else 's'}"
        st.caption(
            f"It has {n_rows}, one for each spectrum above, and empty columns for the 11 major oxides, Thickness, "
            "and Sigma_Thickness. "
            + ("Fill it in and upload it under **Chemistry and thickness** in the menu at left, "
               "or type the values into the table below." if not use_example else
               "The example spectra already have their chemistry and thickness filled in below.")
        )
    chemthick = chemthick_table(names, chemthick_source, report_unmatched=not use_example)

    missing_thickness = chemthick.loc[~(chemthick["Thickness"] > 0), "Sample"].tolist()
    if missing_thickness:
        st.info(f"Enter a thickness for: {missing_thickness}")
    run = st.button(f"Fit {len(files)} spectr{'um' if len(files) == 1 else 'a'}", type="primary",
                    disabled=bool(missing_thickness))

    if run:
        lock = job_lock()
        if not lock.acquire(blocking=False):
            st.error("Another fit is running right now. Please try again in a few minutes.")
            st.stop()
        try:
            st.session_state["pig_results"] = run_baselines(
                files, chemthick.fillna({c: 0.0 for c in OXIDES}), T, P, ignore_nir)
        finally:
            lock.release()

    res = st.session_state.get("pig_results")
    if res and res["names"] == names:
        failed = [s for s in res["names"] if res["peaks"] is None or s not in res["peaks"].index]
        if failed:
            st.error(f"Fitting failed for: {failed}. See the log below.")
        if res["peaks"] is not None and not res["peaks"].empty:
            st.subheader("Peak heights")
            ph_cols = ["PH_3550_M", "PH_3550_STD", "PH_1635_BP", "PH_1635_STD", "PH_1515_BP", "PH_1515_STD",
                       "PH_1430_BP", "PH_1430_STD", "PH_5200_M", "PH_5200_STD", "PH_4500_M", "PH_4500_STD"]
            st.dataframe(res["peaks"][[c for c in ph_cols if c in res["peaks"].columns]],
                         column_config={c: st.column_config.NumberColumn(format="%.4f") for c in ph_cols})
            st.caption("Best-fit peak heights and standard deviations (absorbance) for H₂Oₜ at 3550, H₂Oₘ at 1635, "
                       "CO₃²⁻ at 1515 and 1430, H₂Oₘ at 5200, and OH⁻ at 4500 cm⁻¹. Baseline and peak parameters are "
                       "in the downloads.")
            st.download_button("Download peak heights CSV", res["peaks"].to_csv().encode(),
                               file_name="PyIRoGlass_peak_heights.csv", mime="text/csv")
        if res["conc"] is not None:
            st.subheader("Concentrations")
            key_cols = ["H2Ot_MEAN", "H2Ot_STD", "CO2_MEAN", "CO2_STD", "H2Om_1635_BP", "H2Om_1635_STD",
                        "H2Om_5200_M", "OH_4500_M", "Density"]
            st.dataframe(res["conc"][[c for c in key_cols if c in res["conc"].columns]],
                         column_config={c: st.column_config.NumberColumn(format="%.3f") for c in key_cols})
            st.caption("H₂O species in wt%, CO₂ in ppm, density in kg/m³. All columns, including molar "
                       "absorptivities, are in the downloads.")
            d1, d2, _ = st.columns([1.3, 1.3, 1])
            d1.download_button("Download concentrations CSV", res["conc"].to_csv().encode(),
                               file_name="PyIRoGlass_H2OCO2.csv", mime="text/csv")
            d2.download_button("Download all outputs (zip)", res["zip"], file_name="PyIRoGlass_outputs.zip",
                               mime="application/zip",
                               help="Concentrations, peak heights, figures, MCMC diagnostics, and logs.")
        # Both versions are made during the fit, so the sidebar toggle switches figures without refitting
        figures = res["figures_subtracted"] if plot_subtracted and res.get("figures_subtracted") else res["figures"]
        if figures:
            st.subheader("Baseline fits")
            for title, png in figures:
                st.image(png, caption=title)
        with st.expander("Log"):
            st.code(res["log"] or "(empty)", language=None)
