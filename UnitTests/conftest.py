import pytest


@pytest.fixture(autouse=True)
def run_in_tmp_path(tmp_path, monkeypatch):
    """
    Runs every test in its own temporary working directory.
    calculate_baselines and calculate_concentrations write FINALDATA/,
    FIGURES/, etc. into the working directory, and test_MCMC_exportpath
    deletes FINALDATA/ afterwards, so running the suite from the repository
    root would otherwise overwrite or delete real outputs. Tests locate their
    input files relative to __file__, so they are unaffected.
    """
    monkeypatch.chdir(tmp_path)
