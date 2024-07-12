==========
Change Log
==========


Version 0.6.3
=============
Return UserWarning for compositions outside of the calibration ranges for the epsilon absorption coefficients in function calculate_concentrations and calculate_epsilon, UserWarning for when data do not span the full wavenumber range of 1000-5500 cm^-1 whilst using class SampleDataLoader. Thanks to Dr. Shuo Ding and Emilia Pelegano-Titmuss for identifying common errors during test usage, which contributed to these improvements.


Version 0.6.2
=============
Return ValueError for mismatched samples in peak height and composition DataFrames in function calculate_concentrations, UserWarning for missing data columns in composition DataFrame in class SampleDataLoader. Thanks to Dr. Shuo Ding for identifying common errors during test usage, which contributed to these improvements.


Version 0.6.1
=============
Update composition-dependent epsilon inversions to add Shi et al., 2024 values for H2Ot, 3550 and carbonate, slight correction. Correct data export paths in calculate_baselines and calculate_concentrations. Update PC vector creation.


Version 0.5.2
=============
Account for variability in export_path naming practice. Remove remove_baseline argument from thickness functions.


Version 0.5.1
=============
Updating molar absorptivity inversions to add Shi et al., 2024 values for H2Ot, 3550 and carbonate. Update data export paths to be more sensible within the calculate_baselines and calculate_concentrations functions. Add to the inversion.py functionality for calculating statistics. 


Version 0.4.2
=============
Streamline exports from calculate_concentrations function, merge sheets. Working through reviewer comments from Volcanica!


Version 0.4.1
=============
Updating functions loading data to be in object-oriented structure. Streamlining and separating out plotting functions from remaining code. Removing excess variables. Renaming for consistency with Python function guidance. Correcting for pep8. Working through reviewer comments from Volcanica!


Version 0.3.1
=============
Updating molar absorptivity inversion to add Brounce et al., 2021 value for carbonate. Paper submitted to Volcanica!


Version 0.2.2
=============
Accidental deletion of GitHub commits, this restores all commits to v0.2.1. .git file size reduced. 


Version 0.2.1
=============
Minimal changes to clean code, fix UnitTesting, and prepare for publication. 


Version 0.2.0
=============
Update parameter estimation regions with the guidance of devolatilized spectra, improve functionality. 


Version 0.1.0
=============
Update version on PyPi to be compatible with mc3 v3.1.2.


Version 0.0.1
=============
Update version on PyPi to fix .npz read. 


Version 0.0.0
=============
First version on PyPi. 




