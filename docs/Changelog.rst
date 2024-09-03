==========
Change Log
==========


Version 0.6.4
=============
Update output directory generation, correct density model implementation. Updated `ReadtheDocs <https://pyiroglass.readthedocs.io/en/latest/>`_ to account for these new functions and warnings. Rename ``composition`` parameters as ``chemistry`` for consistency.


Version 0.6.3
=============
Return ``UserWarning`` for compositions outside of the calibration ranges for the epsilon (molar absorptivities) in function :func:`pig.calculate_concentrations` and :func:`pig.calculate_epsilon`, ``UserWarning`` for when data do not span the full wavenumber range of 1000-5500 cm\ :sup:`-1` whilst using class :class:`pig.SampleDataLoader`. Thanks to Dr. Shuo Ding and Emilia Pelegano-Titmuss for identifying common errors during test usage, which contributed to these improvements.


Version 0.6.2
=============
Return ``ValueError`` for mismatched samples in peak height and composition DataFrames in function :func:`pig.calculate_concentrations`, ``UserWarning`` for missing data columns in composition DataFrame in class :class:`pig.SampleDataLoader`. Thanks to Dr. Shuo Ding for identifying common errors during test usage, which contributed to these improvements.


Version 0.6.1
=============
Update composition-dependent epsilon inversions to add :cite:t:`Shietal2024` values for :math:`\text{H}_2\text{O}_{t, 3550}` and :math:`\text{CO}_3^{2-}`, slight correction. Correct data export paths in :func:`pig.calculate_baselines` and :func:`pig.calculate_concentrations`. Update PC vector creation.


Version 0.5.2
=============
Account for variability in ``export_path`` naming practice. Remove ``remove_baseline`` argument from thickness functions.


Version 0.5.1
=============
Updating molar absorptivity inversions to add :cite:t:`Shietal2024` values for :math:`\text{H}_2\text{O}_{t, 3550}` and :math:`\text{CO}_3^{2-}`. Update data export paths to be more sensible within the functions :func:`pig.calculate_baselines` and :func:`pig.calculate_concentrations`. Add to the inversion.py functionality for calculating statistics. 


Version 0.4.2
=============
Streamline exports from function :func:`pig.calculate_concentrations`, merge sheets. Working through reviewer comments from Volcanica! Thanks to Daniel Lee for identifying a to_csv bug during test usage.


Version 0.4.1
=============
Updating functions loading data to be in object-oriented structure. Streamlining and separating out plotting functions from remaining code. Removing excess variables. Renaming for consistency with `Python <https://www.python.org/>`_ function guidance. Correcting for `pep8 <https://peps.python.org/pep-0008/>`_. Working through reviewer comments!


Version 0.3.1
=============
Updating epsilon (molar absorptivity) inversion to add :cite:t:`Brounceetal2021` value for :math:`\text{CO}_3^{2-}`. Paper submitted to Volcanica!


Version 0.2.2
=============
Accidental deletion of `GitHub <https://github.com/sarahshi/PyIRoGlass>`_ commits, this restores all commits to v0.2.1. .git file size reduced. 


Version 0.2.1
=============
Minimal changes to clean code, fix unit testing, and prepare for publication. 


Version 0.2.0
=============
Update parameter estimation regions with the guidance of devolatilized spectra, improve functionality. 


Version 0.1.0
=============
Update version on `PyPi <https://pypi.org/project/PyIRoGlass/>`_ to be compatible with ``mc3`` v3.1.2.


Version 0.0.1
=============
Update version on `PyPi <https://pypi.org/project/PyIRoGlass/>`_ to fix .npz read. 


Version 0.0.0
=============
First version on `PyPi <https://pypi.org/project/PyIRoGlass/>`_. 

