PyIRoGlass Documentation
========================


Data Imports 
============

.. autofunction:: PyIRoGlass.Load_SampleCSV
   :members:

.. autofunction:: PyIRoGlass.Load_PCA
   :members:

.. autofunction:: PyIRoGlass.Load_Wavenumber
   :members:

.. autofunction:: PyIRoGlass.Load_ChemistryThickness
   :members:


Building-blocks functions for fitting baselines and peaks
=========================================================

.. autofunction:: PyIRoGlass.Gauss
   :members:

.. autofunction:: PyIRoGlass.Linear
   :members:

.. autofunction:: PyIRoGlass.Carbonate
   :members:

.. autofunction:: PyIRoGlass.als_baseline
   :members:

.. autoclass:: PyIRoGlass.WhittakerSmoother
   :members:

.. autofunction:: PyIRoGlass.NearIR_Process
   :members:

.. autofunction:: PyIRoGlass.MidIR_Process
   :members:

.. autofunction:: PyIRoGlass.MCMC
   :members:

.. autofunction:: PyIRoGlass.Run_All_Spectra
   :members:


Functions for calculating concentrations
========================================


.. autofunction:: PyIRoGlass.Beer_Lambert
   :members:

.. autofunction:: PyIRoGlass.Beer_Lambert_Error
   :members:

.. autofunction:: PyIRoGlass.Concentration_Output
   :members:



Functions for calculating density, molar absorptivity
=====================================================


.. autofunction:: PyIRoGlass.Density_Calculation
   :members:

.. autofunction:: PyIRoGlass.Epsilon_Calculation
   :members:


Functions for plotting MCMC results
===================================


.. autofunction:: PyIRoGlass.modelfit
   :members:

.. autofunction:: PyIRoGlass.histogram
   :members:

.. autofunction:: PyIRoGlass.pairwise
   :members:

.. autofunction:: PyIRoGlass.trace
   :members:

.. autofunction:: PyIRoGlass.subplotter
   :members:


Functions for determining thickness from reflectance FTIR spectra
=================================================================


.. autofunction:: PyIRoGlass.PeakID
   :members:

.. autofunction:: PyIRoGlass.Thickness_Calc
   :members:

.. autofunction:: PyIRoGlass.Thickness_Processing
   :members:

.. autofunction:: PyIRoGlass.Reflectance_Index
   :members:


Functions for molar absorptivity inversions
===========================================


.. autofunction:: PyIRoGlass.Inversion
   :members:

.. autofunction:: PyIRoGlass.Least_Squares
   :members:

.. autofunction:: PyIRoGlass.Calculate_Calibration_Error
   :members:

.. autofunction:: PyIRoGlass.Calculate_Epsilon
   :members:

.. autofunction:: PyIRoGlass.Calculate_SEE
   :members:

.. autofunction:: PyIRoGlass.Calculate_R2
   :members:

.. autofunction:: PyIRoGlass.Calculate_RMSE
   :members:

.. autofunction:: PyIRoGlass.Inversion_Fit_Errors
   :members:
