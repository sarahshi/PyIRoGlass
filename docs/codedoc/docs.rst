PyIRoGlass Documentation
========================


Data Imports 
============

.. autofunction:: Load_SampleCSV

.. autofunction:: Load_PC

.. autofunction:: Load_Wavenumber

.. autofunction:: Load_ChemistryThickness


Building-blocks functions for fitting baselines and peaks
=========================================================

.. autofunction:: Gauss

.. autofunction:: Linear

.. autofunction:: Carbonate

.. autofunction:: als_baseline

.. autoclass:: WhittakerSmoother
   :members:

.. autofunction:: NearIR_Process

.. autofunction:: MidIR_Process

.. autofunction:: MCMC

.. autofunction:: Run_All_Spectra


Functions for calculating concentrations
========================================


.. autofunction:: Beer_Lambert

.. autofunction:: Beer_Lambert_Error

.. autofunction:: Concentration_Output



Functions for calculating density, molar absorptivity
=====================================================


.. autofunction:: Density_Calculation

.. autofunction:: Epsilon_Calculation


Functions for plotting MCMC results
===================================


.. autofunction:: modelfit

.. autofunction:: trace


Functions for determining thickness from reflectance FTIR spectra
=================================================================


.. autofunction:: PeakID

.. autofunction:: Thickness_Calc

.. autofunction:: Thickness_Process

.. autofunction:: Reflectance_Index


Functions for molar absorptivity inversions
===========================================


.. autofunction:: Inversion

.. autofunction:: Least_Squares

.. autofunction:: Calculate_Calibration_Error

.. autofunction:: Calculate_Epsilon

.. autofunction:: Calculate_SEE

.. autofunction:: Calculate_R2

.. autofunction:: Calculate_RMSE

.. autofunction:: Inversion_Fit_Errors
