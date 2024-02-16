PyIRoGlass Documentation
========================


Data Imports 
============

.. autoclass:: PyIRoGlass.SampleDataLoader
   :no-index:

.. autoclass:: PyIRoGlass.VectorLoader
   :no-index:


Building-blocks functions for fitting baselines and peaks
=========================================================

.. autofunction:: PyIRoGlass.gauss

.. autofunction:: PyIRoGlass.linear

.. autofunction:: PyIRoGlass.carbonate

.. autofunction:: PyIRoGlass.als_baseline

.. autoclass:: PyIRoGlass.WhittakerSmoother
   :members:


.. autofunction:: PyIRoGlass.NIR_process


.. autofunction:: PyIRoGlass.MIR_process


.. autofunction:: PyIRoGlass.MCMC

.. autofunction:: PyIRoGlass.calculate_baselines


Functions for calculating concentrations
========================================

.. autofunction:: PyIRoGlass.beer_lambert

.. autofunction:: PyIRoGlass.beer_lambert_error

.. autofunction:: PyIRoGlass.calculate_concentrations


Functions for calculating density, molar absorptivity
=====================================================

.. autofunction:: PyIRoGlass.calculate_density

.. autofunction:: PyIRoGlass.calculate_epsilon


Functions for plotting MCMC results
===================================

.. autofunction:: PyIRoGlass.plot_H2Om_OH

.. autofunction:: PyIRoGlass.plot_H2Ot_3550

.. autofunction:: PyIRoGlass.derive_carbonate

.. autofunction:: PyIRoGlass.plot_carbonate

.. autofunction:: PyIRoGlass.plot_modelfit

.. autofunction:: PyIRoGlass.plot_trace


Functions for determining thickness from reflectance FTIR spectra
=================================================================

.. autofunction:: PyIRoGlass.datacheck_peakdetect

.. autofunction:: PyIRoGlass.peakdetect

.. autofunction:: PyIRoGlass.peakID

.. autofunction:: PyIRoGlass.calculate_thickness

.. autofunction:: PyIRoGlass.calculate_mean_thickness

.. autofunction:: PyIRoGlass.reflectance_index


Functions for molar absorptivity inversions
===========================================

.. autofunction:: PyIRoGlass.inversion

.. autofunction:: PyIRoGlass.least_squares

.. autofunction:: PyIRoGlass.calculate_calibration_error

.. autofunction:: PyIRoGlass.calculate_epsilon_inversion

.. autofunction:: PyIRoGlass.calculate_SEE

.. autofunction:: PyIRoGlass.calculate_R2

.. autofunction:: PyIRoGlass.calculate_RMSE

.. autofunction:: PyIRoGlass.inversion_fit_errors

