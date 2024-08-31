=========================
Introduction and Citation
=========================

Welcome to ``PyIRoGlass``: An Open-Source, Bayesian MCMC Algorithm for Fitting Baselines to FTIR Spectra of Basaltic-Andesitic Glasses. ``PyIRoGlass`` is an open-source Python3 tool for determining $\mathrm{H_2O}$ and $\mathrm{CO_2}$ species concentrations in the transmission FTIR spectra of basaltic to andesitic glasses. 

``PyIRoGlass`` has been reviewed and published at Volcanica. Please refer to the manuscript for a more detailed description of the development and validation of the method. If you use this package in your work, please cite: 

.. code-block:: console

   Shi, S., Towbin, W. H., Plank, T., Barth, A., Rasmussen, D., Moussallam, Y., Lee, H. J. and Menke, W. (2024) “PyIRoGlass: An open-source, Bayesian MCMC algorithm for fitting baselines to FTIR spectra of basaltic-andesitic glasses”, Volcanica, 7(2), pp. 471–501. doi: 10.30909/vol.07.02.471501.

.. code-block:: text

   @article{Shietal2024,
       doi       = {10.30909/vol.07.02.471501},
       url       = {https://doi.org/10.30909/vol.07.02.471501},
       year      = {2024},
       volume    = {7},
       number    = {2},
       pages     = {471-501},
       author    = {Shi, Sarah C. and Towbin, W. Henry and Plank, Terry and Barth, Anna and Rasmussen, Daniel and Moussallam, Yves and Lee, Hyun Joo and Menke, William},
       title     = {PyIRoGlass: An open-source, Bayesian MCMC algorithm for fitting baselines to FTIR spectra of basaltic-andesitic glasses},
       journal   = {Volcanica}
   }

The open-source nature of the tool allows for continuous development. We welcome the submission of devolatilized FTIR spectra that can continue to shape the form of the baseline, and molar absorptivities. You can email `sarahshi@berkeley.edu <mailto:sarahshi@berkeley.edu>`_ or post an enhancement request or report of a bug on the issue page of the `PyIRoGlass GitHub repository <https://github.com/SarahShi/PyIRoGlass>`_. 


=============
Collaborators
=============

These folks have been fundamental to the development of ``PyIRoGlass``: 

- `Sarah Shi <https://github.com/sarahshi>`_ (LDEO, UC Berkeley)
- `Henry Towbin <https://github.com/whtowbin>`_ (LDEO, GIA)
- `Terry Plank <https://github.com/terryplank>`_ (LDEO)
- `Anna Barth <https://github.com/barthac>`_ (LDEO, UC Berkeley)
- `Daniel Rasmussen <https://github.com/DJRgeoscience>`_ (LDEO)
- `Yves Moussallam <https://eesc.columbia.edu/content/yves-moussallam>`_ (LDEO)
- `Hyun Joo Lee <https://people.climate.columbia.edu/users/profile/hyun-joo-lee>`_ (LDEO)
- `William Menke <https://www.ldeo.columbia.edu/users/menke/>`_ (LDEO)


=====
Units
=====

``PyIRoGlass`` performs all calculations using melt compositions in oxide weight percentages, thicknesses in micrometers, analytical temperature in Celsius, and analytical pressure in bars. 

