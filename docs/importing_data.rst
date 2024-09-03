==============
Importing Data
==============

We walk through an implementation of ``PyIRoGlass`` here. We recommend following this tutorial as-is for those not familiar with navigating between directories in `Python`. Create this following file structure locally: 

::

    PyIRoGlass/
    ├── Inputs/
    │   ├── ChemThick.csv
    │   ├── ReflectanceSpectra/
    │   └── TransmissionSpectra/
    │       └── YourDirectoryName
    │           ├── a.CSV
    │           ├── b.CSV
    │           └── c.CSV
    │
    └── PyIRoGlass_RUN.py


Users can batch process their FTIR data by creating directories containing all spectra files, called `TransmissionSpectra/YourDirectoryName` here, in comma separated values (`.CSV`). Users should format their glass composition and thickness data as a spreadsheet of comma separated values (`.CSV`) file with each analysis having its own row and columns of sample name, oxide components in weight percentages, and thicknesses and uncertainties in thickness in micrometers. The spectrum file name **must** match the sample name input in the chemistry and thickness file. The order of columns does not matter, as the `Python` ``pandas`` package will identify the column heading regardless of its location. 

The following columns are required for this `ChemThick.CSV` file:

*  Sample
*  :math:`SiO_{2}`
*  :math:`TiO_{2}`
*  :math:`Al_{2}O_{3}`
*  :math:`Fe_{2}O_{3}`
*  :math:`FeO_{t}`
*  :math:`MnO`
*  :math:`MgO`
*  :math:`CaO`
*  :math:`Na_{2}O`
*  :math:`K_{2}O`
*  :math:`P_{2}O_{5}`
*  Thickness
*  Sigma_Thickness

For example, here a screenshot of a `.CSV` spreadsheet containing the glass composition and thickness data. You can use the `ChemThickTemplate.CSV` from the GitHub repository to create your own. You should fill every cell, else ``PyIRoGlass`` will assume that oxide was not analyzed or detected. For oxides that were not analyzed or not detected, enter 0 into the cell. 

.. image:: _static/chemthick.png


For the liquid composition, ``PyIRoGlass`` allows users to specify how they partition :math:`Fe` between ferrous and ferric iron, because glass density changes due to the proportion of :math:`Fe^{3+}`. To avoid ambiguity, the `ChemThick.CSV` handles this by providing two columns for :math:`FeO` and :math:`Fe_{2}O_{3}`. If the speciation is unknown, input all :math:`Fe` as :math:`FeO` and leave the :math:`Fe_{2}O_{3}` cells empty. This will not constitute the largest uncertainty, as the molar absorptivities and thicknesses impact concentrations more significantly. 

=================
Importing Package
=================

We import the package ``PyIRoGlass`` in `Python`. 

.. code-block:: python

   import PyIRoGlass as pig

========================================
PyIRoGlass for Transmission FTIR Spectra
========================================

We use the ``os`` package in `Python` to facilitate navigation to various directories and files. To load the transmission FTIR spectra, you must provide the path to the directory. Specify the wavenumbers of interest to fit all species peaks between 5500 and 1000 cm\ :sup:`-1`. 

.. code-block:: python

    path = os.getcwd() + '/Inputs/TransmissionSpectra/YourDirectoryName/'
    loader = pig.SampleDataLoader(spectrum_path=path)
    dfs_dict = loader.load_spectrum_directory()

:class:`pig.SampleDataLoader` and :meth:`load_spectrum_directory` returns ``dfs_dict``, a dictionary of the wavenumber and absorbance of each sample. 

To load the `.CSV` containing glass chemistry and thickness information, provide the path to the file. 

.. code-block:: python

    chemistry_thickness_path = os.getcwd() + '/Inputs/ChemThick.csv'
    loader = pig.SampleDataLoader(chemistry_thickness_path=chemistry_thickness_path)
    chemistry, thickness = loader.load_chemistry_thickness()

Inspect each returned data type to ensure that the data imports are successful. 


=========================================
Thicknesses from Reflectance FTIR Spectra 
=========================================

Loading reflectance FTIR spectra occurs through a near-identical process. Define your path to the file, but modify the wavenumbers of interest for either glass or olivine. 

.. code-block:: python

    ref_path = os.getcwd() + '/Inputs/ReflectanceSpectra/YourDirectoryName/'
    loader = pig.SampleDataLoader(spectrum_path=ref_path)
    ref_dfs_dict = loader.load_spectrum_directory(ref_path, wn_high=wn_high, wn_low=wn_low)

For olivine, specify the following wavenumber range based on :cite:t:`NicholsandWysoczanski2007` and calculate the relevant reflectance index :math:`n` for your given :math:`X_{Fo}` from :cite:t:`DHZ1992`. 

.. code-block:: python

    ref_dfs_dict_ol = loader.load_spectrum_directory(ref_path, wn_high=2700, wn_low=2100)
    n_ol = pig.reflectance_index(XFo) 

For glass, specify the following wavenumber range based on :cite:t:`NicholsandWysoczanski2007` and enter the relevant reflectance index :math:`n`. We use the reflectance index for basaltic glasses from :cite:t:`NicholsandWysoczanski2007` here. 

.. code-block:: python

    ref_dfs_dict_gl = loader.load_spectrum_directory(ref_path, wn_high=2850, wn_low=1700)
    n_gl = 1.546 


====================
Data Import Complete 
====================

That is all for loading files! You are ready to get rolling with ``PyIRoGlass``. See the example notebook `PyIRoGlass_RUN.ipynb`, under the big examples heading, to see how to run ``PyIRoGlass`` and export files. 
