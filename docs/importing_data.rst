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

*  :math:`\text{Sample}`
*  :math:`\text{SiO}_{2}`
*  :math:`\text{TiO}_{2}`
*  :math:`\text{Al}_{2}\text{O}_{3}`
*  :math:`\text{Fe}_{2}\text{O}_{3}`
*  :math:`\text{FeO}_{t}`
*  :math:`\text{MnO}`
*  :math:`\text{MgO}`
*  :math:`\text{CaO}`
*  :math:`\text{Na}_{2}\text{O}`
*  :math:`\text{K}_{2}\text{O}`
*  :math:`\text{P}_{2}\text{O}_{5}`
*  :math:`\text{Thickness}`
*  :math:`\text{Sigma_Thickness}`

For example, here is an example of a `.CSV` table containing the glass composition and thickness data. You can use the `ChemThickTemplate.CSV` from the GitHub repository to create your own. You should fill every cell, else ``PyIRoGlass`` will assume that oxide was not analyzed or detected. For oxides that were not analyzed or not detected, enter 0 into the cell. 

+----------------------------------+-------+-------+--------+--------+-------+-------+-------+-------+-------+------+-------+-----------+-----------------+
| Sample                           | SiO2  | TiO2  | Al2O3  | Fe2O3  | FeO   | MnO   | MgO   | CaO   | Na2O  | K2O  | P2O5  | Thickness | Sigma_Thickness |
+==================================+=======+=======+========+========+=======+=======+=======+=======+=======+======+=======+===========+=================+
| AC4_EUH33_030920_256s_20x20_a    | 53.02 | 0.96  | 18.28  | 2.16   | 7.89  | 0.23  | 3.48  | 7.18  | 4.48  | 1.15 | 0.28  | 49.60     | 3.00            |
+----------------------------------+-------+-------+--------+--------+-------+-------+-------+-------+-------+------+-------+-----------+-----------------+
| AC4_EUH102_030920_256s_15x20_a   | 50.26 | 0.92  | 17.60  | 2.17   | 7.92  | 0.21  | 4.31  | 9.30  | 3.57  | 0.69 | 0.16  | 45.63     | 3.00            |
+----------------------------------+-------+-------+--------+--------+-------+-------+-------+-------+-------+------+-------+-----------+-----------------+
| AC4_OL3_101220_256s_30x30_a      | 51.87 | 1.01  | 18.20  | 2.16   | 7.89  | 0.23  | 3.52  | 6.94  | 4.73  | 1.14 | 0.16  | 49.33     | 3.00            |
+----------------------------------+-------+-------+--------+--------+-------+-------+-------+-------+-------+------+-------+-----------+-----------------+
| AC4_OL21_012821_256s_20x20_a     | 50.36 | 1.25  | 17.51  | 2.08   | 7.59  | 0.18  | 3.78  | 8.23  | 4.01  | 0.85 | 0.20  | 31.67     | 3.00            |
+----------------------------------+-------+-------+--------+--------+-------+-------+-------+-------+-------+------+-------+-----------+-----------------+


For the liquid composition, ``PyIRoGlass`` allows users to specify how they partition :math:`\text{Fe}` between ferrous and ferric iron, because glass density changes due to the proportion of :math:`\text{Fe}^{3+}`. Both the :cite:t:`LesherandSpera2015` and :cite:t:`IacovinoandTill2019` density models are available for use. To avoid ambiguity, the `ChemThick.CSV` handles this by providing two columns for :math:`\text{FeO}` and :math:`\text{Fe}_2\text{O}_3`. If the speciation is unknown, input all :math:`\text{Fe}` as :math:`\text{FeO}` and leave the :math:`\text{Fe}_2\text{O}_3` cells empty. This will not constitute the largest uncertainty, as the molar absorptivities and thicknesses impact concentrations more significantly. 

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
