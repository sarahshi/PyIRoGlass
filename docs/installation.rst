=======================
Installation & Updating
=======================

Installation
============

First, obtain Python3 (tested on versions≥3.7). If you haven't used python before, we recomend installing it through `anaconda3 <https://www.anaconda.com/products/individual>`_.

``PyIRoGlass`` can be installed using pip in one line. If you are using a terminal, enter:

.. code-block:: python

   pip install PyIRoGlass

If you are using Jupyter Notebooks (on Google Colab or Binder) or Jupyter Lab, you can also install it by entering the following code into a notebook cell (note the !):

.. code-block:: python

   !pip install PyIRoGlass

You then need to import ``PyIRoGlass`` into the script you are running code in. In all the examples, we import ``PyIRoGlass`` as pig:

.. code-block:: python

   import PyIRoGlass as pig

This means any time you want to call a function from ``PyIRoGlass``, you do pig.function_name.



Updating
========

To upgrade to the most recent version of ``PyIRoGlass``, type the following into terminal:

.. code-block:: python

   pip install PyIRoGlass --upgrade

Or in your Jupyter environment:

.. code-block:: python

   !pip install PyIRoGlass --upgrade


For maximum reproducability, you should state which version of ``PyIRoGlass`` you are using. If you have imported ``PyIRoGlass`` as pig, you can find this using:

.. code-block:: python

    pig.__version__