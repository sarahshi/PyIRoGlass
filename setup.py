# %% -*- coding: utf-8 -*-
""" Created on August 29, 2022 // @author: Sarah Shi"""

# %% 

from filecmp import dircmp
from setuptools import setup, find_packages
from os import path

from MC3_BACKEND import __version__

# %% 


dir = path.abspath(path.dirname(__file__))

with open(path.join(dir, 'README.md'), encoding='utf-8') as description:
    ext_description = description.read()

setup(
    name='MC3_BASELINES',
    version=__version__,
    author='Sarah Shi and Henry Towbin',
    description='MC3_BASELINES',
    long_description = ext_description,

    url='https://github.com/sarahshi/BASELINES',
    author_email='sarah.c.shi@gmail.com',

    packages=find_packages(),

    install_requires=[
            'numpy',
            'pandas',
            'scipy',
            'matplotlib',
            'scikit-learn',
            'mc3',
            'pykrige',
            'peakdetect',
            ],

    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
)
