# %% -*- coding: utf-8 -*-
""" Created on August 29, 2022 // @author: Sarah Shi"""

# %% 

from filecmp import dircmp
from setuptools import setup, find_packages
import os, codecs
# from MC3_BASELINES import __version__

# %% 


dir = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(dir, 'README.md'), encoding='utf-8') as description:
    ext_description = description.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name='MC3_BASELINES',
    version=get_version('MC3_BASELINES/__init__.py'),
    author='Sarah Shi and Henry Towbin',
    author_email='sarah.c.shi@gmail.com',

    description='MC3_BASELINES',
    long_description = ext_description,
    long_description_content_type="text/markdown",
    url='https://github.com/sarahshi/BASELINES',

    packages=find_packages(where='src'),

    install_requires=[
            'numpy',
            'pandas',
            'scipy',
            'matplotlib',
            'scikit-learn',
            'pykrige',
            'mc3',
            'peakdetect',
            ],

    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    
    python_requires='>=3.6',
)
