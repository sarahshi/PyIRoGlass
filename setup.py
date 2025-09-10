
#!/usr/bin/env python
from os import path
from setuptools import setup, find_packages

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'src', 'PyIRoGlass', '_version.py'), encoding='utf-8') as f:
    exec(f.read())

with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="PyIRoGlass",
    version=__version__,
    author="Sarah Shi",
    author_email="sarah.c.shi@gmail.com",
    description="PyIRoGlass",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sarahshi/PyIRoGlass",
    package_dir={'': 'src'},  # Optional
    packages=find_packages(where='src'),  # Required

    package_data={
        # Include all pickle files
        "": ["*.pkl", "*.npz"],
    },
    install_requires=[
            'pandas',
            'numpy',
            'matplotlib',
            'scikit-learn',
            'scipy',
            'mc3', 
            'pykrige', 
            ],

    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
