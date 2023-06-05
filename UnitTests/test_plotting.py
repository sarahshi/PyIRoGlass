import unittest
from unittest.mock import patch, MagicMock
import os
import mc3.utils as mu
import mc3.stats as ms
import numpy as np
import pandas as pd
import PyIRoGlass as pig


class test_plotting_trace(unittest.TestCase):

    @patch('numpy.lexsort')
    @patch('numpy.ediff1d')
    @patch('numpy.where')
    @patch('numpy.amax')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplot')
    @patch('mc3.utils.default_parnames')

    def test_trace(self, mock_default_parnames, mock_subplot, mock_figure, mock_amax, mock_where, mock_ediff1d, mock_lexsort):

        mock_default_parnames.return_value = ['a', 'b', 'c']
        mock_amax.return_value = 1
        mock_where.return_value = [np.array([1])]
        mock_ediff1d.return_value = np.array([1])
        mock_lexsort.return_value = np.array([1])
        mock_figure.return_value = MagicMock()
        mock_subplot.return_value = MagicMock()

        posterior = np.array([[1,2,3],[4,5,6]])
        zchain = np.array([0,1])
        pnames = None
        thinning = 50
        burnin = 0
        fignum = 1000
        savefile = None
        fmt = "."
        ms = 2.5
        fs = 12

        result = pig.trace(posterior, 'test', zchain, pnames, thinning, burnin, fignum, savefile, fmt, ms, fs)
        self.assertIsNotNone(result)  # basic check, adjust this according to what you want to assert

class test_plotting_modelfit(unittest.TestCase):
    
    @patch('numpy.size')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.axes')
    @patch('mc3.stats.bin_array')  # adjust this according to where this function is defined

    def test_modelfit(self, mock_bin_array, mock_axes, mock_figure, mock_size):

        mock_bin_array.return_value = np.array([1])
        mock_axes.return_value = MagicMock()
        mock_figure.return_value = MagicMock()
        mock_size.return_value = 1

        data = np.array([1,2,3])
        uncert = np.array([1,2,3])
        indparams = np.array([1,2,3])
        model = np.array([1,2,3])
        nbins = 75
        fignum = 1400
        savefile = None
        fmt = "."

        result = pig.modelfit(data, uncert, indparams, model, 'test', nbins, fignum, savefile, fmt)
        self.assertIsNotNone(result)  # basic check, adjust this according to what you want to assert


if __name__ == '__main__':
     unittest.main()
