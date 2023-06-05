import unittest
from unittest.mock import patch, MagicMock
import os
import numpy as np
import pandas as pd
import PyIRoGlass as pig


class test_plotting(unittest.TestCase):

    @patch('numpy.lexsort')
    @patch('numpy.ediff1d')
    @patch('numpy.where')
    @patch('numpy.amax')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplot')
    @patch('pig.mu.default_parnames')  # adjust this according to where this function is defined

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


if __name__ == '__main__':
     unittest.main()
