import unittest
import numpy as np
import pandas as pd
import PyIRoGlass as pig
import sys

class test_thickness(unittest.TestCase):
    def setUp(self): 
        sys.path.append('../Inputs/ReflectanceSpectra/FuegoOl')
        self.csv = 'AC4_OL21_REF_a'

        self.xfo = 0.72
        self.decimalPlace = 4
        self.wn_high = 2700
        self.wn_low = 2100

    def test_reflectance_index(self):
        result = pig.Reflectance_Index(self.xfo)
        expected = 1.7097733333333334
        self.assertAlmostEqual(result, expected, self.decimalPlace, msg="Reflectance index test and expected values from the Reflectance_Index function do not agree")

    def test_process_thickness(self): 

        result = pig.Reflectance_Index(self.xfo)
        df_files, df_dicts = pig.Load_SampleCSV(self.csv, wn_high = 2700, wn_low = 2100)
        thickness_results = pig.Thickness_Processing(df_dicts, result, wn_high, wn_low, remove_baseline=False, plotting=False, phaseol=True)
        result = thickness_results['Thickness_M']
        expected = 79.81
        self.assertAlmostEqual(result, expected, self.decimalPlace-3, msg="Thickness test and expected values from the Thickness_Processing function do not agree")

if __name__ == '__main__':
     unittest.main()
