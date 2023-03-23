import unittest
import numpy as np
import pandas as pd
import PyIRoGlass as pig
import os
import glob

class test_thickness(unittest.TestCase):
    def setUp(self): 
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
        self.ref_path = os.path.join(self.dir_path, '/Inputs/ReflectanceSpectra/FuegoOl/')
        self.path = sorted(glob.glob(self.ref_path + "*"))
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
        df_files, df_dicts = pig.Load_SampleCSV(self.path, self.wn_high, self.wn_low)
        print(df_files.index)
        thickness_results = pig.Thickness_Processing(df_dicts, result, self.wn_high, self.wn_low, remove_baseline=False, plotting=False, phaseol=True)
        result = float(thickness_results.loc['AC4_OL27_REF_a']['Thickness_M'])
        expected = 79.81
        self.assertAlmostEqual(result, expected, self.decimalPlace-2, msg="Thickness test and expected values from the Thickness_Processing function do not agree")

if __name__ == '__main__':
     unittest.main()