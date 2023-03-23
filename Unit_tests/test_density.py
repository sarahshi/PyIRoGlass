import unittest
import numpy as np
import pandas as pd
import PyIRoGlass as pig

class test_density_calculation(unittest.TestCase):
    def setUp(self): 
        self.MI_Composition = pd.DataFrame([{'Sample': 'AC4_OL53_101220_256s_30x30_a', 'SiO2': 47.95, 'TiO2': 1.00, 'Al2O3': 18.88, 'Fe2O3': 2.04, 'FeO': 7.45, 'MnO': 0.19, 'MgO': 4.34, 'CaO': 9.84, 'Na2O': 3.47, 'K2O': 0.67, 'P2O5': 0.11}])
        self.MI_Composition.set_index('Sample', inplace = True)
        self.T_room = 25 
        self.P_room = 1 
        self.decimalPlace = 3

    def test_beer_lambert(self):
        mol, density = pig.Density_Calculation(self.MI_Composition, self.T_room, self.P_room)
        expected = 2702.703546
        self.assertAlmostEqual(density, expected, self.decimalPlace, msg="Density test and expected values from the Density_Calculation function do not agree")


if __name__ == '__main__':
     unittest.main()
