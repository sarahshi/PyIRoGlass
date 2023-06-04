import unittest
import numpy as np
import pandas as pd
import PyIRoGlass as pig


class test_conc_outputs_h2ot(unittest.TestCase):

    def setUp(self): 

        self.molar_mass = 18.01528
        self.absorbance = 1.523342931
        self.sigma_absorbance = 0.003308868
        self.density = 2702.703546
        self.sigma_density = self.density * 0.025
        self.thickness = 39
        self.sigma_thickness = 3
        self.epsilon = 64.46286878
        self.sigma_epsilon = 7.401239521
        self.N = 500000
        self.MI_Composition = {'SiO2': 47.95, 'TiO2': 1.00, 'Al2O3': 18.88, 'Fe2O3': 2.04, 'FeO': 7.45, 'MnO': 0.19,
                               'MgO': 4.34, 'CaO': 9.84, 'Na2O': 3.47, 'K2O': 0.67, 'P2O5': 0.11}
        self.T_room = 25 
        self.P_room = 1 
        self.decimalPlace = 4

    def test_beer_lambert(self):

        result = pig.Beer_Lambert(self.molar_mass, self.absorbance, self.sigma_absorbance, self.density, self.sigma_density, self.thickness, self.sigma_thickness, self.epsilon, self.sigma_epsilon)
        expected = 4.03892743514451
        self.assertAlmostEqual(result, expected, self.decimalPlace, msg="H2Ot test and expected values from the Beer_Lambert function do not agree")

    def test_beer_lambert_error(self):

        result = pig.Beer_Lambert_Error(self.N, self.molar_mass, self.absorbance, self.sigma_absorbance, self.density, self.sigma_density, self.thickness, self.sigma_thickness, self.epsilon, self.sigma_epsilon)
        expected = 0.433803676139589        
        self.assertAlmostEqual(result, expected, self.decimalPlace-3, msg="H2Ot test and expected errors from the Beer_Lambert_Error function do not agree")


class test_conc_outputs_co2(unittest.TestCase):

    def setUp(self): 

        self.molar_mass = 44.01
        self.absorbance = 0.052887397
        self.sigma_absorbance = 0.005128284
        self.density = 2702.703546
        self.sigma_density = self.density * 0.025
        self.thickness = 39
        self.sigma_thickness = 3
        self.epsilon = 302.327096
        self.sigma_epsilon = 18.06823009
        self.N = 500000
        self.MI_Composition = {'SiO2': 47.95, 'TiO2': 1.00, 'Al2O3': 18.88, 'Fe2O3': 2.04, 'FeO': 7.45, 'MnO': 0.19,
                               'MgO': 4.34, 'CaO': 9.84, 'Na2O': 3.47, 'K2O': 0.67, 'P2O5': 0.11}
        self.T_room = 25 
        self.P_room = 1 
        self.decimalPlace = 2

    def test_beer_lambert(self):

        result = pig.Beer_Lambert(self.molar_mass, self.absorbance, self.sigma_absorbance, self.density, self.sigma_density, self.thickness, self.sigma_thickness, self.epsilon, self.sigma_epsilon) * 10000
        expected = 730.4045443
        self.assertAlmostEqual(result, expected, self.decimalPlace, msg="CO2_1515 test and expected values from the Beer_Lambert equation do not agree")

    def test_beer_lambert_error(self):
        
        result = pig.Beer_Lambert_Error(self.N, self.molar_mass, self.absorbance, self.sigma_absorbance, self.density, self.sigma_density, self.thickness, self.sigma_thickness, self.epsilon, self.sigma_epsilon) * 10000
        expected = 97.35842352      
        self.assertAlmostEqual(result, expected, self.decimalPlace-2, msg="CO2_1515 test and expected errors from the Beer_Lambert_Error equation do not agree")


if __name__ == '__main__':
     unittest.main()
