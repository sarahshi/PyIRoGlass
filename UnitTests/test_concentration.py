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
        self.epsilon = 64.4628687805379
        self.sigma_epsilon = 7.401239521
        self.N = 500000
        self.MI_Composition = {'SiO2': 47.95, 'TiO2': 1.00, 'Al2O3': 18.88, 'Fe2O3': 2.04, 'FeO': 7.45, 'MnO': 0.19,
                               'MgO': 4.34, 'CaO': 9.84, 'Na2O': 3.47, 'K2O': 0.67, 'P2O5': 0.11}
        self.T_room = 25 
        self.P_room = 1 
        self.decimalPlace = 5

    def test_beer_lambert(self):

        result = pig.Beer_Lambert(self.molar_mass, self.absorbance, self.sigma_absorbance, self.density, self.sigma_density, self.thickness, self.sigma_thickness, self.epsilon, self.sigma_epsilon)
        expected = 4.03892743514451
        self.assertAlmostEqual(result, expected, self.decimalPlace, msg="H2Ot test and expected values from the Beer_Lambert function do not agree")

    def test_beer_lambert_error(self):

        result = pig.Beer_Lambert_Error(self.N, self.molar_mass, self.absorbance, self.sigma_absorbance, self.density, self.sigma_density, self.thickness, self.sigma_thickness, self.epsilon, self.sigma_epsilon)
        expected = 0.432638234770662
        self.assertAlmostEqual(result, expected, self.decimalPlace-2, msg="H2Ot test and expected errors from the Beer_Lambert_Error function do not agree")


class test_conc_outputs_co2(unittest.TestCase):

    def setUp(self): 

        self.molar_mass = 44.01
        self.absorbance = 0.053059942737517
        self.sigma_absorbance = 0.00376445440661992
        self.density = 2702.703546
        self.sigma_density = self.density * 0.025
        self.thickness = 39
        self.sigma_thickness = 3
        self.epsilon = 302.327095954195
        self.sigma_epsilon = 18.0682300900254
        self.N = 500000
        self.MI_Composition = {'SiO2': 47.95, 'TiO2': 1.00, 'Al2O3': 18.88, 'Fe2O3': 2.04, 'FeO': 7.45, 'MnO': 0.19,
                               'MgO': 4.34, 'CaO': 9.84, 'Na2O': 3.47, 'K2O': 0.67, 'P2O5': 0.11}
        self.T_room = 25 
        self.P_room = 1 
        self.decimalPlace = 5

    def test_beer_lambert(self):

        result = pig.Beer_Lambert(self.molar_mass, self.absorbance, self.sigma_absorbance, self.density, self.sigma_density, self.thickness, self.sigma_thickness, self.epsilon, self.sigma_epsilon) * 10000
        expected = 732.787503648457
        self.assertAlmostEqual(result, expected, self.decimalPlace, msg="CO2_1515 test and expected values from the Beer_Lambert equation do not agree")

    def test_beer_lambert_error(self):
        
        result = pig.Beer_Lambert_Error(self.N, self.molar_mass, self.absorbance, self.sigma_absorbance, self.density, self.sigma_density, self.thickness, self.sigma_thickness, self.epsilon, self.sigma_epsilon) * 10000
        expected = 84.3764722769866
        self.assertAlmostEqual(result, expected, self.decimalPlace-4, msg="CO2_1515 test and expected errors from the Beer_Lambert_Error equation do not agree")


class test_conc_outputs(unittest.TestCase):

    def setUp(self):

        self.MI_Composition = pd.DataFrame({'SiO2': [47.95], 'TiO2': [1.00], 'Al2O3': [18.88], 'Fe2O3': [2.04], 'FeO': [7.45],
                              'MnO': [0.19], 'MgO': [4.34], 'CaO': [9.84], 'Na2O': [3.47], 'K2O': [0.67], 'P2O5': [0.11]},
                              index=['AC4_OL53_101220_256s_30x30_a'])
        self.PH = pd.DataFrame({'PH_3550_M': [1.52334293070956], 'PH_3550_STD': [0.00330886816920362], 'H2OT_3550_SAT?': ['-'], 'PH_1635_BP': [0.299675069462131],
                   'PH_1635_STD': [0.00312186211814352], 'PH_1515_BP': [0.053059942737517], 'PH_1515_STD': [0.00376445440661992],
                   'PH_1430_BP': [0.050243218748645], 'PH_1430_STD': [0.00435858737555569], 'PH_5200_M': [0.00895907201720724], 'PH_5200_STD': [0.000445131906892326], 
                   'PH_4500_M': [0.0125175731361643], 'PH_4500_STD': [0.000320314738906861], 'S2N_P5200': [6.90415549754328], 'S2N_P4500': [5.52765303774205],
                   'ERR_5200': ['-'], 'ERR_4500': ['-']}, index=['AC4_OL53_101220_256s_30x30_a'])
        self.thickness = pd.DataFrame({'Thickness': [39], 'Sigma_Thickness': [3]}, index=['AC4_OL53_101220_256s_30x30_a'])
        self.N = 500000
        self.T_room = 25 
        self.P_room = 1 
        self.decimalPlace = 5

    def test_concentration(self):

        density_epsilon, mega_spreadsheet = pig.Concentration_Output(self.PH, self.N, self.thickness, self.MI_Composition, self.T_room, self.P_room)
        expected_H2O = 4.03892743514451
        expected_CO2 = 713.3372363302
        self.assertAlmostEqual(float(mega_spreadsheet['H2OT_MEAN']), expected_H2O, self.decimalPlace, msg="H2Ot test values from the Concentration_Output equation do not agree")
        self.assertAlmostEqual(float(mega_spreadsheet['CO2_MEAN']), expected_CO2, self.decimalPlace, msg="CO2m test values from the Concentration_Output equation do not agree")


if __name__ == '__main__':
     unittest.main()