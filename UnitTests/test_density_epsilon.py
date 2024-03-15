import unittest
import pandas as pd
import PyIRoGlass as pig


class test_density_epsilon_calculation(unittest.TestCase):

    def setUp(self):

        self.MI_Composition = pd.DataFrame(
            [{
                'Sample': 'AC4_OL53_101220_256s_30x30_a',
                'SiO2': 47.95,
                'TiO2': 1.00,
                'Al2O3': 18.88,
                'Fe2O3': 2.04,
                'FeO': 7.45,
                'MnO': 0.19,
                'MgO': 4.34,
                'CaO': 9.84,
                'Na2O': 3.47,
                'K2O': 0.67,
                'P2O5': 0.11,
                'H2O': 4.034754627
                }]
                )
        self.MI_Composition.set_index('Sample', inplace=True)
        self.MI_Composition_dry = pd.DataFrame(
            [{
                'Sample': 'AC4_OL53_101220_256s_30x30_a',
                'SiO2': 47.95,
                'TiO2': 1.00,
                'Al2O3': 18.88,
                'Fe2O3': 2.04,
                'FeO': 7.45,
                'MnO': 0.19,
                'MgO': 4.34,
                'CaO': 9.84,
                'Na2O': 3.47,
                'K2O': 0.67,
                'P2O5': 0.11,
                'H2O': 0
                }]
                )
        self.MI_Composition_dry.set_index('Sample', inplace=True)
        self.T_room = 25
        self.P_room = 1
        self.decimalPlace = 3

    def test_density_calculation(self):

        _, density_ls = pig.calculate_density(self.MI_Composition, self.T_room, self.P_room, model='LS')
        result_ls = float(density_ls.values[0])
        expected_ls = 2702.815500
        self.assertAlmostEqual(
            result_ls,
            expected_ls,
            self.decimalPlace,
            msg="Density test and expected values from the "
            "calculate_density function with LS do not agree")

        _, density_it = pig.calculate_density(self.MI_Composition, self.T_room, self.P_room, model='IT')
        result_it = float(density_it.values[0])
        expected_it = 2751.083691
        self.assertAlmostEqual(
            result_it,
            expected_it,
            self.decimalPlace,
            msg="Density test and expected values from the "
            "calculate_density function with IT do not agree")

    def test_epsilon_calculation(self):

        epsilon = pig.calculate_epsilon(self.MI_Composition_dry, self.T_room, self.P_room)
        tau = float(epsilon['Tau'].iloc[0])
        expected_tau = 0.682894853
        epsilon_h2ot = float(epsilon['epsilon_H2Ot_3550'].iloc[0])
        expected_epsilon_h2ot = 64.5268644303552
        sigma_epsilon_h2ot = float(epsilon['sigma_epsilon_H2Ot_3550'].iloc[0])
        expected_sigma_epsilon_h2ot = 7.37609147230662
        self.assertAlmostEqual(
            tau,
            expected_tau,
            self.decimalPlace,
            msg="Tau test and expected values from the "
            "calculate_epsilon function do not agree")
        self.assertAlmostEqual(
            epsilon_h2ot,
            expected_epsilon_h2ot,
            self.decimalPlace,
            msg="epsilon_H2Ot test and expected values from the "
            "calculate_epsilon function do not agree")
        self.assertAlmostEqual(
            sigma_epsilon_h2ot,
            expected_sigma_epsilon_h2ot,
            self.decimalPlace,
            msg="sigma_epsilon_H2Ot test and expected values from the "
            "calculate_epsilon function do not agree")


if __name__ == '__main__':
    unittest.main()
