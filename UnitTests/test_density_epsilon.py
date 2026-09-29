import unittest
import warnings
import numpy as np
import pandas as pd
import PyIRoGlass as pig


class test_density_epsilon_calculation(unittest.TestCase):
    def setUp(self):
        self.MI_Composition = pd.DataFrame(
            [
                {
                    "Sample": "AC4_OL53_101220_256s_30x30_a",
                    "SiO2": 47.95,
                    "TiO2": 1.00,
                    "Al2O3": 18.88,
                    "Fe2O3": 2.04,
                    "FeO": 7.45,
                    "MnO": 0.19,
                    "MgO": 4.34,
                    "CaO": 9.84,
                    "Na2O": 3.47,
                    "K2O": 0.67,
                    "P2O5": 0.11,
                    "H2O": 4.034754627,
                }
            ]
        )
        self.MI_Composition.set_index("Sample", inplace=True)
        self.MI_Composition_dry = pd.DataFrame(
            [
                {
                    "Sample": "AC4_OL53_101220_256s_30x30_a",
                    "SiO2": 47.95,
                    "TiO2": 1.00,
                    "Al2O3": 18.88,
                    "Fe2O3": 2.04,
                    "FeO": 7.45,
                    "MnO": 0.19,
                    "MgO": 4.34,
                    "CaO": 9.84,
                    "Na2O": 3.47,
                    "K2O": 0.67,
                    "P2O5": 0.11,
                    "H2O": 0,
                }
            ]
        )
        self.MI_Composition_dry.set_index("Sample", inplace=True)
        self.T_room = 25
        self.P_room = 1
        self.decimalPlace = 3

    def test_density_calculation(self):
        _, density_ls = pig.calculate_density(
            self.MI_Composition, self.T_room, self.P_room, model="LS"
        )
        result_ls = float(density_ls.values[0])
        expected_ls = 2702.815500
        self.assertAlmostEqual(
            result_ls,
            expected_ls,
            self.decimalPlace,
            msg="Density test and expected values from the "
            "calculate_density function with LS do not agree",
        )

        _, density_it = pig.calculate_density(
            self.MI_Composition, self.T_room, self.P_room, model="IT"
        )
        result_it = float(density_it.values[0])
        expected_it = 2751.083691
        self.assertAlmostEqual(
            result_it,
            expected_it,
            self.decimalPlace,
            msg="Density test and expected values from the "
            "calculate_density function with IT do not agree",
        )

    def test_epsilon_calculation(self):
        epsilon = pig.calculate_epsilon(
            self.MI_Composition_dry, self.T_room, self.P_room
        )
        tau = float(epsilon["Tau"].iloc[0])
        expected_tau = 0.682894853
        epsilon_h2ot = float(epsilon["epsilon_H2Ot_3550"].iloc[0])
        expected_epsilon_h2ot = 64.49365540510483
        sigma_epsilon_h2ot = float(epsilon["sigma_epsilon_H2Ot_3550"].iloc[0])
        expected_sigma_epsilon_h2ot = 7.380834041750025
        self.assertAlmostEqual(
            tau,
            expected_tau,
            self.decimalPlace,
            msg="Tau test and expected values from the "
            "calculate_epsilon function do not agree",
        )
        self.assertAlmostEqual(
            epsilon_h2ot,
            expected_epsilon_h2ot,
            self.decimalPlace,
            msg="epsilon_H2Ot test and expected values from the "
            "calculate_epsilon function do not agree",
        )
        self.assertAlmostEqual(
            sigma_epsilon_h2ot,
            expected_sigma_epsilon_h2ot,
            self.decimalPlace,
            msg="sigma_epsilon_H2Ot test and expected values from the "
            "calculate_epsilon function do not agree",
        )


class test_epsilon_carbonate_co2(unittest.TestCase):
    """epsilon_carbonate (Na/(Na+Ca) regression, for the 1515 and 1430 cm^-1
    carbonate peaks) and epsilon_CO2 (fixed, molecular CO2) are separate."""

    def setUp(self):
        self.oxides = ["SiO2", "TiO2", "Al2O3", "Fe2O3", "FeO", "MnO",
                       "MgO", "CaO", "Na2O", "K2O", "P2O5", "H2O"]
        # AC4_OL53 basaltic andesite, within all calibration ranges.
        self.in_range = pd.DataFrame(
            [[47.95, 1.00, 18.88, 2.04, 7.45, 0.19, 4.34, 9.84, 3.47, 0.67,
              0.11, 0]],
            columns=self.oxides, index=["AC4_OL53_101220_256s_30x30_a"],
        )
        # Na-rich, Ca-poor glass: Eta (Na/(Na+Ca)) above the carbonate
        # calibration range.
        self.high_eta = pd.DataFrame(
            [[60.0, 0.5, 18.0, 1.0, 3.0, 0.1, 1.0, 1.0, 8.0, 3.0, 0.1, 0]],
            columns=self.oxides, index=["high_eta"],
        )
        self.T_room = 25
        self.P_room = 1

    def test_columns(self):
        epsilon = pig.calculate_epsilon(self.in_range, self.T_room,
                                        self.P_room)
        for col in ["epsilon_carbonate", "sigma_epsilon_carbonate",
                    "epsilon_CO2", "sigma_epsilon_CO2", "Notes"]:
            self.assertIn(col, epsilon.columns,
                          msg=f"{col} missing from calculate_epsilon output")

    def test_epsilon_carbonate_regression(self):
        epsilon = pig.calculate_epsilon(self.in_range, self.T_room,
                                        self.P_room)
        eta = float(epsilon["Eta"].iloc[0])
        expected = 417.17390625 - 318.09377591 * eta
        self.assertAlmostEqual(
            float(epsilon["epsilon_carbonate"].iloc[0]), expected, 6,
            msg="epsilon_carbonate does not follow the Na/(Na+Ca) regression",
        )
        # Same values as the former epsilon_CO2 and sigma_epsilon_CO2 (v0.6.7).
        self.assertAlmostEqual(
            float(epsilon["epsilon_carbonate"].iloc[0]), 293.26130023322935, 6,
            msg="epsilon_carbonate test and expected values do not agree",
        )
        self.assertAlmostEqual(
            float(epsilon["sigma_epsilon_carbonate"].iloc[0]),
            16.28711970278645, 6,
            msg="sigma_epsilon_carbonate test and expected values do not agree",
        )

    def test_epsilon_co2_fixed(self):
        epsilon = pig.calculate_epsilon(
            pd.concat([self.in_range, self.high_eta]), self.T_room,
            self.P_room)
        self.assertTrue((epsilon["epsilon_CO2"].astype(float) == 830).all(),
                        msg="epsilon_CO2 should be fixed at 830")
        self.assertTrue(np.allclose(epsilon["sigma_epsilon_CO2"].astype(float), 41.5),
                        msg="sigma_epsilon_CO2 should be 5% of 830")
        self.assertFalse(
            np.isclose(float(epsilon["epsilon_carbonate"].iloc[0]), 830),
            msg="epsilon_carbonate should not equal the fixed epsilon_CO2",
        )

    def test_notes_in_range(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no calibration-range warnings
            epsilon = pig.calculate_epsilon(self.in_range, self.T_room,
                                            self.P_room)
        self.assertEqual(epsilon["Notes"].iloc[0], "",
                         msg="Notes should be empty within calibration ranges")

    def test_notes_out_of_range(self):
        with self.assertWarns(UserWarning):
            epsilon = pig.calculate_epsilon(self.high_eta, self.T_room,
                                            self.P_room)
        self.assertGreater(float(epsilon["Eta"].iloc[0]), 0.8406489374848108)
        self.assertIn("Eta outside range (epsilon_carbonate)",
                      epsilon["Notes"].iloc[0],
                      msg="Notes should record the out-of-range Eta")


if __name__ == "__main__":
    unittest.main()
