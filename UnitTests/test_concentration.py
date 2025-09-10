import unittest
import pandas as pd
import PyIRoGlass as pig


class test_conc_outputs_h2ot(unittest.TestCase):
    def setUp(self):
        self.molar_mass = 18.01528
        self.absorbance = 1.523342931
        self.sigma_absorbance = 0.003308868
        self.density = 2702.81550010862
        self.sigma_density = self.density * 0.025
        self.thickness = 39
        self.sigma_thickness = 3
        self.epsilon = 64.5268644303552
        self.sigma_epsilon = 7.37609147230662
        self.N = 500000
        self.MI_Composition = {
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
        }
        self.T_room = 25
        self.P_room = 1
        self.decimalPlace = 5

    def test_beer_lambert(self):
        result = pig.beer_lambert(
            self.molar_mass, self.absorbance, self.density, self.thickness, self.epsilon
        )
        expected = 4.034754627293028
        self.assertAlmostEqual(
            result,
            expected,
            self.decimalPlace,
            msg="H2Ot test and expected values from the "
            "Beer_Lambert function do not agree",
        )

    def test_beer_lambert_error(self):
        result = pig.beer_lambert_error(
            self.N,
            self.molar_mass,
            self.absorbance,
            self.sigma_absorbance,
            self.density,
            self.sigma_density,
            self.thickness,
            self.sigma_thickness,
            self.epsilon,
            self.sigma_epsilon,
        )
        expected = 0.431636377873926
        self.assertAlmostEqual(
            result,
            expected,
            delta=0.01,
            msg="H2Ot test and expected errors from the "
            "Beer_Lambert_Error function do not agree",
        )


class test_conc_outputs_co2(unittest.TestCase):
    def setUp(self):
        self.molar_mass = 44.01
        self.absorbance = 0.053059942737517
        self.sigma_absorbance = 0.00376445440661992
        self.density = 2702.81550010862
        self.sigma_density = self.density * 0.025
        self.thickness = 39
        self.sigma_thickness = 3
        self.epsilon = 293.381038986771
        self.sigma_epsilon = 16.307943404879
        self.N = 500000
        self.MI_Composition = {
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
        }
        self.T_room = 25
        self.P_room = 1
        self.decimalPlace = 5

    def test_beer_lambert(self):  # OL53, 1515 peak
        result = (
            pig.beer_lambert(
                self.molar_mass,
                self.absorbance,
                self.density,
                self.thickness,
                self.epsilon,
            )
            * 10000
        )
        expected = 755.101090599377
        self.assertAlmostEqual(
            result,
            expected,
            self.decimalPlace,
            msg="CO2_1515 test and expected values from the "
            "Beer_Lambert function do not agree",
        )

    def test_beer_lambert_error(self):
        result = (
            pig.beer_lambert_error(
                self.N,
                self.molar_mass,
                self.absorbance,
                self.sigma_absorbance,
                self.density,
                self.sigma_density,
                self.thickness,
                self.sigma_thickness,
                self.epsilon,
                self.sigma_epsilon,
            )
            * 10000
        )
        expected = 86.3485059939307
        self.assertAlmostEqual(
            result,
            expected,
            delta=1,
            msg="CO2_1515 test and expected errors from the "
            "Beer_Lambert_Error function do not agree",
        )


class test_conc_outputs(unittest.TestCase):
    def setUp(self):
        self.MI_Composition = pd.DataFrame(
            {
                "SiO2": [47.95],
                "TiO2": [1.00],
                "Al2O3": [18.88],
                "Fe2O3": [2.04],
                "FeO": [7.45],
                "MnO": [0.19],
                "MgO": [4.34],
                "CaO": [9.84],
                "Na2O": [3.47],
                "K2O": [0.67],
                "P2O5": [0.11],
            },
            index=["AC4_OL53_101220_256s_30x30_a"],
        )
        self.PH = pd.DataFrame(
            {
                "PH_3550_M": [1.52334293070956],
                "PH_3550_STD": [0.00330886816920362],
                "H2Ot_3550_SAT": ["-"],
                "PH_1635_BP": [0.299675069462131],
                "PH_1635_STD": [0.00312186211814352],
                "PH_1515_BP": [0.053059942737517],
                "PH_1515_STD": [0.00376445440661992],
                "PH_1430_BP": [0.050243218748645],
                "PH_1430_STD": [0.00435858737555569],
                "PH_5200_M": [0.00895907201720724],
                "PH_5200_STD": [0.000445131906892326],
                "PH_4500_M": [0.0125175731361643],
                "PH_4500_STD": [0.000320314738906861],
                "STN_P5200": [6.90415549754328],
                "STN_P4500": [5.52765303774205],
                "ERR_5200": ["-"],
                "ERR_4500": ["-"],
            },
            index=["AC4_OL53_101220_256s_30x30_a"],
        )
        self.thickness = pd.DataFrame(
            {"Thickness": [39], "Sigma_Thickness": [3]},
            index=["AC4_OL53_101220_256s_30x30_a"],
        )
        self.N = 500000
        self.T_room = 25
        self.P_room = 1
        self.decimalPlace = 5

    def test_concentration(self):  # OL53
        concentrations = pig.calculate_concentrations(
            self.PH, self.MI_Composition, self.thickness, None
        )
        expected_H2O = 4.036918931617838
        expected_CO2 = 735.3744776620634
        self.assertAlmostEqual(
            float(concentrations["H2Ot_MEAN"].iloc[0]),
            expected_H2O,
            self.decimalPlace,
            msg="H2Ot test values from the calculate_concentrations "
            "function do not agree",
        )
        self.assertAlmostEqual(
            float(concentrations["CO2_MEAN"].iloc[0]),
            expected_CO2,
            self.decimalPlace,
            msg="CO2m test values from the calculate_concentrations "
            "function do not agree",
        )


class test_conc_outputs_saturated(unittest.TestCase):  # OL49
    def setUp(self):
        self.MI_Composition = pd.DataFrame(
            {
                "SiO2": [52.34],
                "TiO2": [1.04],
                "Al2O3": [17.92],
                "Fe2O3": [1.93],
                "FeO": [7.03],
                "MnO": [0.20],
                "MgO": [3.63],
                "CaO": [7.72],
                "Na2O": [4.25],
                "K2O": [0.78],
                "P2O5": [0.14],
            },
            index=["AC4_OL49_021920_30x30_H2O_a"],
        )
        self.PH = pd.DataFrame(
            {
                "PH_3550_M": [2.17224950314409],
                "PH_3550_STD": [0.00220916618090868],
                "H2Ot_3550_SAT": ["*"],
                "PH_1635_BP": [0.658349188362703],
                "PH_1635_STD": [0.00307592496318145],
                "PH_1515_BP": [0.106864326434928],
                "PH_1515_STD": [0.0035555642315165],
                "PH_1430_BP": [0.109429135799036],
                "PH_1430_STD": [0.00405404583060287],
                "PH_5200_M": [0.0249095178877388],
                "PH_5200_STD": [0.000801149484503151],
                "PH_4500_M": [0.014569658580663],
                "PH_4500_STD": [0.0000638336854435091],
                "STN_P5200": [12.1959313061391],
                "STN_P4500": [8.15732983956986],
                "ERR_5200": ["-"],
                "ERR_4500": ["-"],
            },
            index=["AC4_OL49_021920_30x30_H2O_a"],
        )
        self.thickness = pd.DataFrame(
            {"Thickness": [91.25], "Sigma_Thickness": [3]},
            index=["AC4_OL49_021920_30x30_H2O_a"],
        )
        self.N = 500000
        self.T_room = 25
        self.P_room = 1
        self.decimalPlace = 5

    def test_concentration(self):
        concentrations = pig.calculate_concentrations(
            self.PH, self.MI_Composition, self.thickness, None
        )
        expected_H2O = 2.54709950708554
        expected_CO2 = 752.5429016613664
        self.assertAlmostEqual(
            float(concentrations["H2Ot_MEAN"].iloc[0]),
            expected_H2O,
            self.decimalPlace,
            msg="H2Ot test values from the saturated calculate_concentrations "
            "function do not agree",
        )
        self.assertAlmostEqual(
            float(concentrations["CO2_MEAN"].iloc[0]),
            expected_CO2,
            self.decimalPlace,
            msg="CO2m test values from the saturated calculate_concentrations "
            "function do not agree",
        )


if __name__ == "__main__":
    unittest.main()
