import unittest
import numpy as np
import PyIRoGlass as pig


class test_inversion(unittest.TestCase):
    def setUp(self):
        self.tau = np.array(
            [
                0.62733751,
                0.859,
                0.86,
                0.708,
                0.74332446,
                0.746,
                0.795,
                0.859,
                0.79949467,
            ]
        )
        self.sigma_tau = self.tau * 0.025
        self.epsilon_1635 = np.array(
            [25.0, 56.0, 55.0, 42.0, 40.8, 42.34, 52.05, 55.0, 49.0]
        )
        self.sigma_epsilon_1635 = self.epsilon_1635 * 0.05
        self.decimalPlace = 5
        self.cov = np.identity(2)

    def test_epsilon_invert(self):
        mest_f, _, _ = pig.inversion(
            self.tau, self.epsilon_1635, self.sigma_tau, self.sigma_epsilon_1635
        )
        mls, _ = pig.least_squares(self.tau, self.epsilon_1635, self.sigma_epsilon_1635)
        m0 = float(mest_f[0])
        m1 = float(mest_f[1])
        expected_m0 = -50.3975642
        expected_m1 = 124.2505339
        m0_ls = float(mls[0])
        m1_ls = float(mls[1])
        expected_m0_ls = -49.05342621
        expected_m1_ls = 122.71710923
        self.assertAlmostEqual(
            m0,
            expected_m0,
            self.decimalPlace,
            msg="m0 test and expected values from the "
            "inversion function do not agree",
        )
        self.assertAlmostEqual(
            m1,
            expected_m1,
            self.decimalPlace,
            msg="m1 test and expected values from the "
            "inversion function do not agree",
        )
        self.assertAlmostEqual(
            m0_ls,
            expected_m0_ls,
            self.decimalPlace,
            msg="m0 test and expected values from the "
            "least_squares function do not agree",
        )
        self.assertAlmostEqual(
            m1_ls,
            expected_m1_ls,
            self.decimalPlace,
            msg="m1 test and expected values from the "
            "least_squares function do not agree",
        )

    def test_calibration_error(self):
        e_calib = pig.calculate_calibration_error(self.cov)
        expected_e_calib = 2
        self.assertAlmostEqual(
            e_calib,
            expected_e_calib,
            self.decimalPlace,
            msg="e_calib test and expected values from the "
            "calculate_calibration_error function do not agree",
        )

    def test_epsilon(self):
        m = np.array([0, 1])
        comp = 1
        eps = pig.calculate_y_inversion(m, comp)
        expected_eps = 1
        self.assertAlmostEqual(
            eps,
            expected_eps,
            self.decimalPlace,
            msg="epsilon test and expected values from the "
            "calculate_y_inversion function do not agree",
        )

    def test_residuals(self):
        residuals = np.array([0, 1, 2])
        see = pig.calculate_SEE(residuals)
        expected_see = 2.23606797749979

        rmse = pig.calculate_RMSE(residuals)
        expected_rmse = 1.2909944487358056

        self.assertAlmostEqual(
            see,
            expected_see,
            self.decimalPlace,
            msg="SEE test and expected values from the "
            "calculate_SEE function do not agree",
        )
        self.assertAlmostEqual(
            rmse,
            expected_rmse,
            self.decimalPlace,
            msg="RMSE test and expected values from the "
            "calculate_RMSE function do not agree",
        )

    def test_r2(self):
        values = np.array([0, 1, 2])
        r2 = pig.calculate_R2(values, values)
        expected_r2 = 1.0
        self.assertAlmostEqual(
            r2,
            expected_r2,
            self.decimalPlace,
            msg="R2 test and expected values from the "
            "calculate_R2 function do not agree",
        )

    def test_epsilon_invert_errors(self):
        mest_f, covm_est_f, covepsilon_est_f = pig.inversion(
            self.tau, self.epsilon_1635, self.sigma_tau, self.sigma_epsilon_1635
        )
        (
            E_calib,
            SEE_inv,
            R2_inv,
            RMSE_inv,
            RRMSE_inv,
            CCC_inv,
        ) = pig.inversion_fit_errors(
            self.tau, self.epsilon_1635, mest_f, covepsilon_est_f
        )
        E_calib = float(E_calib)
        SEE_inv = float(SEE_inv)
        R2_inv = float(R2_inv)
        RMSE_inv = float(RMSE_inv)
        RRMSE_inv = float(RRMSE_inv)
        CCC_inv = float(CCC_inv)
        expected_E_calib = 3.1321319317425775
        expected_SEE = 0.9576169271536646
        expected_R2 = 0.9441049086781923
        expected_RMSE = 2.234439496691885
        expected_RRMSE = 0.04741097494507926
        expected_CCC = 0.9716055450168912
        self.assertAlmostEqual(
            E_calib,
            expected_E_calib,
            self.decimalPlace,
            msg="E_calib test and expected values from the "
            "inversion_fit_errors function do not agree",
        )
        self.assertAlmostEqual(
            SEE_inv,
            expected_SEE,
            self.decimalPlace,
            msg="SEE test and expected values from the "
            "inversion_fit_errors function do not agree",
        )
        self.assertAlmostEqual(
            R2_inv,
            expected_R2,
            self.decimalPlace,
            msg="R2 test and expected values from the "
            "inversion_fit_errors function do not agree",
        )
        self.assertAlmostEqual(
            RMSE_inv,
            expected_RMSE,
            self.decimalPlace,
            msg="RMSE test and expected values from the "
            "inversion_fit_errors function do not agree",
        )
        self.assertAlmostEqual(
            RRMSE_inv,
            expected_RRMSE,
            self.decimalPlace,
            msg="RRMSE test and expected values from the "
            "inversion_fit_errors function do not agree",
        )
        self.assertAlmostEqual(
            CCC_inv,
            expected_CCC,
            self.decimalPlace,
            msg="CCC test and expected values from the "
            "inversion_fit_errors function do not agree",
        )

    def test_inversion_error_plot(self):
        mest_f, _, _ = pig.inversion(
            self.tau, self.epsilon_1635, self.sigma_tau, self.sigma_epsilon_1635
        )
        (
            tau_arr,
            epsilon_arr,
            conf_low,
            conf_up,
            pred_low,
            pred_up,
        ) = pig.inversion_fit_errors_plotting(self.tau, self.epsilon_1635, mest_f)

        self.assertTrue(
            tau_arr[0] == 0, msg="The first value of tau_arr is not as expected."
        )

        self.assertTrue(
            len(tau_arr) == 100, msg="The length of tau_arr is not 100 as expected."
        )

        self.assertAlmostEqual(
            epsilon_arr[0],
            -50.3975642,
            self.decimalPlace,
            msg="The first value of epsilon_arr is not as expected.",
        )

        self.assertAlmostEqual(
            conf_low[0],
            -40.96516294,
            self.decimalPlace,
            msg="The first value of conf_low is not as expected.",
        )

        self.assertAlmostEqual(
            pred_low[0],
            -40.58311073,
            self.decimalPlace,
            msg="The first value of pred_low is not as expected.",
        )


if __name__ == "__main__":
    unittest.main()
