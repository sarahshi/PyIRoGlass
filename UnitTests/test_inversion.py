import unittest
import numpy as np
import pandas as pd
import PyIRoGlass as pig



class test_inversion(unittest.TestCase):

    def setUp(self): 

        self.tau = np.array([0.62733751, 0.859 , 0.86 , 0.708 , 0.74332446, 0.746, 0.795, 0.859, 0.79949467])
        self.sigma_tau = self.tau * 0.025 
        self.epsilon_1635 = np.array([25., 56., 55., 42., 40.8, 42.34, 52.05, 55., 49.])
        self.sigma_epsilon_1635 = self.epsilon_1635 * 0.05
        self.decimalPlace = 5
        self.cov = np.identity(2)

    def test_epsilon_invert(self):

        mest_f, covm_est_f, covepsilon_est_f = pig.Inversion(self.tau, self.epsilon_1635, self.sigma_tau, self.sigma_epsilon_1635)
        mls, covls = pig.Least_Squares(self.tau, self.epsilon_1635, self.sigma_tau, self.sigma_epsilon_1635)
        m0 = float(mest_f[0])
        m1 = float(mest_f[1])
        expected_m0 = -50.3975642
        expected_m1 = 124.2505339
        m0_ls = float(mls[0])
        m1_ls = float(mls[1])
        expected_m0_ls = -49.05342621
        expected_m1_ls = 122.71710923
        self.assertAlmostEqual(m0, expected_m0, self.decimalPlace, msg="m0 test and expected values from the Inversion function do not agree")
        self.assertAlmostEqual(m1, expected_m1, self.decimalPlace, msg="m1 test and expected values from the Inversion function do not agree")
        self.assertAlmostEqual(m0_ls, expected_m0_ls, self.decimalPlace, msg="m0 test and expected values from the Least_Squares function do not agree")
        self.assertAlmostEqual(m1_ls, expected_m1_ls, self.decimalPlace, msg="m1 test and expected values from the Least_Squares function do not agree")

    def test_calibration_error(self): 
        
        e_calib = pig.Calculate_Calibration_Error(self.cov)
        expected_e_calib = 2
        self.assertAlmostEqual(e_calib, expected_e_calib, self.decimalPlace, msg="e_calib test and expected values from the Calculate_Calibration_Error function do not agree")

    def test_epsilon(self): 
        
        m = np.array([0, 1])
        comp = 1 
        eps = pig.Calculate_Epsilon(m, comp)
        expected_eps = 1
        self.assertAlmostEqual(eps, expected_eps, self.decimalPlace, msg="epsilon test and expected values from the Calculate_Epsilon function do not agree")

    def test_residuals(self): 
        
        residuals = np.array([0, 1, 2])
        see = pig.Calculate_SEE(residuals)
        expected_see = 2.23606797749979

        rmse = pig.Calculate_RMSE(residuals)
        expected_rmse = 1.2909944487358056

        self.assertAlmostEqual(see, expected_see, self.decimalPlace, msg="SEE test and expected values from the Calculate_SEE function do not agree")
        self.assertAlmostEqual(rmse, expected_rmse, self.decimalPlace, msg="RMSE test and expected values from the Calculate_RMSE function do not agree")

    def test_r2(self): 
        
        values = np.array([0, 1, 2])
        r2 = pig.Calculate_R2(values, values)
        expected_r2 = 1.0
        self.assertAlmostEqual(r2, expected_r2, self.decimalPlace, msg="R2 test and expected values from the Calculate_R2 function do not agree")


    def test_epsilon_invert_errors(self):

        mest_f, covm_est_f, covepsilon_est_f = pig.Inversion(self.tau, self.epsilon_1635, self.sigma_tau, self.sigma_epsilon_1635)
        E_calib, SEE_inv, R2_inv, RMSE_inv = pig.Inversion_Fit_Errors(self.tau, self.epsilon_1635, mest_f, covm_est_f, covepsilon_est_f)
        E_calib = float(E_calib)
        RMSE_inv = float(RMSE_inv)
        expected_E_calib = 3.1321319317425775
        expected_RMSE = 2.234439496691885
        self.assertAlmostEqual(E_calib, expected_E_calib, self.decimalPlace, msg="E_calib test and expected values from the Inversion_Fit_Errors function do not agree")
        self.assertAlmostEqual(RMSE_inv, expected_RMSE, self.decimalPlace, msg="RMSE test and expected values from the Inversion_Fit_Errors function do not agree")


if __name__ == '__main__':
     unittest.main()