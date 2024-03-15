# %%

__author__ = "Sarah Shi"

import numpy as np
import pandas as pd
from scipy import stats

# %% Functions for performing the Newtonian inversion


def inversion(x, y, sigma_x, sigma_y, intercept_zero=False):

    """
    Perform a Newtonian inversion on a given set of x and y data.

    Parameters:
        x (np.ndarray): A 1D array containing the x (composition) data.
        y (np.ndarray): A 1D array containing the y (absorbance coefficient)
            data.
        sigma_x (float): The standard deviation of the x (composition) data.
        sigma_y (float): The standard deviation of the y (absorbance
            coefficient) data.
        intercept_zero (boolean): Determines if the intercept is explicitly
            set to zero. Setting intercept_zero to True forces the intecept to
            zero. Setting intercept_zero to False allows the intercept to be
            normally estimated as part of the regression.

    Returns:
        Tuple containing the following elements:
            mls (np.ndarray): A 1D array of the least squares estimate of
                the coefficients.
            mest_f (np.ndarray): A 1D array of the final estimate of the
                coefficients.
            covls (np.ndarray): The covariance matrix of the least squares
                estimate.
            covm_est_f (np.ndarray): The covariance matrix of the final
                estimate.
            covy_est_f (np.ndarray): The covariance matrix of the final
                estimate of the absorbance coefficients.
            x_pre (np.ndarray): A 1D array of the predicted x values based
                on the final estimate.
            y_pre (np.ndarray): A 1D array of the predicted absorbance
                coefficient values based on the final estimate.
            y_linear (np.ndarray): A 1D array of the predicted absorbance
                coefficient values based on the linear regression estimate.
    """

    M = 2  # Number of calibration parameters
    N = len(x)  # Number of data points

    # Create a matrix with the x data and a column of ones for intercept
    # calculation
    G = np.array([np.ones(N), x]).T

    # Solve for calibration parameters using regular least squares
    mls = np.linalg.solve(np.dot(G.T, G), np.dot(G.T, y))

    if intercept_zero:
        mls[0] = 0.0

    # Compute covariance matrix for regular least squares solution
    covls = np.linalg.inv(np.linalg.multi_dot([G.T, np.diag(sigma_y**-2), G]))

    # Combine all parameters into a single vector for use in optimization
    xbar = np.concatenate([y, x, mls])

    # Trial solution based on regular least squares solution
    xg = xbar

    # Initialize gradient vector
    Fg = np.zeros([N, M * N + M])

    # Initialize covariance matrix
    covx = np.zeros([M * N + M, M * N + M])

    # Set covariance matrix for measurement uncertainties
    covx[0 * N : 1 * N, 0 * N : 1 * N] = np.diag(sigma_y**2)
    covx[1 * N : 2 * N, 1 * N : 2 * N] = np.diag(sigma_x**2)

    # Set large covariance matrix for model parameters
    scale = 1
    if intercept_zero:
        covx[M * N + 0, M * N + 0] = 1e-3 * scale * covls[0, 0]
    else:
        covx[M * N + 0, M * N + 0] = scale * covls[0, 0]
    covx[M * N + 1, M * N + 1] = scale * covls[1, 1]

    # Set number of iterations for optimization
    Nit = 100

    # Initialize arrays to store calculated values at each iteration
    y_pre_all = np.zeros([N, Nit])
    y_linear_all = np.zeros([N, Nit])
    mest_all = np.zeros([2, Nit])

    # Perform optimization
    for i in range(0, Nit):
        # Calculate residual vector and its squared norm
        f = (
            -xg[0:N]
            + (xg[M * N + 1] * xg[1 * N : 2 * N])
            + (xg[M * N + 0] * np.ones(N))
        )
        Ef = np.dot(f.T, f)

        # Print error at first iteration and every 10th iteration
        if i == 0:
            print("Initial error in implicit equation = " + str(Ef))
        elif i % 10 == 0:
            print("Final error in implicit equation = ", Ef)

        # Compute gradient vector
        Fg[0:N, 0:N] = -np.eye(N, N)
        Fg[0:N, N : 2 * N] = xg[M * N + 1] * np.eye(N, N)
        Fg[0:N, M * N + 0] = np.ones([N])
        Fg[0:N, M * N + 1] = xg[1 * N : 2 * N]

        # Set regularization parameter
        epsi = 0

        # Solve linear system
        left = Fg.T
        right = np.linalg.multi_dot([Fg, covx, Fg.T]) + (epsi * np.eye(N, N))
        solve = np.linalg.solve(right.conj().T, left.conj().T).conj().T
        MO = np.dot(covx, solve)
        xg2 = xbar + np.dot(MO, (np.dot(Fg, (xg - xbar)) - f))
        xg = xg2

        # Store some variables for later use
        mest = xg[M * N + 0 : M * N + M]
        y_pre = xg[0:N]
        y_linear = mest[0] + mest[1] * x
        y_pre_all[0:N, i] = y_pre[0:N]
        mest_all[0:N, i] = mest
        y_linear_all[0:N, i] = y_linear[0:N]

    # Compute some additional statistics
    MO2 = np.dot(MO, Fg)
    covx_est = np.linalg.multi_dot([MO2, covx, MO2.T])
    covy_est_f = covx_est[0:N, 0:N]
    covm_est_f = covx_est[-2:, -2:]

    mest_f = xg[M * N : M * N + M]

    # Return relevant variables
    return mest_f, covm_est_f, covy_est_f


def least_squares(x, y, sigma_y):

    """
    Perform a least squares regression on a given set of composition and
    absorbance coefficient data.

    Parameters:
        x (np.ndarray): A 1D array containing the x (composition) data.
        y (np.ndarray): A 1D array containing the y (absorbance coefficient)
            data.
        sigma_y (float): The standard deviation of the absorbance
            coefficient data.

    Returns:
        Tuple containing the following elements:
            mls (np.ndarray): A 1D array of the least squares estimate of the
                coefficients.
            covls (np.ndarray): The covariance matrix of the least squares
                estimate.
    """

    N = len(x)  # Number of data points

    # Create a matrix with the composition and a column of ones for
    # intercept calculation
    G = np.array([np.ones(N), x]).T

    # Solve for calibration parameters using regular least squares
    mls = np.linalg.solve(np.dot(G.T, G), np.dot(G.T, y))

    # Compute covariance matrix for regular least squares solution
    covls = np.linalg.inv(np.linalg.multi_dot([G.T, np.diag(sigma_y**-2), G]))

    return mls, covls


def calculate_calibration_error(covariance_matrix):

    """
    Calculate the calibration error based on the diagonal elements of
    a covariance matrix.

    Parameters:
        covariance_matrix (np.ndarray): A covariance matrix.

    Returns:
        A float representing the calibration error.
    """

    diagonal = np.diag(covariance_matrix)
    return 2 * np.sqrt(np.mean(diagonal))


def calculate_y_inversion(m, x):

    """
    Calculate y values using coefficients and composition.

    Parameters:
        m (np.ndarray): An array of coefficients.
        x (np.ndarray): An array of x (composition) data.

    Returns:
        A 1D array of calculated y values.
    """

    return m[0] + m[1] * x


def calculate_SEE(residuals):

    """
    Calculate the standard error of estimate given an array of residuals.

    Parameters:
        residuals (np.ndarray): An array of residuals.

    Returns:
        A float representing the standard error of estimate.
    """

    return np.sqrt(np.sum(residuals**2)) / (len(residuals) - 2)


def calculate_R2(true_y, pred_y):

    """
    Calculate the coefficient of determination given actual and predicted
    values.

    Parameters:
        true_y (np.ndarray): An array of actual values.
        pred_y (np.ndarray): An array of predicted values.

    Returns:
        A float representing the coefficient of determination.
    """

    y_bar = np.mean(true_y)
    total_sum_of_squares = np.sum((true_y - y_bar) ** 2)
    residual_sum_of_squares = np.sum((true_y - pred_y) ** 2)
    return 1 - (residual_sum_of_squares / total_sum_of_squares)


def calculate_RMSE(residuals):

    """
    Calculate the root mean squared error given an array of residuals.

    Parameters:
        residuals (np.ndarray): An array of residuals.

    Returns:
        A float representing the root mean squared error.
    """

    return np.sqrt(np.mean(residuals**2))


def calculate_RRMSE(true_y, pred_y):

    """
    Calculate the relative root mean squared error (RRMSE) between
    true and predicted values.

    Parameters:
        true_y (np.ndarray): An array of true values.
        pred_y (np.ndarray): An array of predicted values.

    Returns:
        A float representing the relative root mean squared error.

    """

    num = np.sum(np.square(true_y - pred_y))
    den = np.sum(np.square(pred_y))
    squared_error = num / den
    rrmse_loss = np.sqrt(squared_error)

    return rrmse_loss


def calculate_CCC(true_y, pred_y):

    """
    Calculate the Concordance Correlation Coefficient (CCC) between
    true and predicted values.

    Parameters:
        true_y (np.ndarray): An array of true values.
        pred_y (np.ndarray): An array of predicted values.

    Returns:
        A float representing the CCC.

    """

    # Remove NaNs
    df = pd.DataFrame({"true_y": true_y, "pred_y": pred_y})
    df = df.dropna()
    true_y = df["true_y"]
    pred_y = df["pred_y"]
    # Pearson product-moment correlation coefficients
    cor = np.corrcoef(true_y, pred_y)[0][1]
    # Mean
    mean_true = np.mean(true_y)
    mean_pred = np.mean(pred_y)
    # Variance
    var_true = np.var(true_y)
    var_pred = np.var(pred_y)
    # Standard deviation
    sd_true = np.std(true_y)
    sd_pred = np.std(pred_y)
    # Calculate CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2

    return numerator / denominator


def inversion_fit_errors(x, y, mest_f, covy_est_f):

    """
    Calculate error metrics for a given set of data.

    Parameters:
        x (np.ndarray): A 1D array containing the x (composition) data.
        y: A 1D array containing the absorbance coefficient data.
        mest_f (np.ndarray): A 1D array of the final estimate of the
            coefficients.
        covy_est_f (np.ndarray): The covariance matrix of the final
            estimate of the absorbance coefficients.

    Returns:
        Tuple containing the following elements:
            E_calib: A float representing the error in calibration.
            see_inv: A float representing the standard error of estimate.
            r2_inv: A float representing the coefficient of determination.
            rmse_inv: A float representing the root mean squared error.
    """

    y_final_estimate = calculate_y_inversion(mest_f, x)
    residuals = y_final_estimate - y
    E_calib = calculate_calibration_error(covy_est_f)
    SEE_inv = calculate_SEE(residuals)
    R2_inv = calculate_R2(y, y_final_estimate)
    RMSE_inv = calculate_RMSE(residuals)
    RRMSE_inv = calculate_RRMSE(y, y_final_estimate)
    CCC_inv = calculate_CCC(y, y_final_estimate)

    return E_calib, SEE_inv, R2_inv, RMSE_inv, RRMSE_inv, CCC_inv


def inversion_fit_errors_plotting(x, y, mest_f):

    """
    Generate data for plotting the inversion fit along with its confidence and
    prediction intervals. This function calculates the fitted line using the
    inversion method, along with the corresponding confidence and prediction
    intervals for the regression. These intervals provide an estimation of the
    uncertainty associated with the regression fit and future predictions,
    respectively.

    Parameters:
        x (np.ndarray): A 1D array containing the independent variable data.
        y (np.ndarray): A 1D array containing the dependent variable data.
        mest_f (np.ndarray): A 1D array containing the model parameters
            estimated by the inversion method.

    Returns:
        line_x (np.ndarray): A 1D array of 100 linearly spaced values between
            0 and 1, representing the independent variable values for plotting
            the fitted line and intervals.
        line_y (np.ndarray): The y values of the fitted line evaluated at
            `line_x`, using the inversion method.
        conf_lower (np.ndarray): The lower bound of the confidence interval for
            the fitted line, evaluated at `line_x`.
        conf_upper (np.ndarray): The upper bound of the confidence interval for
            the fitted line, evaluated at `line_x`.
        pred_lower (np.ndarray): The lower bound of the prediction interval for
            the fitted line, evaluated at `line_x`.
        pred_upper (np.ndarray): The upper bound of the prediction interval for
            the fitted line, evaluated at `line_x`.

    """

    n = len(y)
    line_x = np.linspace(0, np.max(np.ceil(x)), 100)
    line_y = calculate_y_inversion(mest_f, line_x)

    y_inv = calculate_y_inversion(mest_f, x)
    x_m = np.mean(x)

    y_hat = y_inv
    ssresid = np.sum((y - y_hat) ** 2)
    ssxx = sum((x - x_m) ** 2)

    ttest = stats.t.ppf(((1 - 0.68) / 2), n - 2)
    se = np.sqrt(ssresid / (n - 2))

    conf_upper = line_y + (ttest * se *
                           np.sqrt(1 / n + (line_x - x_m) ** 2 / ssxx))
    conf_lower = line_y - (ttest * se *
                           np.sqrt(1 / n + (line_x - x_m) ** 2 / ssxx))

    pred_upper = line_y + (ttest * se *
                           np.sqrt(1 + 1 / n + (line_x - x_m) ** 2 / ssxx))
    pred_lower = line_y - (ttest * se *
                           np.sqrt(1 + 1 / n + (line_x - x_m) ** 2 / ssxx))

    return line_x, line_y, conf_lower, conf_upper, pred_lower, pred_upper
