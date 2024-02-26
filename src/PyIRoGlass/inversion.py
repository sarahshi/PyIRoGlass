# %%

__author__ = 'Sarah Shi'

import numpy as np

# %% Functions for performing the Newtonian inversion


def inversion(comp, epsilon, sigma_comp, sigma_epsilon):

    """
    Perform a Newtonian inversion on a given set of composition and absorbance
    coefficient data.

    Parameters:
        comp (np.ndarray): A 1D array containing the composition data.
        epsilon (np.ndarray): A 1D array containing the absorbance coefficient
            data.
        sigma_comp (float): The standard deviation of the composition data.
        sigma_epsilon (float): The standard deviation of the absorbance
            coefficient data.

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
            covepsilon_est_f (np.ndarray): The covariance matrix of the final
                estimate of the absorbance coefficients.
            comp_pre (np.ndarray): A 1D array of the predicted composition
                values based on the final estimate.
            epsilon_pre (np.ndarray): A 1D array of the predicted absorbance
                coefficient values based on the final estimate.
            epsilon_linear (np.ndarray): A 1D array of the predicted absorbance
                coefficient values based on the linear regression estimate.
    """

    M = 2  # Number of calibration parameters
    N = len(comp)  # Number of data points

    # Create a matrix with the composition and a column of ones for intercept
    # calculation
    G = np.array([np.ones(N), comp]).T

    # Solve for calibration parameters using regular least squares
    mls = np.linalg.solve(np.dot(G.T, G), np.dot(G.T, epsilon))

    # Compute covariance matrix for regular least squares solution
    covls = np.linalg.inv(np.linalg.multi_dot([G.T,
                                               np.diag(sigma_epsilon**-2), G]))

    # Combine all parameters into a single vector for use in optimization
    xbar = np.concatenate([epsilon, comp, mls])

    # Trial solution based on regular least squares solution
    xg = xbar

    # Initialize gradient vector
    Fg = np.zeros([N, M*N+M])

    # Initialize covariance matrix
    covx = np.zeros([M*N+M, M*N+M])

    # Set covariance matrix for measurement uncertainties
    covx[0*N:1*N, 0*N:1*N] = np.diag(sigma_epsilon**2)
    covx[1*N:2*N, 1*N:2*N] = np.diag(sigma_comp**2)

    # Set large covariance matrix for model parameters
    scale = 1
    covx[M*N+0, M*N+0] = scale * covls[0, 0]
    covx[M*N+1, M*N+1] = scale * covls[1, 1]

    # Set number of iterations for optimization
    Nit = 100

    # Initialize arrays to store calculated values at each iteration
    epsilon_pre_all = np.zeros([N, Nit])
    epsilon_linear_all = np.zeros([N, Nit])
    mest_all = np.zeros([2, Nit])

    # Perform optimization
    for i in range(0, Nit):
        # Calculate residual vector and its squared norm
        f = -xg[0:N] + (xg[M*N+1]*xg[1*N:2*N]) + (xg[M*N+0]*np.ones(N))
        Ef = np.dot(f.T, f)

        # Print error at first iteration and every 10th iteration
        if (i == 0):
            print('Initial error in implicit equation = ' + str(Ef))
        elif (i % 10 == 0):
            print('Final error in implicit equation = ', Ef)

        # Compute gradient vector
        Fg[0:N, 0:N] = -np.eye(N, N)
        Fg[0:N, N:2*N] = xg[M*N+1] * np.eye(N, N)
        Fg[0:N, M*N+0] = np.ones([N])
        Fg[0:N, M*N+1] = xg[1*N:2*N]

        # Set regularization parameter
        epsi = 0

        # Solve linear system
        left = Fg.T
        right = np.linalg.multi_dot([Fg, covx, Fg.T]) + (epsi*np.eye(N, N))
        solve = np.linalg.solve(right.conj().T, left.conj().T).conj().T
        MO = np.dot(covx, solve)
        xg2 = xbar + np.dot(MO, (np.dot(Fg, (xg-xbar))-f))
        xg = xg2

        # Store some variables for later use
        mest = xg[M*N+0:M*N+M]
        epsilon_pre = xg[0:N]
        epsilon_linear = mest[0] + mest[1]*comp
        epsilon_pre_all[0:N, i] = epsilon_pre[0:N]
        mest_all[0:N, i] = mest
        epsilon_linear_all[0:N, i] = epsilon_linear[0:N]

    # Compute some additional statistics
    MO2 = np.dot(MO, Fg)
    covx_est = np.linalg.multi_dot([MO2, covx, MO2.T])
    covepsilon_est_f = covx_est[0:N, 0:N]
    covm_est_f = covx_est[-2:, -2:]

    mest_f = xg[M*N:M*N+M]

    # Return relevant variables
    return mest_f, covm_est_f, covepsilon_est_f


def least_squares(comp, epsilon, sigma_comp, sigma_epsilon):

    """
    Perform a least squares regression on a given set of composition and
    absorbance coefficient data.

    Parameters:
        comp (np.ndarray): A 1D array containing the composition data.
        epsilon (np.ndarray): A 1D array containing the absorbance coefficient
            data.
        sigma_comp (float): The standard deviation of the composition data.
        sigma_epsilon (float): The standard deviation of the absorbance
            coefficient data.

    Returns:
        Tuple containing the following elements:
            mls (np.ndarray): A 1D array of the least squares estimate of the
                coefficients.
            covls (np.ndarray): The covariance matrix of the least squares
                estimate.
    """

    N = len(comp)  # Number of data points

    # Create a matrix with the composition and a column of ones for
    # intercept calculation
    G = np.array([np.ones(N), comp]).T

    # Solve for calibration parameters using regular least squares
    mls = np.linalg.solve(np.dot(G.T, G), np.dot(G.T, epsilon))

    # Compute covariance matrix for regular least squares solution
    covls = np.linalg.inv(np.linalg.multi_dot([G.T,
                                               np.diag(sigma_epsilon**-2), G]))

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


def calculate_epsilon_inversion(m, composition):

    """
    Calculate epsilon values using coefficients and composition.

    Parameters:
        m (np.ndarray): An array of coefficients.
        composition (np.ndarray): An array of composition data.

    Returns:
        A 1D array of calculated epsilon values.
    """

    return m[0] + m[1] * composition


def calculate_SEE(residuals):

    """
    Calculate the standard error of estimate given an array of residuals.

    Parameters:
        residuals (np.ndarray): An array of residuals.

    Returns:
        A float representing the standard error of estimate.
    """

    return np.sqrt(np.sum(residuals ** 2)) / (len(residuals) - 2)


def calculate_R2(actual_values, predicted_values):

    """
    Calculate the coefficient of determination given actual and predicted
    values.

    Parameters:
        actual_values (np.ndarray): An array of actual values.
        predicted_values (np.ndarray): An array of predicted values.

    Returns:
        A float representing the coefficient of determination.
    """

    y_bar = np.mean(actual_values)
    total_sum_of_squares = np.sum((actual_values - y_bar) ** 2)
    residual_sum_of_squares = np.sum((actual_values - predicted_values) ** 2)
    return 1 - (residual_sum_of_squares / total_sum_of_squares)


def calculate_RMSE(residuals):

    """
    Calculate the root mean squared error given an array of residuals.

    Parameters:
        residuals (np.ndarray): An array of residuals.

    Returns:
        A float representing the root mean squared error.
    """

    return np.sqrt(np.mean(residuals ** 2))


def inversion_fit_errors(comp, epsilon, mest_f, covm_est_f, covepsilon_est_f):

    """
    Calculate error metrics for a given set of data.

    Parameters:
        comp (np.ndarray): A 1D array containing the composition data.
        epsilon: A 1D array containing the absorbance coefficient data.
        mest_f (np.ndarray): A 1D array of the final estimate of the
            coefficients.
        covm_est_f (np.ndarray): The covariance matrix of the final estimate.
        covepsilon_est_f (np.ndarray): The covariance matrix of the final
            estimate of the absorbance coefficients.

    Returns:
        Tuple containing the following elements:
            E_calib: A float representing the error in calibration.
            see_inv: A float representing the standard error of estimate.
            r2_inv: A float representing the coefficient of determination.
            rmse_inv: A float representing the root mean squared error.
    """

    epsilon_final_estimate = calculate_epsilon_inversion(mest_f, comp)
    residuals = epsilon_final_estimate - epsilon
    E_calib = calculate_calibration_error(covepsilon_est_f)
    SEE_inv = calculate_SEE(residuals)
    R2_inv = calculate_R2(epsilon, epsilon_final_estimate)
    RMSE_inv = calculate_RMSE(residuals)

    return E_calib, SEE_inv, R2_inv, RMSE_inv
