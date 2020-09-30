import numpy as np
from scipy.stats import norm


class LGSS:
    """Linear Gaussian State Space model. The observation is assumed to be one-dimensional."""

    def __init__(self, T, R, Q, Z, H, a1, P1):
        self.d = T.shape[0]  # State dimension
        self.deta = R.shape[1]  # Second dimension is process noise dim
        self.T = T  # Process model
        self.R = R  # Process noise prefactor
        self.Q = Q  # Process noise covariance
        self.Z = Z  # Measurement model
        self.H = H  # Measurement noise variance
        self.a1 = a1  # Initial state mean
        self.P1 = P1  # Initial state covariance

    def get_params(self):
        """Return all model parameters.

        T, R, Q, Z, H, a1, P1 = model.get_params()
        """
        return self.T, self.R, self.Q, self.Z, self.H, self.a1, self.P1


class kfs_res:
    """Container class to store result of Kalman filter and smoother."""

    def __init__(self, alpha_pred, P_pred, alpha_filt, P_filt, y_pred, F_pred):
        """Initialize with KF results"""
        self.alpha_pred = alpha_pred
        self.P_pred = P_pred
        self.alpha_filt = alpha_filt
        self.P_filt = P_filt
        self.y_pred = y_pred
        self.F_pred = F_pred

    def set_ks_res(self, alpha_sm, V, eps_hat, eps_var, eta_hat, eta_cov):
        """Update to contain also KS results"""
        self.alpha_sm = alpha_sm
        self.V = V
        self.eps_hat = eps_hat
        self.eps_var = eps_var
        self.eta_hat = eta_hat
        self.eta_cov = eta_cov


def kalman_smoother(y, model: LGSS, kf: kfs_res):
    """Kalman (state and disturbance) smoother for LGSS model with one-dimensional observation.

    :param y: (n,) array of observations. May contain nan, which encodes missing observations.
    :param model: LGSS object with the model specification.
    :parma kf: kfs_res object with result from a Kalman filter foward pass.

    :return kfs_res: Container class. The original Kalman filter result is augmented with the following member variables,
        alpha_sm: (d,1,n) array of smoothed state means.
        V: (d,d,n) array of smoothed state covariances.
        eps_hat: (n,) array of smoothed means of observation disturbances.
        eps_var: (n,) array of smoothed variances of observation disturbances.
        eta_hat: (deta,1,n) array of smoothed means of state disturbances.
        eta_cov: (deta,deta,n) array of smoothed covariances of state disturbances.
    """
    d = model.d  # State dimension
    deta = model.deta  # Number of state noise components
    n = len(y)

    # Allocate memory, see DK (4.44)
    r = np.zeros((d, 1, n))
    N = np.zeros((d, d, n))
    alpha_sm = np.zeros((d, 1, n))
    V = np.zeros((d, d, n))

    # Disturbances
    eps_hat = np.zeros(n)  # Observation noise
    eps_var = np.zeros(n)
    eta_hat = np.zeros((deta, 1, n))  # State noise
    eta_cov = np.zeros((deta, deta, n))  # State noise covariance

    # Get all model parameters (for brevity)
    T, R, Q, Z, H, a1, P1 = model.get_params()

    # Get the innovations and their variances from forward pass
    v = y - kf.y_pred
    F = kf.F_pred.copy()

    # Simple way of handling missing observations; treat them as present but with infinite variance!
    ind = np.isnan(v)
    v[ind] = 0.
    F[ind] = np.inf

    # Compute the "L-matrices", DKp87
    L = np.zeros((d, d, n))
    K = np.zeros((d, 1, n))
    for t in range(n):
        K[:, :, t] = kf.P_pred[:, :, t] @ Z.T / F[t]  # Kalman gain (without the leading T that DK use)
        L[:, :, t] = T @ (np.identity(d) - K[:, :, t] @ Z)

    # Initialize. r and N are defined for t=0,...,n-1 in DK,  whereas other quantities are defined for t=1,...,n.
    # Hence, alpha_sm[:,t-1] = \hat alpha_{t} but r[t-1] = r_{t-1}.
    r[:, :, -1] = (Z.T / F[-1]) * v[-1]
    N[:, :, -1] = (Z.T / F[-1]) @ Z
    # This is actually an unnecessary computation, since we simply compute the filter estimates again
    # (= to smoother at time step t=n), but we keep them to agree with the algorithm in DK
    alpha_sm[:, :, -1] = kf.alpha_pred[:, :, -1] + kf.P_pred[:, :, -1] @ r[:, :, -1]
    V[:, :, -1] = kf.P_pred[:, :, -1] - kf.P_pred[:, :, -1] @ N[:, :, -1] @ kf.P_pred[:, :, -1]

    # Disturbances
    eps_hat[-1] = (H / F[-1]) * v[-1]
    eps_var[-1] = H - (H / F[-1]) * H
    eta_cov[:, :, -1] = Q

    for t in np.flip(range(n - 1)):
        # State smoothing
        r[:, :, t] = (Z.T / F[t]) * v[t] + L[:, :, t].T @ r[:, :, t + 1]
        N[:, :, t] = (Z.T / F[t]) @ Z + L[:, :, t].T @ N[:, :, t + 1] @ L[:, :, t]
        alpha_sm[:, :, t] = kf.alpha_pred[:, :, t] + kf.P_pred[:, :, t] @ r[:, :, t]
        V[:, :, t] = kf.P_pred[:, :, t] - kf.P_pred[:, :, t] @ N[:, :, t] @ kf.P_pred[:, :, t]

        # Disturbance smoothing
        eps_hat[t] = H * (v[t] / F[t] - K[:, :, t].T @ T.T @ r[:, :, t + 1])
        eps_var[t] = H - (H / F[t]) * H - H * K[:, :, t].T @ T.T @ N[:, :, t + 1] @ T @ K[:, :, t] * H
        eta_hat[:, :, t] = Q @ R.T @ r[:, :, t + 1]
        eta_cov[:, :, t] = Q - Q @ R.T @ N[:, :, t + 1] @ R @ Q

    kf.set_ks_res(alpha_sm, V, eps_hat, eps_var, eta_hat, eta_cov)

    return kf