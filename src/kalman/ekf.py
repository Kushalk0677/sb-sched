"""
src/kalman/ekf.py

Extended Kalman Filter for nonlinear systems.
Used in Experiment 6 to validate staleness budget generalisation.
The EKF linearises f() and h() at the current estimate to get F_k, H_k,
then uses those Jacobians in the staleness budget computation.
"""

import numpy as np
from typing import Callable
from .kalman_filter import KalmanState, SensorModel


class ExtendedKalmanFilter:
    """
    EKF supporting multi-rate asynchronous sensor fusion.

    System:
        x_{k+1} = f(x_k) + w_k,    w_k ~ N(0, Q)
        y_k^i   = h_i(x_k) + v_k,  v_k ~ N(0, R_i)

    Args:
        f:       State transition function  x -> x
        jac_f:   Jacobian of f wrt x  ->  [n x n]
        Q:       Process noise covariance [n x n]
        sensors: List of (SensorModel, h_i, jac_h_i) tuples
        x0, P0:  Initial state and covariance
    """

    def __init__(
        self,
        f: Callable,
        jac_f: Callable,
        Q: np.ndarray,
        sensors: list,
        x0: np.ndarray,
        P0: np.ndarray,
    ):
        self.f = f
        self.jac_f = jac_f
        self.Q = Q
        self.sensors = sensors  # [(SensorModel, h, jac_h), ...]
        self.sensor_map = {s[0].name: s for s in sensors}
        self.n = x0.shape[0]
        self.state = KalmanState(x=x0.copy(), P=P0.copy())

    def predict(self) -> KalmanState:
        F_k = self.jac_f(self.state.x)
        self.state.x = self.f(self.state.x)
        self.state.P = F_k @ self.state.P @ F_k.T + self.Q
        self.state.step += 1
        return self.state

    def update(self, sensor_name: str, z: np.ndarray) -> KalmanState:
        sensor_model, h, jac_h = self.sensor_map[sensor_name]
        H_k = jac_h(self.state.x)
        R = sensor_model.R
        y = z - h(self.state.x)
        S = H_k @ self.state.P @ H_k.T + R
        K = self.state.P @ H_k.T @ np.linalg.inv(S)
        self.state.x = self.state.x + K @ y
        I_KH = np.eye(self.n) - K @ H_k
        self.state.P = I_KH @ self.state.P @ I_KH.T + K @ R @ K.T
        return self.state

    def predict_covariance(self, steps_ahead: int) -> np.ndarray:
        """
        Conservative covariance prediction using current Jacobian F_k.
        Valid locally around current state; used for staleness budget in EKF.
        """
        F_k = self.jac_f(self.state.x)
        P_pred = self.state.P.copy()
        for _ in range(steps_ahead):
            P_pred = F_k @ P_pred @ F_k.T + self.Q
        return P_pred

    @property
    def trace_P(self) -> float:
        return float(np.trace(self.state.P))
