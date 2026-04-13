"""
src/kalman/kalman_filter.py

Linear Kalman filter with support for multi-rate asynchronous updates.
Each sensor has its own measurement model (H_i, R_i) and can be updated
independently at any time step.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KalmanState:
    """Holds the current Kalman filter state estimate and covariance."""
    x: np.ndarray          # State vector  [n x 1]
    P: np.ndarray          # Error covariance matrix [n x n]
    t: float = 0.0         # Current timestamp (seconds)
    step: int = 0          # Discrete timestep counter


@dataclass
class SensorModel:
    """Measurement model for a single sensor."""
    name: str
    H: np.ndarray          # Observation matrix [m x n]
    R: np.ndarray          # Measurement noise covariance [m x m]
    native_hz: float       # Native sampling rate in Hz


class KalmanFilter:
    """
    Linear Kalman filter supporting multi-rate asynchronous sensor fusion.

    System model:
        x_{k+1} = F x_k + w_k,    w_k ~ N(0, Q)
        y_k^i   = H_i x_k + v_k,  v_k ~ N(0, R_i)

    Args:
        F:  State transition matrix [n x n]
        Q:  Process noise covariance [n x n]
        sensors: List of SensorModel objects
        x0: Initial state [n x 1]
        P0: Initial covariance [n x n]
    """

    def __init__(
        self,
        F: np.ndarray,
        Q: np.ndarray,
        sensors: list[SensorModel],
        x0: np.ndarray,
        P0: np.ndarray,
    ):
        self.F = F
        self.Q = Q
        self.sensors = {s.name: s for s in sensors}
        self.n = F.shape[0]
        self.state = KalmanState(x=x0.copy(), P=P0.copy())

    def predict(self, dt_steps: int = 1) -> KalmanState:
        """
        Propagate state forward by dt_steps discrete timesteps.
        Applies: x = F^dt x,  P = F^dt P (F^dt)^T + sum_k Q
        For dt_steps > 1 this accumulates process noise correctly.
        """
        Fk = np.linalg.matrix_power(self.F, dt_steps)
        # Accumulated process noise (exact for time-invariant F, Q)
        Qk = self._accumulated_Q(dt_steps)
        self.state.x = Fk @ self.state.x
        self.state.P = Fk @ self.state.P @ Fk.T + Qk
        self.state.step += dt_steps
        return self.state

    def update(self, sensor_name: str, z: np.ndarray) -> KalmanState:
        """
        Incorporate a measurement from a named sensor.

        Args:
            sensor_name: Key matching a SensorModel.name
            z: Measurement vector [m x 1]
        Returns:
            Updated KalmanState
        """
        sensor = self.sensors[sensor_name]
        H, R = sensor.H, sensor.R
        # Innovation
        y = z - H @ self.state.x
        # Innovation covariance
        S = H @ self.state.P @ H.T + R
        # Kalman gain
        K = self.state.P @ H.T @ np.linalg.inv(S)
        # State update
        self.state.x = self.state.x + K @ y
        # Covariance update (Joseph form for numerical stability)
        I_KH = np.eye(self.n) - K @ H
        self.state.P = I_KH @ self.state.P @ I_KH.T + K @ R @ K.T
        return self.state

    def predict_covariance(self, steps_ahead: int) -> np.ndarray:
        """
        Return the predicted covariance k steps into the future
        WITHOUT updating internal state. Used by the staleness budget.

        Args:
            steps_ahead: Number of predict steps to simulate
        Returns:
            Predicted P matrix [n x n]
        """
        Fk = np.linalg.matrix_power(self.F, steps_ahead)
        Qk = self._accumulated_Q(steps_ahead)
        return Fk @ self.state.P @ Fk.T + Qk

    def _accumulated_Q(self, steps: int) -> np.ndarray:
        """
        Compute sum_{k=0}^{steps-1} F^k Q (F^k)^T  (process noise accumulation).
        Uses iterative computation; for large steps consider closed-form via Lyapunov.
        """
        Qk = np.zeros_like(self.Q)
        Fk = np.eye(self.n)
        for _ in range(steps):
            Qk += Fk @ self.Q @ Fk.T
            Fk = self.F @ Fk
        return Qk

    @property
    def trace_P(self) -> float:
        """Scalar quality metric: trace of current covariance."""
        return float(np.trace(self.state.P))

    def copy_state(self) -> KalmanState:
        return KalmanState(
            x=self.state.x.copy(),
            P=self.state.P.copy(),
            t=self.state.t,
            step=self.state.step,
        )
