"""
src/datasets/models.py

Kalman filter system model (F, Q, H, R, x0, P0) for each dataset.

Modeling scope
──────────────
All three models use a simplified constant-velocity state:

    EuRoC / TUM-VI  [px, py, pz, vx, vy, vz]  6-dim
    KITTI           [px, py, pz, vx, vy, vz, roll, pitch, yaw]  9-dim

IMU data drives the predict step (F propagation) — it is NOT processed
as a full inertial measurement update with bias estimation.  This keeps
the filter linear and lets the staleness-budget guarantee hold exactly.

Camera measurements (EuRoC / TUM-VI) are ground-truth-interpolated
position pseudo-observations — oracle, not physically measured.

GPS measurements (KITTI) are physically real ENU positions converted
from OXTS RTK packets — not oracle.  R is tuned to ~0.1 m RMS consistent
with the KITTI OXTS RT3003 spec (horizontal accuracy ~0.02 m RTK,
~0.1 m post-processed; we use 0.1 m as a conservative figure).

RMSE values from these models measure filtering consistency under the
chosen motion model, not navigation accuracy comparable to full VIO /
GPS-INS fusion systems in the literature.
"""

import numpy as np
from ..kalman.kalman_filter import SensorModel


# ---------------------------------------------------------------------------
# EuRoC / TUM-VI: IMU (predict) + Camera pseudo-measurement (update)
# State: [px, py, pz, vx, vy, vz]  6-dim
# ---------------------------------------------------------------------------

def euroc_model(imu_hz: float = 200.0, cam_hz: float = 20.0):
    """
    Returns (F, Q, sensors, x0, P0) for EuRoC / TUM-VI.

    Camera measurement: 3-D position pseudo-observation from GT interpolation.
    R = 0.01 m² ≈ 10 cm — kept small to reflect oracle accuracy.
    Process noise Q tuned for indoor MAV dynamics at imu_hz.
    P0 initialised near steady-state so warmup violations are minimal.
    """
    dt = 1.0 / imu_hz
    n  = 6  # [px, py, pz, vx, vy, vz]

    F = np.eye(n)
    F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt

    q_pos = 0.001   # m²/step  position diffusion
    q_vel = 0.010   # (m/s)²/step  velocity diffusion
    Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]) * dt

    H_cam = np.zeros((3, n))
    H_cam[0, 0] = H_cam[1, 1] = H_cam[2, 2] = 1.0
    R_cam = np.eye(3) * 0.01   # oracle pseudo-measurement: ~10 cm

    sensors = [SensorModel(name="camera", H=H_cam, R=R_cam, native_hz=cam_hz)]

    x0 = np.zeros(n)
    P0 = np.eye(n) * 0.003   # near steady-state (trace ≈ 0.016 at ss)

    return F, Q, sensors, x0, P0


# ---------------------------------------------------------------------------
# KITTI: IMU (predict) + GPS ENU position (update, physically real)
# State: [px, py, pz, vx, vy, vz, roll, pitch, yaw]  9-dim
#
# Important: KITTILoader yields IMU and GPS at the OXTS packet rate (~10 Hz)
# because pykitti does not expose an independent high-rate IMU stream.
# Set imu_hz=10 (default here) to match the actual predict-step rate.
# ---------------------------------------------------------------------------

def kitti_model(imu_hz: float = 10.0, gps_hz: float = 10.0):
    """
    Returns (F, Q, sensors, x0, P0) for KITTI GPS/IMU fusion.

    GPS measurement: physically real ENU position in metres.
    R = 0.01 m² ≈ 10 cm RMS — conservative for OXTS RT3003 post-processed.
    imu_hz defaults to 10 to match the OXTS packet rate from KITTILoader.
    Process noise Q tuned for on-road vehicle dynamics.
    P0 initialised near steady-state to avoid warmup violations.
    """
    dt = 1.0 / imu_hz
    n  = 9  # [px, py, pz, vx, vy, vz, roll, pitch, yaw]

    F = np.eye(n)
    F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt

    q_pos = 0.005   # m²/step
    q_vel = 0.050   # (m/s)²/step  — higher than EuRoC (faster dynamics)
    q_att = 0.001   # rad²/step
    Q = np.diag([q_pos]*3 + [q_vel]*3 + [q_att]*3) * dt

    H_gps = np.zeros((3, n))
    H_gps[0, 0] = H_gps[1, 1] = H_gps[2, 2] = 1.0
    # OXTS RT3003: ~0.02 m RTK, ~0.10 m post-processed (conservative)
    R_gps = np.eye(3) * 0.01   # 0.1 m RMS → variance = 0.01 m²

    sensors = [SensorModel(name="gps", H=H_gps, R=R_gps, native_hz=gps_hz)]

    x0 = np.zeros(n)
    P0 = np.eye(n) * 0.010   # near steady-state for KITTI model

    return F, Q, sensors, x0, P0


def get_model(dataset: str):
    """Factory: return model builder for named dataset."""
    dataset = dataset.lower()
    if dataset == "euroc":
        return euroc_model
    elif dataset == "kitti":
        return kitti_model
    elif dataset in ("tumvi", "tum_vi", "tum-vi"):
        return euroc_model   # same sensor pair as EuRoC
    elif dataset in ("uci_gas", "uci-gas", "ucigas", "onemonth_gas", "onemonth-gas", "onemonthgas"):
        return gas_signal_model
    else:
        raise ValueError(f"No model defined for dataset: {dataset!r}")

# ---------------------------------------------------------------------------
# Gas datasets: local linear trend model for one scalar sensor response
# State: [signal, drift]
# ---------------------------------------------------------------------------

def gas_signal_model(sample_hz: float = 1.0, measurement_hz: float | None = None):
    """Returns a lightweight 1-D gas-response tracking model.

    The model tracks the selected gas sensor response as a latent signal with a
    slowly varying drift term. It is intentionally simple: the aim is to test
    the scheduler under real drift/noise, not to build a chemistry-specific
    estimator.
    """
    dt = 1.0 / sample_hz
    F = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    # Slightly generous process noise so covariance grows when updates are skipped
    q_level = 2.5e-3
    q_drift = 5.0e-4
    Q = np.diag([q_level, q_drift]) * dt

    H = np.array([[1.0, 0.0]], dtype=float)
    R = np.array([[5.0e-3]], dtype=float)
    native_hz = sample_hz if measurement_hz is None else measurement_hz
    sensors = [SensorModel(name='gas', H=H, R=R, native_hz=native_hz)]

    x0 = np.zeros(2, dtype=float)
    P0 = np.diag([0.05, 0.01]).astype(float)
    return F, Q, sensors, x0, P0
