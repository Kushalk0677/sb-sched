"""
src/datasets/models.py

Kalman filter system model (F, Q, H, R) for each dataset.
These are the standard models used in VIO/navigation literature.
"""

import numpy as np
from ..kalman.kalman_filter import SensorModel


# ---------------------------------------------------------------------------
# EuRoC / TUM-VI: IMU + Camera (position-only update from camera)
# State: [px, py, pz, vx, vy, vz]  (6-dim simplified model)
# High-rate sensor: IMU (predict step)
# Low-rate sensor:  Camera (position update at 20 Hz)
# ---------------------------------------------------------------------------

def euroc_model(imu_hz: float = 200.0, cam_hz: float = 20.0):
    """
    Returns (F, Q, sensors, x0, P0) for EuRoC/TUM-VI.
    Simplified constant-velocity model; IMU drives the predict step.
    """
    dt = 1.0 / imu_hz
    n = 6  # [px, py, pz, vx, vy, vz]

    # State transition: constant velocity
    F = np.eye(n)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    # Process noise (tuned for indoor MAV dynamics)
    q_pos = 0.001
    q_vel = 0.01
    Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel]) * dt

    # Camera: observes position only
    H_cam = np.zeros((3, n))
    H_cam[0, 0] = 1.0
    H_cam[1, 1] = 1.0
    H_cam[2, 2] = 1.0
    R_cam = np.eye(3) * 0.01   # ~1 cm camera position noise

    sensors = [
        SensorModel(name="camera", H=H_cam, R=R_cam, native_hz=cam_hz),
    ]

    x0 = np.zeros(n)
    P0 = np.eye(n) * 0.1

    return F, Q, sensors, x0, P0


# ---------------------------------------------------------------------------
# KITTI: IMU (high-rate) + GPS (low-rate)
# State: [px, py, pz, vx, vy, vz, roll, pitch, yaw]  (9-dim)
# High-rate: IMU angular velocity + acceleration (predict step)
# Low-rate:  GPS position (10 Hz)
# ---------------------------------------------------------------------------

def kitti_model(imu_hz: float = 100.0, gps_hz: float = 10.0):
    """
    Returns (F, Q, sensors, x0, P0) for KITTI GPS/IMU fusion.
    Simplified constant-velocity model; orientation kept flat for GPS fusion.
    """
    dt = 1.0 / imu_hz
    n = 9  # [px, py, pz, vx, vy, vz, roll, pitch, yaw]

    F = np.eye(n)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt

    q_pos = 0.005
    q_vel = 0.05
    q_att = 0.001
    Q = np.diag([q_pos]*3 + [q_vel]*3 + [q_att]*3) * dt

    # GPS: observes position in Cartesian (lat/lon converted externally)
    H_gps = np.zeros((3, n))
    H_gps[0, 0] = 1.0
    H_gps[1, 1] = 1.0
    H_gps[2, 2] = 1.0
    R_gps = np.eye(3) * 0.25   # ~50 cm GPS noise (consumer grade)

    sensors = [
        SensorModel(name="gps", H=H_gps, R=R_gps, native_hz=gps_hz),
    ]

    x0 = np.zeros(n)
    P0 = np.eye(n) * 1.0

    return F, Q, sensors, x0, P0


def get_model(dataset: str):
    """Factory: returns model builder for named dataset."""
    if dataset.lower() == "euroc":
        return euroc_model
    elif dataset.lower() == "kitti":
        return kitti_model
    elif dataset.lower() in ("tumvi", "tum_vi", "tum-vi"):
        return euroc_model  # Same sensor pair as EuRoC
    else:
        raise ValueError(f"No model defined for dataset: {dataset}")
