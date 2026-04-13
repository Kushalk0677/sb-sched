"""
src/datasets/loaders.py

Data loaders for EuRoC MAV, KITTI Raw, and TUM-VI datasets.
Each loader returns a standardised stream of (timestamp, sensor_name, measurement)
tuples so that the experiment runner is dataset-agnostic.
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator


@dataclass
class SensorReading:
    timestamp: float        # Seconds
    sensor_name: str        # e.g. "imu", "camera", "gps", "lidar"
    z: np.ndarray           # Raw measurement vector
    ground_truth: np.ndarray | None = None  # Position [x, y, z] if available


# ---------------------------------------------------------------------------
# EuRoC MAV Loader
# ---------------------------------------------------------------------------

class EuRoCLoader:
    """
    Loads EuRoC MAV sequences.
    Returns IMU (200 Hz) and camera pseudo-measurements (20 Hz).
    Camera measurements are represented as position+orientation updates
    derived from the VIO ground truth for Kalman update purposes.

    Sequence root layout:
        {root}/{sequence}/
            mav0/
                imu0/data.csv          (timestamp, gx, gy, gz, ax, ay, az)
                cam0/data.csv          (timestamp, image_filename)
            state_groundtruth_estimate0/data.csv
    """

    IMU_COLS = slice(1, 7)   # gx, gy, gz, ax, ay, az
    GT_POS_COLS = slice(1, 4)  # px, py, pz

    def __init__(self, root: str, sequence: str):
        self.root = Path(root) / sequence / "mav0"
        self.gt_file = (Path(root) / sequence /
                        "state_groundtruth_estimate0" / "data.csv")

    def _load_csv(self, path: Path) -> np.ndarray:
        return np.loadtxt(path, delimiter=",", skiprows=1)

    def stream(self) -> Iterator[SensorReading]:
        """Yield sensor readings in chronological order."""
        imu_data = self._load_csv(self.root / "imu0" / "data.csv")
        cam_data = self._load_csv(self.root / "cam0" / "data.csv")
        gt_data  = self._load_csv(self.gt_file)

        # Build GT interpolator (position only for simplicity)
        gt_t   = gt_data[:, 0] * 1e-9   # ns -> s
        gt_pos = gt_data[:, 1:4]

        # IMU readings
        imu_readings = []
        for row in imu_data:
            t = row[0] * 1e-9
            z = row[self.IMU_COLS]
            imu_readings.append(SensorReading(t, "imu", z))

        # Camera readings (use GT position interpolated at camera timestamp)
        cam_readings = []
        for row in cam_data:
            t = row[0] * 1e-9
            pos = np.interp(t, gt_t,
                            gt_pos[:, 0]), np.interp(t, gt_t, gt_pos[:, 1]), \
                  np.interp(t, gt_t, gt_pos[:, 2])
            z = np.array(pos)
            gt_at_t = z.copy()
            cam_readings.append(SensorReading(t, "camera", z, gt_at_t))

        # Merge and sort
        all_readings = imu_readings + cam_readings
        all_readings.sort(key=lambda r: r.timestamp)
        yield from all_readings

    def ground_truth_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (timestamps, positions) arrays from GT file."""
        gt = self._load_csv(self.gt_file)
        return gt[:, 0] * 1e-9, gt[:, 1:4]


# ---------------------------------------------------------------------------
# KITTI Raw Loader
# ---------------------------------------------------------------------------

class KITTILoader:
    """
    Loads KITTI raw sequences.
    Returns IMU (100 Hz) and GPS (10 Hz) readings.

    Sequence layout:
        {root}/{date}/{date}_drive_{seq}_sync/
            oxts/data/           (IMU + GPS combined, one file per frame)
            velodyne_points/     (LiDAR, optional)
    """

    def __init__(self, root: str, date: str, drive: str):
        try:
            import pykitti
            self.data = pykitti.raw(root, date, drive)
        except ImportError:
            raise ImportError("pykitti not installed: pip install pykitti")
        self.root = Path(root)

    def stream(self) -> Iterator[SensorReading]:
        """
        Yield GPS (position) and IMU (angular rate + accel) readings.
        GPS is taken at the OXTS 10 Hz rate.
        IMU is interpolated to 100 Hz from the OXTS inertial readings.
        """
        readings = []

        for i, oxts in enumerate(self.data.oxts):
            t = self.data.timestamps[i].timestamp()
            packet = oxts.packet

            # GPS position as (lat, lon, alt) -> (x, y, z) via UTM
            gps_z = np.array([packet.lat, packet.lon, packet.alt])
            readings.append(SensorReading(t, "gps", gps_z, gps_z.copy()))

            # IMU angular velocities and accelerations
            imu_z = np.array([packet.wx, packet.wy, packet.wz,
                               packet.ax, packet.ay, packet.az])
            readings.append(SensorReading(t, "imu", imu_z))

        readings.sort(key=lambda r: r.timestamp)
        yield from readings

    def ground_truth_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (timestamps, lat/lon/alt) from OXTS."""
        ts = np.array([t.timestamp() for t in self.data.timestamps])
        pos = np.array([[o.packet.lat, o.packet.lon, o.packet.alt]
                        for o in self.data.oxts])
        return ts, pos


# ---------------------------------------------------------------------------
# TUM-VI Loader
# ---------------------------------------------------------------------------

class TUMVILoader:
    """
    Loads TUM-VI sequences.
    Format is similar to EuRoC: CSV files for IMU and camera timestamps,
    with mocap ground truth.

    Sequence layout:
        {root}/{sequence}/
            mav0/
                imu0/data.csv
                cam0/data.csv
            dso/
                gt_imu.csv    (mocap ground truth at IMU rate)
    """

    def __init__(self, root: str, sequence: str):
        self.root = Path(root) / sequence / "mav0"
        self.gt_file = Path(root) / sequence / "dso" / "gt_imu.csv"

    def _load_csv(self, path: Path) -> np.ndarray:
        return np.loadtxt(path, delimiter=",", skiprows=1)

    def stream(self) -> Iterator[SensorReading]:
        imu_data = self._load_csv(self.root / "imu0" / "data.csv")
        cam_data = self._load_csv(self.root / "cam0" / "data.csv")
        gt_data  = self._load_csv(self.gt_file)

        gt_t   = gt_data[:, 0] * 1e-9
        gt_pos = gt_data[:, 1:4]

        readings = []

        for row in imu_data:
            t = row[0] * 1e-9
            z = row[1:7]
            readings.append(SensorReading(t, "imu", z))

        for row in cam_data:
            t = row[0] * 1e-9
            pos = (np.interp(t, gt_t, gt_pos[:, 0]),
                   np.interp(t, gt_t, gt_pos[:, 1]),
                   np.interp(t, gt_t, gt_pos[:, 2]))
            z = np.array(pos)
            readings.append(SensorReading(t, "camera", z, z.copy()))

        readings.sort(key=lambda r: r.timestamp)
        yield from readings

    def ground_truth_positions(self) -> tuple[np.ndarray, np.ndarray]:
        gt = self._load_csv(self.gt_file)
        return gt[:, 0] * 1e-9, gt[:, 1:4]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_loader(dataset: str, root: str, sequence: str,
               date: str = None, drive: str = None):
    """Factory function to get the correct loader by dataset name."""
    dataset = dataset.lower()
    if dataset == "euroc":
        return EuRoCLoader(root, sequence)
    elif dataset == "kitti":
        if date is None or drive is None:
            raise ValueError("KITTI loader requires date and drive arguments")
        return KITTILoader(root, date, drive)
    elif dataset in ("tumvi", "tum_vi", "tum-vi"):
        return TUMVILoader(root, sequence)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
