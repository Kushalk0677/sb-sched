"""
src/datasets/loaders.py

Data loaders for EuRoC MAV, KITTI Raw, and TUM-VI datasets.
Each loader yields a standardised stream of SensorReading objects so
the experiment runner is dataset-agnostic.

Measurement model framing
─────────────────────────
EuRoC / TUM-VI
    IMU (200 Hz) drives the Kalman predict step.
    "Camera" events yield a 3-D position pseudo-observation obtained by
    interpolating the motion-capture ground truth at the camera timestamp.
    This is an oracle measurement (no feature tracking noise), suitable for
    evaluating the *scheduler's* covariance management in isolation.
    RMSE values reflect the simplified model and are NOT comparable to
    full VIO results from the literature.

KITTI
    OXTS packets contain physically real GPS lat/lon/alt measurements at
    ~10 Hz interleaved with IMU inertial readings at the same packet rate.
    GPS coordinates are converted to a local East-North-Up (ENU) Cartesian
    frame anchored at the first fix via a UTM projection, giving metric
    positions with physically meaningful noise (~0.1–0.5 m consumer RTK).
    IMU packets drive the predict step.  RMSE is evaluated in metres (ENU).
"""

import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator


@dataclass
class SensorReading:
    timestamp: float            # Seconds since epoch (or sequence start)
    sensor_name: str            # "imu" | "camera" | "gps"
    z: np.ndarray               # Measurement vector delivered to the KF
    ground_truth: np.ndarray | None = None  # [x, y, z] metres for RMSE eval


# ---------------------------------------------------------------------------
# UTM → local ENU helper (used by KITTILoader only)
# ---------------------------------------------------------------------------

def _make_enu_converter(lat0: float, lon0: float, alt0: float):
    """
    Return a function  (lat, lon, alt) -> np.array([E, N, U])
    that converts WGS-84 geodetic coordinates to a local East-North-Up
    frame anchored at (lat0, lon0, alt0).

    UTM zone is inferred from the anchor longitude.  This works correctly
    for the KITTI recording locations (Karlsruhe, Germany; UTM zone 32).
    For datasets spanning a zone boundary you would need a proper geodetic
    library such as pymap3d, but for short trajectories UTM is sufficient.

    Requires: pyproj  (pip install pyproj)
    """
    try:
        from pyproj import Proj
    except ImportError:
        raise ImportError(
            "pyproj is required for KITTI GPS conversion.\n"
            "Install with:  pip install pyproj"
        )

    zone = int((lon0 + 180) / 6) + 1
    proj = Proj(proj="utm", zone=zone, ellps="WGS84")

    x0, y0 = proj(lon0, lat0)

    def convert(lat: float, lon: float, alt: float) -> np.ndarray:
        x, y = proj(lon, lat)
        return np.array([x - x0, y - y0, alt - alt0])

    return convert


# ---------------------------------------------------------------------------
# EuRoC MAV Loader
# ---------------------------------------------------------------------------

class EuRoCLoader:
    """
    Loads EuRoC MAV sequences.

    Yields:
        IMU readings at 200 Hz  (sensor_name="imu", z=6-vec, ground_truth=None)
        Camera events at 20 Hz  (sensor_name="camera", z=3-vec position,
                                  ground_truth=same 3-vec)

    Camera measurements are ground-truth-interpolated position pseudo-
    observations.  See module docstring for the modeling scope note.

    Directory layout:
        {root}/{sequence}/mav0/
            imu0/data.csv
            cam0/data.csv
            state_groundtruth_estimate0/data.csv
    """

    def __init__(self, root: str, sequence: str):
        self.root = Path(root) / sequence / "mav0"
        self.gt_file = (
            Path(root) / sequence / "mav0"
            / "state_groundtruth_estimate0" / "data.csv"
        )

    def _load_csv(self, path: Path) -> np.ndarray:
        return np.loadtxt(path, delimiter=",", skiprows=1)

    def _load_cam_timestamps(self, path: Path) -> np.ndarray:
        """First column only — avoids parsing the image-filename string."""
        return np.loadtxt(path, delimiter=",", skiprows=1, usecols=0)

    def stream(self) -> Iterator[SensorReading]:
        imu_data = self._load_csv(self.root / "imu0" / "data.csv")
        cam_ts   = self._load_cam_timestamps(self.root / "cam0" / "data.csv")
        gt_data  = self._load_csv(self.gt_file)

        gt_t   = gt_data[:, 0] * 1e-9      # ns → s
        gt_pos = gt_data[:, 1:4]            # px, py, pz (metres)

        readings: list[SensorReading] = []

        for row in imu_data:
            t = row[0] * 1e-9
            readings.append(SensorReading(t, "imu", row[1:7]))

        for ts in cam_ts:
            t = float(ts) * 1e-9
            pos = np.array([
                np.interp(t, gt_t, gt_pos[:, 0]),
                np.interp(t, gt_t, gt_pos[:, 1]),
                np.interp(t, gt_t, gt_pos[:, 2]),
            ])
            # z == ground_truth: oracle pseudo-measurement (declared in docstring)
            readings.append(SensorReading(t, "camera", pos, pos.copy()))

        readings.sort(key=lambda r: r.timestamp)
        yield from readings

    def ground_truth_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """(timestamps_s, positions_m) from the GT file."""
        gt = self._load_csv(self.gt_file)
        return gt[:, 0] * 1e-9, gt[:, 1:4]


# ---------------------------------------------------------------------------
# KITTI Raw Loader
# ---------------------------------------------------------------------------

class KITTILoader:
    """
    Loads KITTI raw sequences with physically real GPS measurements.

    GPS lat/lon/alt from OXTS packets are converted to a local ENU
    (East-North-Up) Cartesian frame anchored at the first packet, giving
    metric positions in metres.  The KF measurement vector z is [E, N, U].

    IMU angular-rate and acceleration data from the same OXTS packets are
    yielded as "imu" readings at the same timestamp, driving the predict step.

    Requires:  pykitti  (pip install pykitti)
               pyproj   (pip install pyproj)

    Directory layout expected by pykitti:
        {root}/{date}/{date}_drive_{drive}_sync/oxts/data/*.txt
    """

    def __init__(self, root: str, date: str, drive: str):
        try:
            import pykitti
        except ImportError:
            raise ImportError("pykitti not installed: pip install pykitti")
        self.data   = pykitti.raw(root, date, drive)
        self._enu   = None      # ENU converter, set on first packet

    def _get_enu(self, lat0: float, lon0: float, alt0: float):
        """Lazily initialise the ENU converter from the first fix."""
        if self._enu is None:
            self._enu = _make_enu_converter(lat0, lon0, alt0)
        return self._enu

    def stream(self) -> Iterator[SensorReading]:
        """
        Yields interleaved IMU and GPS readings in chronological order.

        GPS packets carry ground_truth=z (the ENU position is both the
        measurement and the evaluation reference, consistent with using
        the OXTS RTK trajectory as ground truth).

        Note on rate: KITTI OXTS logs both IMU and GPS at the OXTS packet
        rate (~10 Hz).  The IMU is NOT interpolated to 100 Hz here because
        pykitti does not expose an independent high-rate IMU stream.  The
        KF therefore runs its predict step at 10 Hz for KITTI, not 100 Hz.
        Set imu_hz=10 when building the KITTI model for this loader.
        """
        readings: list[SensorReading] = []
        enu = None

        for i, oxts in enumerate(self.data.oxts):
            t      = self.data.timestamps[i].timestamp()
            packet = oxts.packet

            # Initialise ENU frame from first fix
            if enu is None:
                enu = self._get_enu(packet.lat, packet.lon, packet.alt)

            pos_enu = enu(packet.lat, packet.lon, packet.alt)

            # GPS measurement: physically real ENU position in metres
            readings.append(
                SensorReading(t, "gps", pos_enu.copy(), pos_enu.copy())
            )

            # IMU: angular velocity + linear acceleration
            imu_z = np.array([
                packet.wx, packet.wy, packet.wz,
                packet.ax, packet.ay, packet.az,
            ])
            readings.append(SensorReading(t, "imu", imu_z))

        readings.sort(key=lambda r: r.timestamp)
        yield from readings

    def ground_truth_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """
        (timestamps_s, positions_m) in local ENU frame.
        Suitable for RMSE evaluation against KF estimates.
        """
        enu = None
        timestamps = []
        positions  = []

        for i, oxts in enumerate(self.data.oxts):
            t      = self.data.timestamps[i].timestamp()
            packet = oxts.packet
            if enu is None:
                enu = self._get_enu(packet.lat, packet.lon, packet.alt)
            timestamps.append(t)
            positions.append(enu(packet.lat, packet.lon, packet.alt))

        return np.array(timestamps), np.array(positions)


# ---------------------------------------------------------------------------
# TUM-VI Loader
# ---------------------------------------------------------------------------

class TUMVILoader:
    """
    Loads TUM-VI sequences.  Identical measurement model to EuRoC:
    IMU drives predict, camera events yield GT-interpolated position
    pseudo-observations.

    Directory layout:
        {root}/{sequence}/mav0/
            imu0/data.csv
            cam0/data.csv
        {root}/{sequence}/dso/gt_imu.csv   (mocap ground truth)
    """

    def __init__(self, root: str, sequence: str):
        self.root    = Path(root) / sequence / "mav0"
        self.gt_file = Path(root) / sequence / "dso" / "gt_imu.csv"

    def _load_csv(self, path: Path) -> np.ndarray:
        return np.loadtxt(path, delimiter=",", skiprows=1)

    def _load_cam_timestamps(self, path: Path) -> np.ndarray:
        return np.loadtxt(path, delimiter=",", skiprows=1, usecols=0)

    def stream(self) -> Iterator[SensorReading]:
        imu_data = self._load_csv(self.root / "imu0" / "data.csv")
        cam_ts   = self._load_cam_timestamps(self.root / "cam0" / "data.csv")
        gt_data  = self._load_csv(self.gt_file)

        gt_t   = gt_data[:, 0] * 1e-9
        gt_pos = gt_data[:, 1:4]

        readings: list[SensorReading] = []

        for row in imu_data:
            t = row[0] * 1e-9
            readings.append(SensorReading(t, "imu", row[1:7]))

        for ts in cam_ts:
            t = float(ts) * 1e-9
            pos = np.array([
                np.interp(t, gt_t, gt_pos[:, 0]),
                np.interp(t, gt_t, gt_pos[:, 1]),
                np.interp(t, gt_t, gt_pos[:, 2]),
            ])
            readings.append(SensorReading(t, "camera", pos, pos.copy()))

        readings.sort(key=lambda r: r.timestamp)
        yield from readings

    def ground_truth_positions(self) -> tuple[np.ndarray, np.ndarray]:
        gt = self._load_csv(self.gt_file)
        return gt[:, 0] * 1e-9, gt[:, 1:4]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_loader(
    dataset: str,
    root: str,
    sequence: str,
    date: str | None = None,
    drive: str | None = None,
) -> EuRoCLoader | KITTILoader | TUMVILoader:
    """Return the correct loader for the named dataset."""
    dataset = dataset.lower()
    if dataset == "euroc":
        return EuRoCLoader(root, sequence)
    elif dataset == "kitti":
        if date is None or drive is None:
            raise ValueError(
                "KITTI loader requires date and drive. "
                "Use parse_kitti_sequence() to extract them from the sequence string."
            )
        return KITTILoader(root, date, drive)
    elif dataset in ("tumvi", "tum_vi", "tum-vi"):
        return TUMVILoader(root, sequence)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")
