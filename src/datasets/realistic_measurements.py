"""Realistic measurement wrappers for scheduler validation.

This module upgrades the repository's controlled-oracle testbed with
measurement imperfections that are common in deployed perception and GPS
pipelines:

- additive measurement noise
- random dropout
- fixed or random latency
- occasional outliers
- slowly varying bias / drift
- optional use of precomputed external VO trajectories for EuRoC / TUM-VI

The wrappers are intentionally lightweight. They do *not* attempt to build a
full SLAM/VIO stack inside this repository. Instead they let the scheduling
policy consume messier, deployment-like measurements while keeping the KF and
scheduler infrastructure unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple
import csv

import numpy as np

from .loaders import SensorReading, EuRoCLoader, TUMVILoader, KITTILoader


@dataclass(slots=True)
class RealismProfile:
    """Measurement corruption settings for one low-rate sensor stream.

    Parameters are expressed in measurement-event units where relevant.
    For EuRoC/TUM-VI the low-rate sensor is the camera stream (default 20 Hz).
    For KITTI the low-rate sensor is the GPS stream (default 10 Hz).
    """

    name: str
    measurement_noise_std_m: float = 0.0
    dropout_prob: float = 0.0
    fixed_delay_events: int = 0
    jitter_delay_events: int = 0
    outlier_prob: float = 0.0
    outlier_std_m: float = 0.0
    bias_walk_std_m: float = 0.0
    constant_bias_m: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    seed: int = 42
    description: str = ""


@dataclass(slots=True)
class PrecomputedTrajectory:
    timestamps_s: np.ndarray
    positions_m: np.ndarray
    source_path: str


@dataclass(slots=True)
class ProfileRunMetadata:
    profile: RealismProfile
    measurement_sensor: str
    native_hz: float
    external_source: str | None = None
    original_measurement_count: int = 0
    delivered_measurement_count: int = 0
    dropped_measurement_count: int = 0


class RealisticMeasurementStream:
    """Wrap an existing dataset loader and corrupt the low-rate measurements.

    IMU events are passed through unchanged.
    Low-rate measurement events are optionally replaced by a precomputed
    external trajectory, then corrupted according to ``RealismProfile``.
    """

    def __init__(
        self,
        base_loader,
        measurement_sensor: str,
        native_hz: float,
        profile: RealismProfile,
        external_trajectory: PrecomputedTrajectory | None = None,
    ):
        self.base_loader = base_loader
        self.measurement_sensor = measurement_sensor
        self.native_hz = float(native_hz)
        self.profile = profile
        self.external_trajectory = external_trajectory
        self.metadata = ProfileRunMetadata(
            profile=profile,
            measurement_sensor=measurement_sensor,
            native_hz=self.native_hz,
            external_source=None if external_trajectory is None else external_trajectory.source_path,
        )

    def stream(self) -> Iterator[SensorReading]:
        rng = np.random.default_rng(self.profile.seed)
        delay_dt = 1.0 / self.native_hz
        bias: np.ndarray | None = None
        delayed_events: list[SensorReading] = []

        for reading in self.base_loader.stream():
            if reading.sensor_name != self.measurement_sensor:
                delayed_events.append(reading)
                continue

            self.metadata.original_measurement_count += 1

            if self.profile.dropout_prob > 0.0 and rng.random() < self.profile.dropout_prob:
                self.metadata.dropped_measurement_count += 1
                continue

            gt = None if reading.ground_truth is None else np.asarray(reading.ground_truth, dtype=float).copy()
            z = self._measurement_value(reading.timestamp, reading.z, gt)

            if bias is None:
                base_bias = np.asarray(self.profile.constant_bias_m, dtype=float).reshape(-1)
                if base_bias.size == 0:
                    bias = np.zeros_like(z)
                elif base_bias.size == z.size:
                    bias = base_bias.copy()
                elif base_bias.size == 1:
                    bias = np.full_like(z, float(base_bias[0]))
                else:
                    bias = np.zeros_like(z)
                    bias[: min(z.size, base_bias.size)] = base_bias[: min(z.size, base_bias.size)]

            if self.profile.bias_walk_std_m > 0.0:
                bias = bias + rng.normal(0.0, self.profile.bias_walk_std_m, size=z.shape)
                z = z + bias
            elif np.any(bias != 0.0):
                z = z + bias

            if self.profile.measurement_noise_std_m > 0.0:
                z = z + rng.normal(0.0, self.profile.measurement_noise_std_m, size=z.shape)

            if self.profile.outlier_prob > 0.0 and rng.random() < self.profile.outlier_prob:
                z = z + rng.normal(0.0, self.profile.outlier_std_m, size=z.shape)

            delay_events = self.profile.fixed_delay_events
            if self.profile.jitter_delay_events > 0:
                delay_events += int(rng.integers(0, self.profile.jitter_delay_events + 1))
            delayed_ts = float(reading.timestamp) + delay_events * delay_dt

            delayed_events.append(
                SensorReading(
                    timestamp=delayed_ts,
                    sensor_name=reading.sensor_name,
                    z=z,
                    ground_truth=gt,
                )
            )
            self.metadata.delivered_measurement_count += 1

        delayed_events.sort(key=lambda r: r.timestamp)
        yield from delayed_events

    def _measurement_value(self, timestamp_s: float, default_z: np.ndarray, gt: np.ndarray | None) -> np.ndarray:
        if self.external_trajectory is not None:
            return interpolate_positions(self.external_trajectory.timestamps_s, self.external_trajectory.positions_m, timestamp_s)
        return np.asarray(default_z, dtype=float).copy()


def interpolate_positions(timestamps_s: np.ndarray, positions_m: np.ndarray, query_t_s: float) -> np.ndarray:
    """Linear interpolation helper for [x, y, z] trajectories."""
    x = np.interp(query_t_s, timestamps_s, positions_m[:, 0])
    y = np.interp(query_t_s, timestamps_s, positions_m[:, 1])
    z = np.interp(query_t_s, timestamps_s, positions_m[:, 2])
    return np.array([x, y, z], dtype=float)


# ---------------------------------------------------------------------------
# Precomputed trajectory readers
# ---------------------------------------------------------------------------

def load_precomputed_trajectory(path: str | Path) -> PrecomputedTrajectory:
    """Load an external trajectory for EuRoC/TUM-VI.

    Accepted CSV header examples:
    - timestamp,x,y,z
    - timestamp_ns,px,py,pz
    - t,x,y,z

    Timestamps may be expressed in seconds or nanoseconds. Nanoseconds are
    detected heuristically by magnitude (>1e12).
    """
    path = Path(path)
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        field_map = {name.lower().strip(): name for name in reader.fieldnames or []}
        time_key = _first_present(field_map, ["timestamp", "timestamp_s", "timestamp_ns", "time", "t"])
        x_key = _first_present(field_map, ["x", "px", "tx"])
        y_key = _first_present(field_map, ["y", "py", "ty"])
        z_key = _first_present(field_map, ["z", "pz", "tz"])
        if not all([time_key, x_key, y_key, z_key]):
            raise ValueError(
                f"Unsupported trajectory CSV columns in {path}. Expected timestamp/x/y/z-style columns."
            )

        rows = list(reader)
        if not rows:
            raise ValueError(f"Trajectory file {path} is empty.")

        timestamps = np.array([float(r[time_key]) for r in rows], dtype=float)
        if float(np.nanmedian(np.abs(timestamps))) > 1.0e12:
            timestamps *= 1.0e-9
        positions = np.array(
            [[float(r[x_key]), float(r[y_key]), float(r[z_key])] for r in rows],
            dtype=float,
        )
    return PrecomputedTrajectory(timestamps_s=timestamps, positions_m=positions, source_path=str(path))


def _first_present(field_map: Dict[str, str], candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in field_map:
            return field_map[c]
    return None


# ---------------------------------------------------------------------------
# Dataset-specific wrapper factories
# ---------------------------------------------------------------------------

def make_euroc_realistic_loader(
    root: str,
    sequence: str,
    profile: RealismProfile,
    precomputed_trajectory_csv: str | None = None,
):
    base_loader = EuRoCLoader(root, sequence)
    external = None if precomputed_trajectory_csv is None else load_precomputed_trajectory(precomputed_trajectory_csv)
    return RealisticMeasurementStream(
        base_loader=base_loader,
        measurement_sensor="camera",
        native_hz=20.0,
        profile=profile,
        external_trajectory=external,
    )


def make_tumvi_realistic_loader(
    root: str,
    sequence: str,
    profile: RealismProfile,
    precomputed_trajectory_csv: str | None = None,
):
    base_loader = TUMVILoader(root, sequence)
    external = None if precomputed_trajectory_csv is None else load_precomputed_trajectory(precomputed_trajectory_csv)
    return RealisticMeasurementStream(
        base_loader=base_loader,
        measurement_sensor="camera",
        native_hz=20.0,
        profile=profile,
        external_trajectory=external,
    )


def make_kitti_realistic_loader(
    root: str,
    date: str,
    drive: str,
    profile: RealismProfile,
):
    base_loader = KITTILoader(root, date, drive)
    return RealisticMeasurementStream(
        base_loader=base_loader,
        measurement_sensor="gps",
        native_hz=10.0,
        profile=profile,
        external_trajectory=None,
    )


# ---------------------------------------------------------------------------
# Default profiles used by the new realism benchmark script
# ---------------------------------------------------------------------------

def default_euroc_profiles(seed: int = 42) -> dict[str, RealismProfile]:
    return {
        "oracle": RealismProfile(name="oracle", seed=seed, description="Current repository behaviour: no corruption."),
        "noisy": RealismProfile(
            name="noisy",
            measurement_noise_std_m=0.20,
            outlier_prob=0.01,
            outlier_std_m=1.00,
            seed=seed,
            description="Camera-position proxy with moderate Gaussian noise and rare outliers.",
        ),
        "noisy_dropout": RealismProfile(
            name="noisy_dropout",
            measurement_noise_std_m=0.20,
            dropout_prob=0.20,
            outlier_prob=0.01,
            outlier_std_m=1.00,
            seed=seed,
            description="Noisy camera proxy with 20% frame loss.",
        ),
        "noisy_delay": RealismProfile(
            name="noisy_delay",
            measurement_noise_std_m=0.20,
            fixed_delay_events=1,
            jitter_delay_events=2,
            outlier_prob=0.01,
            outlier_std_m=1.00,
            seed=seed,
            description="Noisy camera proxy with 1-3 frame delivery latency.",
        ),
    }


def default_kitti_profiles(seed: int = 42) -> dict[str, RealismProfile]:
    return {
        "gps_raw": RealismProfile(name="gps_raw", seed=seed, description="Current repository behaviour: raw ENU GPS."),
        "gps_noisy": RealismProfile(
            name="gps_noisy",
            measurement_noise_std_m=1.5,
            bias_walk_std_m=0.03,
            seed=seed,
            description="GPS with added 1.5 m noise and slow drift.",
        ),
        "gps_dropout": RealismProfile(
            name="gps_dropout",
            measurement_noise_std_m=1.5,
            bias_walk_std_m=0.03,
            dropout_prob=0.30,
            seed=seed,
            description="GPS with 30% dropout, representing tunnels / urban canyon outages.",
        ),
        "gps_delay": RealismProfile(
            name="gps_delay",
            measurement_noise_std_m=1.5,
            bias_walk_std_m=0.03,
            fixed_delay_events=1,
            jitter_delay_events=2,
            seed=seed,
            description="GPS with 100-300 ms latency at 10 Hz.",
        ),
    }


# ---------------------------------------------------------------------------
# Gas-dataset realism profiles
# ---------------------------------------------------------------------------

def default_gas_profiles(seed: int = 42) -> dict[str, RealismProfile]:
    return {
        'gas_raw': RealismProfile(
            name='gas_raw',
            measurement_noise_std_m=0.0,
            description='Raw dataset stream without extra corruption; intrinsic drift/noise remain.',
            seed=seed,
        ),
        'gas_dropout': RealismProfile(
            name='gas_dropout',
            measurement_noise_std_m=0.0,
            dropout_prob=0.25,
            description='Raw gas stream with 25% dropped samples.',
            seed=seed + 1,
        ),
        'gas_delay': RealismProfile(
            name='gas_delay',
            measurement_noise_std_m=0.0,
            fixed_delay_events=2,
            description='Raw gas stream with a fixed 2-sample delay.',
            seed=seed + 2,
        ),
        'gas_dropout_delay': RealismProfile(
            name='gas_dropout_delay',
            measurement_noise_std_m=0.0,
            dropout_prob=0.25,
            fixed_delay_events=2,
            description='Raw gas stream with 25% dropout and 2-sample delay.',
            seed=seed + 3,
        ),
    }
