"""Gas-dataset loaders for realism benchmarking.

These loaders adapt two gas-sensing datasets to the repository's scheduler/KF
stream interface.

Design choice
-------------
The scheduler framework expects a predict-driving stream ("imu") and a low-rate
measurement stream. Gas datasets do not contain IMU. To preserve the existing
benchmark loop without rewriting the scheduler stack, each sample emits:

1. one synthetic predict tick named ``imu``
2. one gas measurement event named ``gas``

This turns each sample interval into one predict-update opportunity, which is
appropriate for resource-constrained sampling studies where the decision is
whether to take the next gas measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
import csv

import numpy as np
import pandas as pd

from .loaders import SensorReading


@dataclass(slots=True)
class GasSequenceMetadata:
    sequence: str
    feature_name: str
    native_hz: float
    n_samples: int
    source_path: str


class OneMonthGasLoader:
    """Loader for the user's one-month real gas dataset.

    Parameters
    ----------
    root:
        Directory containing ``gas_sensor_dataset.csv``.
    sensor_id:
        Which sensor channel to use as the low-rate measurement source.
    feature:
        Column to track, default ``rs_r0_ratio`` because it is a normalized gas
        response measure and more stable across channels than raw ADC.
    """

    def __init__(self, root: str, sensor_id: int = 0, feature: str = 'rs_r0_ratio'):
        self.root = Path(root)
        self.csv_path = self.root / 'gas_sensor_dataset.csv'
        if not self.csv_path.exists():
            raise FileNotFoundError(f'Expected {self.csv_path}')
        self.sensor_id = int(sensor_id)
        self.feature = feature
        self._df = self._load_df()
        dt_s = np.median(np.diff(self._df['timestamp_s'].values)) if len(self._df) > 1 else 60.0
        native_hz = 1.0 / max(float(dt_s), 1e-9)
        self.metadata = GasSequenceMetadata(
            sequence=f'sensor_{self.sensor_id}',
            feature_name=self.feature,
            native_hz=native_hz,
            n_samples=len(self._df),
            source_path=str(self.csv_path),
        )

    def _load_df(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)
        if self.feature not in df.columns:
            raise ValueError(f'Feature {self.feature!r} not present. Available columns: {list(df.columns)}')
        if 'sensor_id' not in df.columns or 'timestamp' not in df.columns:
            raise ValueError('gas_sensor_dataset.csv must contain sensor_id and timestamp columns.')
        df = df[df['sensor_id'] == self.sensor_id].copy()
        if df.empty:
            raise ValueError(f'No rows found for sensor_id={self.sensor_id}')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        t0 = df['timestamp'].iloc[0]
        df['timestamp_s'] = (df['timestamp'] - t0).dt.total_seconds().astype(float)
        df.sort_values('timestamp_s', inplace=True)
        return df.reset_index(drop=True)

    def stream(self) -> Iterator[SensorReading]:
        for row in self._df.itertuples(index=False):
            t = float(row.timestamp_s)
            value = np.array([float(getattr(row, self.feature))], dtype=float)
            yield SensorReading(t, 'imu', np.zeros(1, dtype=float))
            yield SensorReading(t, 'gas', value.copy(), value.copy())


class UCIGasDriftLoader:
    """Loader for the UCI Gas Sensor Array Drift dataset.

    The ``*.dat`` files are in LIBSVM-style sparse format with a class label and
    128 features. We interpret row order as time order within each batch.

    Parameters
    ----------
    root:
        Directory containing the ``Dataset/`` folder from the zip.
    batch:
        Batch file name, e.g. ``batch1`` or ``batch1.dat``.
    feature_index:
        1-based feature index from the sparse file. Default 1 uses the first
        raw sensor response channel.
    dt_s:
        Synthetic spacing between consecutive samples.
    """

    def __init__(self, root: str, batch: str = 'batch1', feature_index: int = 2, dt_s: float = 1.0):
        self.root = Path(root)
        self.batch = batch
        self.feature_index = int(feature_index)
        self.dt_s = float(dt_s)
        self._values, source = self._load_values()
        self.metadata = GasSequenceMetadata(
            sequence=self.batch.replace('.dat', ''),
            feature_name=f'feature_{self.feature_index}',
            native_hz=1.0 / max(self.dt_s, 1e-9),
            n_samples=len(self._values),
            source_path=source,
        )

    def _iter_batch_paths(self):
        dataset_dir = self.root / 'Dataset'
        if self.batch.lower() in {'all', 'all_batches'}:
            paths = sorted(dataset_dir.glob('batch*.dat'))
            if not paths:
                raise FileNotFoundError(f'No batch*.dat files found under {dataset_dir}')
            return paths
        batch_name = self.batch if self.batch.endswith('.dat') else f'{self.batch}.dat'
        path = dataset_dir / batch_name
        if not path.exists():
            raise FileNotFoundError(f'Expected {path}')
        return [path]

    def _load_values(self) -> tuple[np.ndarray, str]:
        vals = []
        target = f'{self.feature_index}:'
        paths = self._iter_batch_paths()
        for path in paths:
            with path.open('r') as f:
                for line in f:
                    toks = line.strip().split()
                    found = None
                    for tok in toks[1:]:
                        if tok.startswith(target):
                            found = float(tok.split(':', 1)[1])
                            break
                    if found is None:
                        raise ValueError(f'Feature index {self.feature_index} missing in {path}')
                    vals.append(found)
        arr = np.asarray(vals, dtype=float)
        if arr.size == 0:
            raise ValueError('No samples parsed from the requested batch selection.')
        return arr, ','.join(str(p) for p in paths)

    def stream(self) -> Iterator[SensorReading]:
        for i, value in enumerate(self._values):
            t = i * self.dt_s
            z = np.array([float(value)], dtype=float)
            yield SensorReading(t, 'imu', np.zeros(1, dtype=float))
            yield SensorReading(t, 'gas', z.copy(), z.copy())
