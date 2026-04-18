from pathlib import Path

from src.datasets.gas_loaders import OneMonthGasLoader, UCIGasDriftLoader


def test_onemonth_loader_reads_rows(tmp_path: Path):
    csv_path = tmp_path / 'gas_sensor_dataset.csv'
    csv_path.write_text(
        'timestamp,arduino_id,sensor_type,sensor_id,adc_value,voltage,resistance,rs_r0_ratio,temperature,humidity\n'
        '2024-04-01 00:00:00,ARD_01,MQ-2,0,500,2.4,10000,1.05,25,60\n'
        '2024-04-01 00:01:00,ARD_01,MQ-2,0,501,2.4,10010,1.06,25,60\n'
    )
    loader = OneMonthGasLoader(str(tmp_path), sensor_id=0, feature='rs_r0_ratio')
    readings = list(loader.stream())
    assert len(readings) == 4
    assert readings[0].sensor_name == 'imu'
    assert readings[1].sensor_name == 'gas'
    assert float(readings[1].z[0]) == 1.05


def test_uci_loader_parses_sparse_feature(tmp_path: Path):
    d = tmp_path / 'Dataset'
    d.mkdir()
    (d / 'batch1.dat').write_text('1 1:10.0 2:20.0\n2 1:11.0 2:21.0\n')
    loader = UCIGasDriftLoader(str(tmp_path), batch='batch1', feature_index=2, dt_s=1.0)
    readings = list(loader.stream())
    assert len(readings) == 4
    assert readings[1].sensor_name == 'gas'
    assert float(readings[1].z[0]) == 20.0
