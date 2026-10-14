"""Tests for time ordering, preprocessing, download integrity, and training."""

import json
import hashlib
import io

import numpy as np
import pytest
import torch

from pulse_forecast.data import SPLITS, prepare, read_ppg, verify, windows
from pulse_forecast import data
from pulse_forecast.model import GRUForecaster, predict_ridge
from pulse_forecast.train import experiment, predict


def test_windows_predict_only_future_samples():
    signal = np.arange(12, dtype=np.float32)
    x, y = windows(signal, context=4, horizon=3, stride=2)
    np.testing.assert_array_equal(x[0, :, 0], [0, 1, 2, 3])
    np.testing.assert_array_equal(y[0], [4, 5, 6])
    np.testing.assert_array_equal(y[-1], [8, 9, 10])
    with pytest.raises(ValueError, match="too short"):
        windows(signal[:3], 4, 3, 1)


def test_held_out_values_cannot_change_scaler_or_training_windows():
    records = {r: np.arange(20, dtype=np.float32) + r * 100 for r in range(1, 8)}
    before, mean, std = prepare(records, 4, 2, 2)
    records[6] += 10000
    records[7] *= -100
    after, new_mean, new_std = prepare(records, 4, 2, 2)
    assert (mean, std) == (new_mean, new_std)
    np.testing.assert_array_equal(before["train"][0], after["train"][0])
    # A consecutive training window stays entirely within one recording.
    raw_windows = before["train"][0][..., 0] * std + mean
    np.testing.assert_allclose(np.diff(raw_windows, axis=1), 1, atol=1e-4)
    ids = [r for split in SPLITS.values() for r in split]
    assert len(ids) == len(set(ids))


def test_block_average_uses_only_its_own_samples(tmp_path):
    path = tmp_path / "signal.csv"
    path.write_text("Time [s], PLETH\n" + "\n".join(f"{i / 125:.3f},{i}" for i in range(10)))
    np.testing.assert_array_equal(read_ppg(path), [2, 7])
    path.write_text(
        "Time [s], PLETH\n"
        + "\n".join(f"{i / 125:.3f},{'nan' if i == 3 else i}" for i in range(10))
    )
    with pytest.raises(ValueError, match="nonfinite"):
        read_ppg(path)


def test_time_grid_allows_published_rounding_and_rejects_gaps(tmp_path):
    path = tmp_path / "rounded.csv"
    path.write_text("Time [s], PLETH\n" + "\n".join(f"{100 + i / 125:.2f},{i}" for i in range(20)))
    np.testing.assert_array_equal(read_ppg(path), [2, 7, 12, 17])
    path.write_text(
        "Time [s], PLETH\n"
        + "\n".join(f"{100 + i / 125 + (0.08 if i >= 10 else 0):.2f},{i}" for i in range(20))
    )
    with pytest.raises(ValueError, match="sample grid"):
        read_ppg(path)


def test_corrupt_download_is_rejected(tmp_path):
    path = tmp_path / "download.csv"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        verify(path, "0" * 64)


def test_downloader_verifies_network_and_cached_files(tmp_path, monkeypatch):
    payloads = {record: f"record-{record}".encode() for record in range(1, 8)}
    hashes = {record: hashlib.sha256(content).hexdigest() for record, content in payloads.items()}
    calls = []

    def response(request, timeout):
        record = int(request.full_url.rsplit("/", 1)[-1].split("_")[1])
        calls.append(record)
        return io.BytesIO(payloads[record])

    monkeypatch.setattr(data, "SHA256", hashes)
    monkeypatch.setattr(data.urllib.request, "urlopen", response)
    paths = data.download(tmp_path)
    assert sorted(calls) == list(range(1, 8))
    assert not list(tmp_path.glob("*.tmp"))
    data.download(tmp_path)
    assert len(calls) == 7
    paths[1].write_bytes(b"corrupt cached data")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        data.download(tmp_path)


def test_training_and_checkpoint_roundtrip(tmp_path):
    t = np.arange(150, dtype=np.float32) / 25
    records = {r: np.sin(2 * np.pi * t + r / 10).astype(np.float32) for r in range(1, 8)}
    out = tmp_path / "run"
    report = experiment(records, out, epochs=2, threads=1)
    assert np.isfinite(report["test_metrics"]["gru"]["rmse"])
    assert 1 <= report["config"]["best_epoch"] <= 2
    assert json.loads((out / "metrics.json").read_text())["data"]["source"] == "test fixture"
    checkpoint = torch.load(out / "model.pt", weights_only=True)
    config = checkpoint["config"]
    restored = GRUForecaster(config["hidden_size"], config["horizon"])
    restored.load_state_dict(checkpoint["state_dict"])
    datasets, mean, std = prepare(records, config["context"], config["horizon"], config["stride"])
    x, _ = datasets["test"]
    with np.load(out / "test_predictions.npz") as saved:
        np.testing.assert_allclose(predict(restored, x) * std + mean, saved["gru"], atol=1e-6)
        coefficients = np.load(out / "ridge.npy")
        np.testing.assert_allclose(predict_ridge(x, coefficients) * std + mean, saved["ridge"])
    assert (out / "forecasts.png").is_file()
    with pytest.raises(ValueError, match="not empty"):
        experiment(records, out, epochs=1)
