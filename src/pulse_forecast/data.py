"""Verified BIDMC downloads and preprocessing without future observations."""

import csv
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SOURCE = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv"
SOURCE_HZ = 125
BLOCK_SIZE = 5
SAMPLE_HZ = SOURCE_HZ // BLOCK_SIZE
SPLITS = {"train": (1, 2, 3, 4, 5), "validation": (6,), "test": (7,)}
# PhysioNet's version 1.0.0 SHA256SUMS.txt pins the raw CSV bytes.
SHA256 = {
    1: "21495169cc36e82352d8e12cc6430927946d02d240b030715088d02396ea2d5c",
    2: "cbe4cc6532df50aaa56382cc6348effc76ed2347f6562a590876a0dfb747e2ea",
    3: "c8a642b9d5c0054d401489a96b050f8977890fa45c7ffe1f778bd32ce7086c53",
    4: "46363e8552709b13b0ff35fc81914f441ac84e1ee656a5f5dbbf66cf7bcfea59",
    5: "1963671232fca4962400f660bcb437db3d53a414de2f9538ca423a67212112cf",
    6: "fd82ddab95eb5c0d83b5e68f787809f29f8c3d2e3c02f6798d8630bc093e1a51",
    7: "eb58108883ba819b26eb80e1984e395b31b4cb7fb4c3341c84b733bff97d090f",
}


def verify(path: Path, expected: str) -> None:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path}. Delete it and download again.")


def download(data_dir: Path) -> dict[int, Path]:
    data_dir.mkdir(parents=True, exist_ok=True)

    def fetch(record: int) -> tuple[int, Path]:
        checksum = SHA256[record]
        path = data_dir / f"bidmc_{record:02d}_Signals.csv"
        if not path.exists():
            temporary = path.with_suffix(".tmp")
            print(f"Downloading recording {record:02d}...", flush=True)
            try:
                request = urllib.request.Request(
                    f"{SOURCE}/{path.name}", headers={"User-Agent": "pulse-forecast/0.1"}
                )
                with urllib.request.urlopen(request, timeout=90) as response:
                    temporary.write_bytes(response.read())
                verify(temporary, checksum)
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        verify(path, checksum)
        print(f"Verified recording {record:02d}.", flush=True)
        return record, path

    with ThreadPoolExecutor(max_workers=3) as pool:
        return dict(pool.map(fetch, SHA256))


def read_ppg(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, skipinitialspace=True)
        names = {name.strip().lower(): name for name in reader.fieldnames or []}
        if "pleth" not in names or "time [s]" not in names:
            raise ValueError(f"{path} must contain Time [s] and PLETH columns.")
        rows = [(float(row[names["time [s]"]]), float(row[names["pleth"]])) for row in reader]
    if len(rows) < BLOCK_SIZE:
        raise ValueError(f"{path} has too few samples.")
    values = np.asarray(rows, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{path} contains missing or nonfinite samples.")
    # BIDMC exports later timestamps with centisecond precision, including ties.
    # Check the full indexed time grid, allowing half of the displayed time unit.
    timestamps = values[:, 0]
    expected = timestamps[0] + np.arange(len(timestamps)) / SOURCE_HZ
    if (np.diff(timestamps) < 0).any() or not np.allclose(
        timestamps, expected, atol=0.005001, rtol=0
    ):
        raise ValueError(f"{path} is inconsistent with the {SOURCE_HZ} Hz sample grid.")
    signal = values[:, 1]
    length = len(signal) // BLOCK_SIZE * BLOCK_SIZE
    # Each output is available at its block's final raw sample. Blocks never overlap.
    return signal[:length].reshape(-1, BLOCK_SIZE).mean(axis=1).astype(np.float32)


def fit_scaler(records: list[np.ndarray]) -> tuple[float, float]:
    pooled = np.concatenate(records).astype(np.float64)
    mean, std = float(pooled.mean()), float(pooled.std())
    if not np.isfinite([mean, std]).all() or std < 1e-8:
        raise ValueError("Training signal must be finite and have nonzero variance.")
    return mean, std


def windows(signal: np.ndarray, context: int, horizon: int, stride: int):
    if min(context, horizon, stride) < 1:
        raise ValueError("Context, horizon and stride must be positive.")
    starts = range(0, len(signal) - context - horizon + 1, stride)
    pairs = [(signal[s : s + context], signal[s + context : s + context + horizon]) for s in starts]
    if not pairs:
        raise ValueError("Recording is too short for the requested window.")
    x, y = zip(*pairs)
    return np.stack(x)[..., None].copy(), np.stack(y).copy()


def prepare(records: dict[int, np.ndarray], context: int, horizon: int, stride: int):
    mean, std = fit_scaler([records[r] for r in SPLITS["train"]])
    datasets = {}
    for split, ids in SPLITS.items():
        # Construct windows inside each recording to prevent boundary crossings.
        parts = [windows((records[r] - mean) / std, context, horizon, stride) for r in ids]
        datasets[split] = tuple(np.concatenate([part[i] for part in parts]) for i in (0, 1))
    return datasets, mean, std
