# Pulse forecast

This project trains a small GRU on public pulse-oximeter waveforms.
It uses two seconds of history to predict the next 0.4 seconds.

The data come from the [BIDMC dataset](https://physionet.org/content/bidmc/1.0.0/)
(Pimentel et al., 2016), under the Open Data Commons Attribution License v1.0.
Recordings 01–05 train the model, 06 selects the checkpoint, and 07 tests it.

## Model and results

The model has one GRU layer with 32 hidden units and 3,690 parameters.
Signals are averaged from 125 Hz to 25 Hz before training.
Training-only normalization scales the validation and test recordings.
A linear head predicts ten future samples from 50 history samples.
Adam trains the model for 20 epochs with seed 42.
Validation selected the checkpoint from epoch 19.
Training used 5,975 windows; evaluation used 1,195 test windows.
GRU test MAE was 0.0368 and RMSE was 0.0497 in source PPG units.
Ridge test RMSE was 0.0606; persistence test RMSE was 0.2367.
The GRU reduced RMSE by 18.0% versus ridge and 79.0% versus persistence.
Training, evaluation, and plotting took 27.4 seconds on two CPU threads.
Results cover one held-out recording and do not establish clinical validity.

## How to run

Python 3.11 or newer is required. Training runs on CPU.

```powershell
git clone https://github.com/piercetaylor/pulse-forecast.git
cd pulse-forecast
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
pulse-forecast train --output-dir runs/example
```

On Linux or macOS, activate with `source .venv/bin/activate`.
The training command downloads seven recordings and saves the model, metrics,
and plots in `runs/example`. Use a new output directory for another run.
Tests run with `pytest -q`.

## Results and license

[Demo results](examples/demo/summary.md) and [forecast plots](examples/demo/forecasts.png)
are included. The code uses the [MIT license](LICENSE).
Dataset reuse should cite [Pimentel et al.](https://doi.org/10.1109/TBME.2016.2613124).
