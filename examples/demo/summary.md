# Experiment results

The test metrics describe forecasting on held-out BIDMC recording 07. Errors are measured in the source PPG amplitude units.

| Model | MAE | RMSE |
| --- | ---: | ---: |
| PERSISTENCE | 0.199514 | 0.236657 |
| RIDGE | 0.045748 | 0.060578 |
| GRU | 0.036788 | 0.049671 |

Validation selected epoch 19 of 20. The model contains 3,690 parameters.

A single held-out recording cannot establish patient-level generalization. The experiment has no diagnostic or clinical endpoint.
