# Model history

Every model that reaches the ship gate is logged here, including the ones that fail.

| Version | What it predicts | Val MAE (baseline) | Test MAE (baseline) | Gate |
|---|---|---|---|---|
| v1 | silica level at t+1 | 0.650 (0.646) | 0.722 (0.722) | DON'T SHIP (-0.0%) |
| v2 | change from last lab value | 0.584 (0.646) | 0.674 (0.722) | SHIP (+6.7%) |

## v1: predict the silica level
- HistGradientBoosting, `absolute_error` loss, 67 features (sensors lags 0-2, lab lags 2-4, hour of day).
- Tied the persistence baseline on test, so the gate blocked it.
- Diagnosis (validation only, test untouched):
  - Cheat check: adding silica at lag 0 (not available in real life) drops val MAE to 0.449, so the
    pipeline works, and the 2 h lab rule really costs accuracy.
  - The mean silica level drifts (train 2.32, val 2.28, test 2.43). Tree models can't predict outside
    the range they trained on, so modelling the level is fragile.
  - Predicting the *change* from the last lab value gave val MAE 0.584 (-9.6% vs baseline).
- Decision: v2 predicts the change. Chosen on validation; test was scored once for v1 and once for v2.

## v2: predict the change from the last lab value
- Same features, same tuning grid, same hyper-parameters picked on validation.
- Forecast = last known lab silica (t-2) + predicted change to t+1.
- Test MAE 0.674 vs baseline 0.722: 6.7% better, passes the 5% gate. SHIP.
- Note: the gain shrank from 9.6% (val) to 6.7% (test), and it clears the gate by only 1.7 points.
  In production I'd watch it closely and re-run the gate on each new month of data.
