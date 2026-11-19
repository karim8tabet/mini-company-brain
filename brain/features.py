"""Step 2: turn the hourly table into (features at hour t) -> (silica at hour t+1), with leakage rules."""
import pandas as pd

from brain.clean import LAB_COLS, TARGET

HORIZON = 1          # forecast 1 hour ahead
LAB_DELAY = 2        # lab results arrive ~2 h after sampling
SENSOR_LAGS = [0, 1, 2]
LAB_LAGS = [LAB_DELAY, LAB_DELAY + 1, LAB_DELAY + 2]
NEVER_USE = ["pct_iron_concentrate"]  # same lab test as the target -> leaks the answer
LAB_FEATURES = [c for c in LAB_COLS if c not in NEVER_USE]


def sensor_cols(hourly: pd.DataFrame) -> list[str]:
    return [c for c in hourly.columns if c not in LAB_COLS + ["n_rows"]]


def build_dataset(hourly: pd.DataFrame) -> pd.DataFrame:
    """Rows indexed by forecast origin t. Columns: features, 'baseline', 'y' (silica at t+HORIZON).

    hourly must be on a complete hourly index (see clean.to_hourly), so shift(k) means k hours.
    """
    feats = {}
    for c in sensor_cols(hourly):
        for k in SENSOR_LAGS:
            feats[f"{c}_lag{k}"] = hourly[c].shift(k)
    for c in LAB_FEATURES:
        for k in LAB_LAGS:
            feats[f"{c}_lag{k}"] = hourly[c].shift(k)
    feats["hour_of_day"] = hourly.index.hour  # shift patterns
    data = pd.DataFrame(feats, index=hourly.index)
    # Persistence baseline: the newest lab silica the plant actually has at time t.
    data["baseline"] = hourly[TARGET].shift(LAB_DELAY)
    data["y"] = hourly[TARGET].shift(-HORIZON)
    # Score model and baseline on the same rows: need a label and a baseline forecast.
    return data.dropna(subset=["y", "baseline"])


def feature_cols(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c not in ("y", "baseline")]


def time_split(data: pd.DataFrame, train: float = 0.70, val: float = 0.15):
    """Chronological split: oldest 70% train, next 15% validation, newest 15% test. No shuffling."""
    data = data.sort_index()
    n = len(data)
    a, b = int(n * train), int(n * (train + val))
    return data.iloc[:a], data.iloc[a:b], data.iloc[b:]
