"""Steps 3-5 + 7: persistence baseline vs gradient-boosting model, then the ship gate."""
import itertools
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from brain.clean import HOURLY_CSV, ROOT
from brain.features import build_dataset, feature_cols, time_split

METRICS = ROOT / "reports" / "metrics.json"
MARGIN = 0.05  # ship only if test MAE is at least 5% better than the baseline
SEED = 0
GRID = {
    "learning_rate": [0.03, 0.1],
    "max_leaf_nodes": [15, 31],
    "max_iter": [200, 500],
}


def make_model(**params) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error", random_state=SEED, early_stopping=False, **params
    )


MODEL_VERSION = "v2"  # v1 predicted the level and failed the gate; see reports/model_history.md


def fit(params: dict, df: pd.DataFrame, cols: list[str]) -> HistGradientBoostingRegressor:
    """v2: learn the *change* from the last known lab value, not the level.

    The baseline carries the current level; the model only has to learn the move. This avoids
    asking a tree model to extrapolate when the plant's average silica drifts.
    """
    return make_model(**params).fit(df[cols], df["y"] - df["baseline"])


def predict(model: HistGradientBoostingRegressor, df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df["baseline"] + model.predict(df[cols])


def tune(train: pd.DataFrame, val: pd.DataFrame, cols: list[str]) -> tuple[dict, float]:
    """Pick hyper-parameters by validation MAE. The test set is not looked at."""
    best = None
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        model = fit(params, train, cols)
        mae = mean_absolute_error(val["y"], predict(model, val, cols))
        if best is None or mae < best[1]:
            best = (params, mae)
    return best


def ship_gate(model_mae: float, baseline_mae: float, margin: float = MARGIN) -> bool:
    return model_mae <= (1 - margin) * baseline_mae


def main(hourly: pd.DataFrame | None = None) -> dict:
    if hourly is None:
        hourly = pd.read_csv(HOURLY_CSV, index_col="date", parse_dates=["date"]).asfreq("h")
    data = build_dataset(hourly)
    cols = feature_cols(data)
    train, val, test = time_split(data)

    params, model_val = tune(train, val, cols)
    final = fit(params, pd.concat([train, val]), cols)
    model_test = mean_absolute_error(test["y"], predict(final, test, cols))
    base_val = mean_absolute_error(val["y"], val["baseline"])
    base_test = mean_absolute_error(test["y"], test["baseline"])
    ship = ship_gate(model_test, base_test)

    metrics = {
        "model_version": MODEL_VERSION,
        "splits": {
            name: {"rows": len(d), "start": str(d.index.min()), "end": str(d.index.max())}
            for name, d in [("train", train), ("val", val), ("test", test)]
        },
        "n_features": len(cols),
        "best_params": params,
        "mae": {
            "baseline_val": base_val, "model_val": model_val,
            "baseline_test": base_test, "model_test": model_test,
        },
        "test_improvement_pct": 100 * (1 - model_test / base_test),
        "gate": {"margin": MARGIN, "decision": "SHIP" if ship else "DON'T SHIP"},
    }
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for name, s in metrics["splits"].items():
        print(f"[split] {name:5s} {s['rows']:5d} rows  {s['start'][:10]} -> {s['end'][:10]}")
    print(f"[model] {MODEL_VERSION}, best params (by val MAE): {params}")
    print("\n            baseline    model")
    print(f"  val MAE    {base_val:8.3f} {model_val:8.3f}")
    print(f"  test MAE   {base_test:8.3f} {model_test:8.3f}   ({metrics['test_improvement_pct']:+.1f}% vs baseline)")
    print(f"\nShip gate: model must be >= {MARGIN:.0%} better than baseline on test MAE.")
    print(f">>> {metrics['gate']['decision']}")
    return metrics


if __name__ == "__main__":
    main()
