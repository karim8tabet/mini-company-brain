"""Steps 3-5 + 7: persistence baseline vs gradient-boosting model, then the ship gate."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from brain.clean import HOURLY_CSV, ROOT
from brain.features import build_dataset, feature_cols, time_split

METRICS = ROOT / "reports" / "metrics.json"
TEST_PREDS = ROOT / "reports" / "test_predictions.csv"
QUANTILES = (0.1, 0.9)  # 80% range
MARGIN = 0.05  # ship only if test MAE is at least 5% better than the baseline
SEED = 0
GRID = {
    "learning_rate": [0.03, 0.1],
    "max_leaf_nodes": [15, 31],
    "max_iter": [200, 500],
}


def make_model(loss: str = "absolute_error", **params) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss=loss, random_state=SEED, early_stopping=False, **params
    )


MODEL_VERSION = "v2"  # v1 predicted the level and failed the gate; see reports/model_history.md


def fit(params: dict, df: pd.DataFrame, cols: list[str], **loss) -> HistGradientBoostingRegressor:
    """v2: learn the *change* from the last known lab value, not the level.

    The baseline carries the current level; the model only has to learn the move. This avoids
    asking a tree model to extrapolate when the plant's average silica drifts.
    """
    return make_model(**loss, **params).fit(df[cols], df["y"] - df["baseline"])


def predict_range(params: dict, fit_df: pd.DataFrame, df: pd.DataFrame, cols: list[str]):
    """80% range from two quantile models (0.1 and 0.9), also on the change scale."""
    lo, hi = (
        predict(fit(params, fit_df, cols, loss="quantile", quantile=q), df, cols) for q in QUANTILES
    )
    # Two separately trained models can cross; sort so lo <= hi.
    return np.minimum(lo, hi), np.maximum(lo, hi)


def coverage(y: pd.Series, lo: pd.Series, hi: pd.Series) -> float:
    return float(((y >= lo) & (y <= hi)).mean())


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
    train_val = pd.concat([train, val])
    final = fit(params, train_val, cols)
    test_pred = predict(final, test, cols)
    model_test = mean_absolute_error(test["y"], test_pred)
    lo, hi = predict_range(params, train_val, test, cols)
    cov = coverage(test["y"], lo, hi)
    pd.DataFrame(
        {"actual": test["y"], "forecast": test_pred, "baseline": test["baseline"], "lo": lo, "hi": hi}
    ).rename_axis("origin").to_csv(TEST_PREDS)
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
        "range_80": {
            "test_coverage": cov,
            "mean_width": float((hi - lo).mean()),
        },
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
    print(f"\n80% range on test: {cov:.1%} of actual values inside (target 80%), "
          f"mean width {metrics['range_80']['mean_width']:.2f} %-points")
    print(f"\nShip gate: model must be >= {MARGIN:.0%} better than baseline on test MAE.")
    print(f">>> {metrics['gate']['decision']}")
    return metrics


if __name__ == "__main__":
    main()
