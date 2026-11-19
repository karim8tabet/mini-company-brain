"""Step 1: load the raw plant data, fix types, resample to hourly, write a data-quality report."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
KAGGLE_CSV = RAW_DIR / "MiningProcess_Flotation_Plant_Database.csv"
OPENML_ID = 43311
HOURLY_CSV = ROOT / "data" / "processed" / "hourly.csv"
REPORT = ROOT / "reports" / "data_quality.md"

TARGET = "pct_silica_concentrate"
# Lab assays: measured hourly or less often, results arrive ~2 h late.
LAB_COLS = ["pct_iron_feed", "pct_silica_feed", "pct_iron_concentrate", TARGET]
ROWS_PER_HOUR = 180  # one row every 20 s


def normalize_name(col: str) -> str:
    """'% Iron Feed' (Kaggle) and '%_Iron_Feed' (OpenML) both become 'pct_iron_feed'."""
    c = col.strip().replace("%", "pct").replace(" ", "_").lower()
    while "__" in c:
        c = c.replace("__", "_")
    return c


def load_raw() -> pd.DataFrame:
    """Raw rows as strings. Uses the Kaggle CSV if present, else OpenML (cached under data/raw)."""
    if KAGGLE_CSV.exists():
        df = pd.read_csv(KAGGLE_CSV, dtype=str)
    else:
        from sklearn.datasets import fetch_openml

        df = fetch_openml(
            data_id=OPENML_ID, as_frame=True, data_home=str(RAW_DIR / "openml"), parser="auto"
        ).frame
    df.columns = [normalize_name(c) for c in df.columns]
    return df.astype(str)


def parse_types(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Comma decimals -> floats, date strings -> timestamps. Returns (df, count of unparseable values)."""
    out = pd.DataFrame({"date": pd.to_datetime(df["date"], format="%Y-%m-%d %H:%M:%S", errors="coerce")})
    bad = {"date": int(out["date"].isna().sum())}
    for col in df.columns.drop("date"):
        out[col] = pd.to_numeric(df[col].str.replace(",", ".", regex=False), errors="coerce")
        bad[col] = int(out[col].isna().sum())
    if sum(bad.values()) > 0.01 * len(df):
        raise ValueError(f"More than 1% of values failed to parse: {bad}")
    return out.dropna(subset=["date"]), bad


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """One row per hour on a complete hourly index; missing hours become all-NaN rows."""
    g = df.groupby("date")
    sensors = [c for c in df.columns if c not in LAB_COLS + ["date"]]
    hourly = g[sensors].mean()
    # First value of the hour = the actual assay; later rows may be interpolated toward the next one.
    hourly[LAB_COLS] = g[LAB_COLS].first()
    hourly["n_rows"] = g.size()
    full = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h", name="date")
    hourly = hourly.reindex(full)
    hourly["n_rows"] = hourly["n_rows"].fillna(0).astype(int)
    return hourly


def find_gaps(hourly: pd.DataFrame) -> pd.DataFrame:
    """Runs of consecutive hours with no data."""
    missing = hourly["n_rows"].eq(0)
    run_id = (missing != missing.shift()).cumsum()
    runs = [
        (grp.index[0], grp.index[-1], len(grp))
        for _, grp in hourly[missing].groupby(run_id[missing])
    ]
    return pd.DataFrame(runs, columns=["start", "end", "hours"])


def out_of_range(df: pd.DataFrame) -> dict:
    """Count raw rows breaking physical limits."""
    checks = {}
    for col in df.columns.drop("date"):
        s = df[col]
        if col.startswith("pct_"):
            checks[col + " outside 0-100"] = int(((s < 0) | (s > 100)).sum())
        elif col == "ore_pulp_ph":
            checks[col + " outside 0-14"] = int(((s < 0) | (s > 14)).sum())
        else:
            checks[col + " negative"] = int((s < 0).sum())
    return {k: v for k, v in checks.items() if v}


def md_table(df: pd.DataFrame) -> str:
    """Tiny markdown table (avoids adding the 'tabulate' dependency)."""
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def write_report(raw: pd.DataFrame, bad: dict, hourly: pd.DataFrame, lab_varies: dict) -> str:
    gaps = find_gaps(hourly)
    present = int(hourly["n_rows"].gt(0).sum())
    odd = hourly[hourly["n_rows"].between(1, ROWS_PER_HOUR - 1) | hourly["n_rows"].gt(ROWS_PER_HOUR)]
    oor = out_of_range(raw)
    lines = [
        "# Data-quality report",
        "",
        f"- Raw rows: {len(raw):,}",
        f"- Date range: {hourly.index.min()} to {hourly.index.max()}",
        f"- Hours in range: {len(hourly):,}; with data: {present:,}; missing: {len(hourly) - present:,}",
        f"- Unparseable values: {sum(bad.values())}",
        "",
        "## Gaps (consecutive missing hours)",
        "",
        md_table(gaps) if len(gaps) else "None.",
        "",
        f"## Hours with an unusual row count (expected {ROWS_PER_HOUR})",
        "",
        md_table(odd["n_rows"].reset_index()) if len(odd) else "None.",
        "",
        "## Out-of-range raw values",
        "",
        "\n".join(f"- {k}: {v}" for k, v in oor.items()) if oor else "None.",
        "",
        "## Lab columns that vary within an hour",
        "",
        "Hours where a lab value is not constant across its 20 s rows (sign of interpolation; we keep the first value).",
        "",
        "\n".join(f"- {k}: {v} hours" for k, v in lab_varies.items()),
        "",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    return f"{len(hourly):,} hours, {len(hourly) - present:,} missing, biggest gap = {gaps['hours'].max() if len(gaps) else 0} h"


def main() -> pd.DataFrame:
    raw, bad = parse_types(load_raw())
    lab_varies = {c: int(raw.groupby("date")[c].nunique().gt(1).sum()) for c in LAB_COLS}
    hourly = to_hourly(raw)
    summary = write_report(raw, bad, hourly, lab_varies)
    HOURLY_CSV.parent.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(HOURLY_CSV)
    print(f"[clean] {summary} -> {REPORT.relative_to(ROOT)}")
    return hourly


if __name__ == "__main__":
    main()
