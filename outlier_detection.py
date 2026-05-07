import numpy as np
import pandas as pd

INPUT_PATH = "CRMLSSold_Engineered.csv"
FLAGGED_PATH = "CRMLSSold_Flagged.csv"
FILTERED_PATH = "CRMLSSold_Filtered.csv"

TARGET_COLS = ["ClosePrice", "LivingArea", "DaysOnMarket"]


def iqr_bounds(series: pd.Series):
    s = series.dropna()
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return q1, q3, iqr, lower, upper


def main():
    print(f"[load] {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"  rows={len(df):,}  cols={df.shape[1]}")

    print("\n[IQR bounds]")
    bounds_rows = []
    for col in TARGET_COLS:
        q1, q3, iqr, lower, upper = iqr_bounds(df[col])
        flag_col = f"outlier_{col}"
        df[flag_col] = (df[col] < lower) | (df[col] > upper)
        bounds_rows.append({
            "field": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower": lower,
            "upper": upper,
        })
    bounds_df = pd.DataFrame(bounds_rows).set_index("field").round(3)
    print(bounds_df.to_string())

    print("\n[outlier counts]")
    n = len(df)
    counts_rows = []
    for col in TARGET_COLS:
        flag_col = f"outlier_{col}"
        n_out = int(df[flag_col].sum())
        counts_rows.append({
            "field": col,
            "outliers": n_out,
            "total": n,
            "pct": round(100.0 * n_out / n, 3),
        })
    counts_df = pd.DataFrame(counts_rows).set_index("field")
    print(counts_df.to_string())

    any_outlier = df[[f"outlier_{c}" for c in TARGET_COLS]].any(axis=1)
    df_filtered = df[~any_outlier].copy()

    print("\n[before vs after] dataset size and medians")
    summary_rows = [{
        "metric": "row_count",
        "before": len(df),
        "after": len(df_filtered),
        "delta": len(df_filtered) - len(df),
    }]
    for col in TARGET_COLS:
        m_before = df[col].median()
        m_after = df_filtered[col].median()
        summary_rows.append({
            "metric": f"{col}_median",
            "before": round(m_before, 3),
            "after": round(m_after, 3),
            "delta": round(m_after - m_before, 3),
        })
    summary_df = pd.DataFrame(summary_rows).set_index("metric")
    print(summary_df.to_string())

    print(f"\n[save] {FLAGGED_PATH} (all rows + flag cols)")
    df.to_csv(FLAGGED_PATH, index=False)
    print(f"  rows={len(df):,}  cols={df.shape[1]}")

    print(f"\n[save] {FILTERED_PATH} (no outliers)")
    df_filtered.to_csv(FILTERED_PATH, index=False)
    print(f"  rows={len(df_filtered):,}  cols={df_filtered.shape[1]}")

    print("\n[summary] before vs after")
    print(f"  rows:              {len(df):,}  ->  {len(df_filtered):,}  (removed {len(df) - len(df_filtered):,})")
    for col in TARGET_COLS:
        m_before = df[col].median()
        m_after = df_filtered[col].median()
        print(f"  {col} median:  {m_before:,.2f}  ->  {m_after:,.2f}")


if __name__ == "__main__":
    main()
