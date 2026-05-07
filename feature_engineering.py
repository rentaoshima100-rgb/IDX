import numpy as np
import pandas as pd

INPUT_PATH = "CRMLSSold_Cleaned.csv"
OUTPUT_PATH = "CRMLSSold_Engineered.csv"


def main():
    print(f"[load] {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH, low_memory=False)
    print(f"  rows={len(df):,}  cols={df.shape[1]}")

    for col in ["CloseDate", "PurchaseContractDate", "ListingContractDate"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df["price_ratio"] = df["ClosePrice"] / df["OriginalListPrice"]
    df["close_to_original_list_ratio"] = df["ClosePrice"] / df["OriginalListPrice"]
    df["price_per_sqft"] = df["ClosePrice"] / df["LivingArea"]
    df["days_on_market"] = df["DaysOnMarket"]
    df["close_year"] = df["CloseDate"].dt.year
    df["close_month"] = df["CloseDate"].dt.month
    df["yr_mo"] = df["CloseDate"].dt.to_period("M").astype(str)
    df["listing_to_contract_days"] = (
        df["PurchaseContractDate"] - df["ListingContractDate"]
    ).dt.days
    df["contract_to_close_days"] = (
        df["CloseDate"] - df["PurchaseContractDate"]
    ).dt.days

    new_cols = [
        "price_ratio",
        "close_to_original_list_ratio",
        "price_per_sqft",
        "days_on_market",
        "close_year",
        "close_month",
        "yr_mo",
        "listing_to_contract_days",
        "contract_to_close_days",
    ]
    df[new_cols] = df[new_cols].replace([np.inf, -np.inf], np.nan)

    print("\n[sample] head(10) of engineered columns")
    sample_cols = [
        "ClosePrice",
        "OriginalListPrice",
        "LivingArea",
        "CloseDate",
        "PurchaseContractDate",
        "ListingContractDate",
    ] + new_cols
    print(df[sample_cols].head(10).to_string())

    metrics = ["price_ratio", "price_per_sqft", "days_on_market", "listing_to_contract_days"]

    print("\n[segment] CountyOrParish summary (mean / median)")
    county_summary = (
        df.groupby("CountyOrParish")[metrics]
        .agg(["mean", "median", "count"])
        .round(3)
    )
    county_summary = county_summary.sort_values(("price_per_sqft", "count"), ascending=False)
    print(county_summary.to_string())

    print("\n[segment] PropertySubType summary (mean / median)")
    subtype_summary = (
        df.groupby("PropertySubType")[metrics]
        .agg(["mean", "median", "count"])
        .round(3)
    )
    subtype_summary = subtype_summary.sort_values(("price_per_sqft", "count"), ascending=False)
    print(subtype_summary.to_string())

    print(f"\n[save] {OUTPUT_PATH}")
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"  rows={len(df):,}  cols={df.shape[1]}")


if __name__ == "__main__":
    main()
