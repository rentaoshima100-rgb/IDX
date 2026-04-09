import pandas as pd
import glob

# ============================================================
# WEEK 1: Monthly Dataset Aggregation
# ============================================================

sold_files = glob.glob('CRMLSSold*.csv')
print(f"Found {len(sold_files)} sold files:")
for f in sorted(sold_files):
    print(f"  {f}")

sold = pd.concat([pd.read_csv(f, encoding='latin-1', low_memory=False) for f in sold_files], ignore_index=True)
print(f"\nTotal records before filter: {len(sold)}")

# Property type breakdown
print("\n--- Property Types ---")
print(sold['PropertyType'].value_counts())

# Filter to Residential only
sold = sold[sold['PropertyType'] == 'Residential']
print(f"\nTotal records after Residential filter: {len(sold)}")

# Save Week 1 output
sold.to_csv('CRMLSSold_Combined_Residential.csv', index=False)
print("Saved: CRMLSSold_Combined_Residential.csv")

# ============================================================
# WEEK 2: Dataset Structuring and Validation
# ============================================================

# --- Basic Structure ---
print("\n\n========== WEEK 2: EDA ==========")
print(f"Rows: {len(sold)}, Columns: {len(sold.columns)}")
print("\n--- Data Types ---")
print(sold.dtypes)

# --- Missing Value Analysis ---
print("\n--- Missing Value Report ---")
null_counts = sold.isnull().sum()
null_pct = (null_counts / len(sold) * 100).round(2)
missing_report = pd.DataFrame({
    'null_count': null_counts,
    'null_pct': null_pct
}).sort_values('null_pct', ascending=False)
print(missing_report[missing_report['null_count'] > 0])

# Flag columns with >90% missing
high_missing = missing_report[missing_report['null_pct'] > 90]
print(f"\n--- Columns with >90% missing ({len(high_missing)}) ---")
print(high_missing)

# --- Numeric Distribution Summary ---
print("\n--- Numeric Distribution: ClosePrice ---")
print(sold['ClosePrice'].describe(percentiles=[.10, .25, .50, .75, .90]))

print("\n--- Numeric Distribution: LivingArea ---")
print(sold['LivingArea'].describe(percentiles=[.10, .25, .50, .75, .90]))

print("\n--- Numeric Distribution: DaysOnMarket ---")
print(sold['DaysOnMarket'].describe(percentiles=[.10, .25, .50, .75, .90]))

# --- Suggested Questions ---
median_close = sold['ClosePrice'].median()
mean_close = sold['ClosePrice'].mean()
print(f"\nMedian ClosePrice: ${median_close:,.0f}")
print(f"Mean ClosePrice: ${mean_close:,.0f}")

above_list = (sold['ClosePrice'] > sold['ListPrice']).sum()
below_list = (sold['ClosePrice'] <= sold['ListPrice']).sum()
total_valid = above_list + below_list
print(f"\nSold above list price: {above_list} ({above_list/total_valid*100:.1f}%)")
print(f"Sold at or below list: {below_list} ({below_list/total_valid*100:.1f}%)")

print("\n--- Median ClosePrice by County ---")
print(sold.groupby('CountyOrParish')['ClosePrice'].median().sort_values(ascending=False))

# Save filtered dataset
sold.to_csv('CRMLSSold_Week2.csv', index=False)
print("\nSaved: CRMLSSold_Week2.csv")