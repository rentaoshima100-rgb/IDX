import pandas as pd
import glob

# ============================================================
# WEEK 1: Monthly Dataset Aggregation
# ============================================================

listing_files = glob.glob('CRMLSListing*.csv')
print(f"Found {len(listing_files)} listing files:")
for f in sorted(listing_files):
    print(f"  {f}")

listed = pd.concat([pd.read_csv(f, encoding='latin-1', low_memory=False) for f in listing_files], ignore_index=True)
print(f"\nTotal records before filter: {len(listed)}")

print("\n--- Property Types ---")
print(listed['PropertyType'].value_counts())

listed = listed[listed['PropertyType'] == 'Residential']
print(f"\nTotal records after Residential filter: {len(listed)}")

listed.to_csv('CRMLSListing_Combined_Residential.csv', index=False)
print("Saved: CRMLSListing_Combined_Residential.csv")

# ============================================================
# WEEK 2: Dataset Structuring and Validation
# ============================================================

print("\n\n========== WEEK 2: EDA ==========")
print(f"Rows: {len(listed)}, Columns: {len(listed.columns)}")
print("\n--- Data Types ---")
print(listed.dtypes)

print("\n--- Missing Value Report ---")
null_counts = listed.isnull().sum()
null_pct = (null_counts / len(listed) * 100).round(2)
missing_report = pd.DataFrame({
    'null_count': null_counts,
    'null_pct': null_pct
}).sort_values('null_pct', ascending=False)
print(missing_report[missing_report['null_count'] > 0])

high_missing = missing_report[missing_report['null_pct'] > 90]
print(f"\n--- Columns with >90% missing ({len(high_missing)}) ---")
print(high_missing)

print("\n--- Numeric Distribution: ListPrice ---")
print(listed['ListPrice'].describe(percentiles=[.10, .25, .50, .75, .90]))

print("\n--- Numeric Distribution: LivingArea ---")
print(listed['LivingArea'].describe(percentiles=[.10, .25, .50, .75, .90]))

print("\n--- Numeric Distribution: DaysOnMarket ---")
print(listed['DaysOnMarket'].describe(percentiles=[.10, .25, .50, .75, .90]))

listed.to_csv('CRMLSListing_Week2.csv', index=False)
print("\nSaved: CRMLSListing_Week2.csv")