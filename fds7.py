# import pandas as pd
# import numpy as np

# # Load the CSV
# excel_path = r"C:\Users\User\Desktop\siemens\freya_schulte\imo_9665671_ME1_FMS_act_kgPh@AVG_dump.csv"
# df = pd.read_csv(excel_path, low_memory=False)

# target_tag = 'FDS7_act_kgPm3@AVG'

# print("=" * 60)
# print(f"Diagnostic Report for: {target_tag}")
# print("=" * 60)

# # Check 1: Does column exist?
# if target_tag not in df.columns:
#     print(f"❌ Column '{target_tag}' does NOT exist in CSV")
#     exit()
# else:
#     print(f"✅ Column exists")

# # Check 2: Basic stats
# print(f"\n📊 Basic Statistics:")
# print(f"   Total rows: {len(df)}")
# print(f"   Non-null values: {df[target_tag].notna().sum()}")
# print(f"   Null values: {df[target_tag].isna().sum()}")
# print(f"   Data type: {df[target_tag].dtype}")

# # Check 3: Numeric check
# if pd.api.types.is_numeric_dtype(df[target_tag]):
#     print(f"   ✅ Is numeric")
# else:
#     print(f"   ❌ NOT numeric")

# # Check 4: Variance
# print(f"\n📈 Variance Analysis:")
# print(f"   Variance: {df[target_tag].var()}")
# print(f"   Standard deviation: {df[target_tag].std()}")
# print(f"   Min: {df[target_tag].min()}")
# print(f"   Max: {df[target_tag].max()}")
# print(f"   Unique values: {df[target_tag].nunique()}")

# # Check 5: Test with one other column
# print(f"\n🔍 Testing with sample column (time):")
# test_col = 'time'
# valid_data = df[[target_tag, test_col]].dropna()
# print(f"   Valid rows after dropna: {len(valid_data)}")
# print(f"   FDS7 variance in valid data: {valid_data[target_tag].var()}")
# print(f"   Time variance in valid data: {valid_data[test_col].var()}")

# # Check 6: Sample values
# print(f"\n🔬 First 10 non-null values:")
# print(df[target_tag].dropna().head(10).tolist())

# print("\n" + "=" * 60)
import pandas as pd
import numpy as np

# Load the CSV
excel_path = r"C:\Users\User\Desktop\siemens\freya_schulte\imo_9665671_ME1_FMS_act_kgPh@AVG_dump.csv"
df = pd.read_csv(excel_path, low_memory=False)

print("=" * 80)
print("DATA QUALITY ANALYSIS - Finding Faulty Tags")
print("=" * 80)
print(f"Total rows: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print()

# Get only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"Numeric columns: {len(numeric_cols)}")
print()

# Categorize issues
issues = {
    "zero_variance": [],
    "all_null": [],
    "mostly_null": [],  # >95% null
    "low_variance": [],  # variance < 0.001
    "constant_with_nulls": []  # only 1 unique value (excluding nulls)
}

print("Analyzing columns...")
print("-" * 80)

for col in numeric_cols:
    # Check 1: All null
    null_count = df[col].isna().sum()
    null_percentage = (null_count / len(df)) * 100
    
    if null_count == len(df):
        issues["all_null"].append({
            "tag": col,
            "issue": "All values are NULL"
        })
        continue
    
    # Check 2: Mostly null (>95%)
    if null_percentage > 95:
        issues["mostly_null"].append({
            "tag": col,
            "issue": f"{null_percentage:.1f}% NULL values"
        })
        continue
    
    # For remaining checks, use non-null values
    valid_data = df[col].dropna()
    
    if len(valid_data) == 0:
        continue
    
    # Check 3: Zero variance (all same value, like FDS7)
    variance = valid_data.var()
    unique_count = valid_data.nunique()
    
    if variance == 0 or unique_count == 1:
        issues["zero_variance"].append({
            "tag": col,
            "issue": f"Zero variance - all values are {valid_data.iloc[0]}",
            "unique_values": unique_count,
            "sample_value": valid_data.iloc[0]
        })
        continue
    
    # Check 4: Very low variance
    if variance < 0.001:
        issues["low_variance"].append({
            "tag": col,
            "issue": f"Very low variance ({variance:.6f})",
            "min": valid_data.min(),
            "max": valid_data.max(),
            "unique_values": unique_count
        })

# Print results
print("\n" + "=" * 80)
print("SUMMARY OF ISSUES")
print("=" * 80)

print(f"\n1. ZERO VARIANCE (Like FDS7) - {len(issues['zero_variance'])} tags:")
print("-" * 80)
if issues['zero_variance']:
    for item in issues['zero_variance']:
        print(f"   {item['tag']}: {item['issue']}")
else:
    print("   None found")

print(f"\n2. ALL NULL - {len(issues['all_null'])} tags:")
print("-" * 80)
if issues['all_null']:
    for item in issues['all_null']:
        print(f"   {item['tag']}: {item['issue']}")
else:
    print("   None found")

print(f"\n3. MOSTLY NULL (>95%) - {len(issues['mostly_null'])} tags:")
print("-" * 80)
if issues['mostly_null']:
    for item in issues['mostly_null'][:10]:  # Show first 10
        print(f"   {item['tag']}: {item['issue']}")
    if len(issues['mostly_null']) > 10:
        print(f"   ... and {len(issues['mostly_null']) - 10} more")
else:
    print("   None found")

print(f"\n4. VERY LOW VARIANCE - {len(issues['low_variance'])} tags:")
print("-" * 80)
if issues['low_variance']:
    for item in issues['low_variance'][:10]:  # Show first 10
        print(f"   {item['tag']}: variance={item['issue']}, unique={item['unique_values']}")
    if len(issues['low_variance']) > 10:
        print(f"   ... and {len(issues['low_variance']) - 10} more")
else:
    print("   None found")

# Summary statistics
total_problematic = (len(issues['zero_variance']) + 
                     len(issues['all_null']) + 
                     len(issues['mostly_null']) + 
                     len(issues['low_variance']))

print("\n" + "=" * 80)
print(f"TOTAL PROBLEMATIC TAGS: {total_problematic} out of {len(numeric_cols)} numeric columns")
print(f"HEALTHY TAGS: {len(numeric_cols) - total_problematic}")
print("=" * 80)

# Save detailed report to file
with open("faulty_tags_report.txt", "w") as f:
    f.write("DETAILED FAULTY TAGS REPORT\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("ZERO VARIANCE TAGS:\n")
    for item in issues['zero_variance']:
        f.write(f"  {item['tag']}: {item['issue']}\n")
    
    f.write("\nALL NULL TAGS:\n")
    for item in issues['all_null']:
        f.write(f"  {item['tag']}: {item['issue']}\n")
    
    f.write("\nMOSTLY NULL TAGS:\n")
    for item in issues['mostly_null']:
        f.write(f"  {item['tag']}: {item['issue']}\n")
    
    f.write("\nLOW VARIANCE TAGS:\n")
    for item in issues['low_variance']:
        f.write(f"  {item['tag']}: {item['issue']}\n")

print("\nDetailed report saved to: faulty_tags_report.txt")