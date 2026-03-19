import pandas as pd
import glob
import os

# Path to folder containing CSV files
folder_path = "raw/with_envelope_40.0_ms"

# Find all CSV files
csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

# List to hold all dataframes
dfs = []

# Read each CSV
for file in csv_files:
    df = pd.read_csv(file)
    dfs.append(df)

# Combine all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Calculate min and max for each column
min_values = combined_df.min(numeric_only=True)
max_values = combined_df.max(numeric_only=True)

# Print results
print("Minimum values per column:")
print(min_values)

print("\nMaximum values per column:")
print(max_values)
