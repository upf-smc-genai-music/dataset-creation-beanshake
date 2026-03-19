import os
import pandas as pd

# Path to folder with CSV files
folder_path = "raw"

for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # Read CSV
        df = pd.read_csv(file_path)

        # Drop the second column (index 1)
        df = df.drop(df.columns[1], axis=1)

        # Save back with same name (overwrite)
        df.to_csv(file_path, index=False)

print("Finished processing all CSV files.")