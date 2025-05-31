# Load the uploaded dataset
import pandas as pd
file_path = "/Volumes/Disk_1/RADHE______FINAL HEALTHCARE PROJECT WITH CSV/healthcare_medimind/medimind_saved_results.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print(df.head())
