import os
import pandas as pd

# Folder path where the .csv files are stored
folder_path = r'C:\Users\Theeraj\Desktop\New folder\dataset'

# To store file names and their headers
csv_files_info = {}

# Loop through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        # Full path of the CSV file
        file_path = os.path.join(folder_path, file_name)
        
        # Read the CSV file using pandas and get the headers (columns)
        try:
            df = pd.read_csv(file_path)
            headers = list(df.columns)  # Get the column names
            csv_files_info[file_name] = headers
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

# Display the file names and headers
for file, headers in csv_files_info.items():
    print(f"File: {file}")
    print(f"Headers: {headers}\n")
