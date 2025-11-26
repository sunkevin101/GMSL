# Convert a txt file to CSV and save only specified columns


import csv
import pandas as pd
from utils import decorator

def txt_to_csv(txt_file, csv_file, delimiter='\t'):
    """
    Convert a tab-delimited text file to a CSV file.
    """
    with open(txt_file, 'r', encoding='utf-8') as infile, open(csv_file, 'w', newline='', encoding='utf-8') as outfile:
        # Use csv moduleto read txt file
        reader = csv.reader(infile, delimiter=delimiter)
        writer = csv.writer(outfile)

        # Read and write each row to CSV file
        for row in reader:
            writer.writerow(row)

@decorator
def main():
    txt_file_path = 'preprocess/data_processing/cBioPortal/data_clinical_patient.txt'
    csv_file_path = 'preprocess/data_processing/cBioPortal/temp_clinical_all.csv'  # intermediate filename
    output_csv_file = 'preprocess/data_processing/cBioPortal/data_clinical_all_clean.csv'

    txt_to_csv(txt_file_path, csv_file_path)

    df = pd.read_csv(csv_file_path)# Read CSV file

    # Select specified columns
    columns_to_keep = ['PATIENT_ID', 'CANCER_TYPE_ACRONYM', 'OS_MONTHS', 'OS_STATUS']
    filtered_df = df[columns_to_keep]

    # Save the result to a new CSV file
    filtered_df.to_csv(output_csv_file, index=False)

    print(f"Successfully saved specified columns to {output_csv_file}")


if __name__ == "__main__":
    main()
