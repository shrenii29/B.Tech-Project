import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

def flatten_mimic_data(hosp_dir, output_file="mimic_iv_flat.csv"):
    print("--- Starting MIMIC-IV Tabular Flattening ---")
    
    # Define exact paths based on your screenshot
    patients_path = os.path.join(hosp_dir, "patients.csv")
    labevents_path = os.path.join(hosp_dir, "labevents.csv")
    labitems_path = os.path.join(hosp_dir, "d_labitems.csv")
    
    print("1. Loading Patient Demographics...")
    patients_df = pd.read_csv(patients_path)
    
    # We only need the core demographics that are numeric or can be mapped
    patients_df = patients_df[['subject_id', 'gender', 'anchor_age']]
    patients_df['gender'] = patients_df['gender'].map({'F': 0, 'M': 1}) # Convert to binary
    
    print("2. Loading Lab Events (This might take a moment)...")
    labs_df = pd.read_csv(labevents_path)
    
    # Drop rows where there is no numerical lab result
    labs_df = labs_df.dropna(subset=['valuenum'])
    
    print(f"Total valid lab readings found: {len(labs_df)}")
    
    # To prevent exploding the columns to 1000+, we will take the top 30 most frequent lab tests
    top_items = labs_df['itemid'].value_counts().head(30).index
    labs_df = labs_df[labs_df['itemid'].isin(top_items)]
    
    print("3. Pivoting data (1 row per patient)...")
    # If a patient had 5 blood tests, we take the mean average of those 5 tests
    pivoted_labs = pd.pivot_table(
        labs_df, 
        values='valuenum', 
        index='subject_id', 
        columns='itemid', 
        aggfunc='mean'
    ).reset_index()
    
    print("4. Translating Medical Codes to English Names...")
    items_dict = pd.read_csv(labitems_path)
    # Create a dictionary mapping the numeric itemid to the actual medical label string
    id_to_label = dict(zip(items_dict.itemid, items_dict.label))
    
    # Rename the columns from numbers (e.g., 50912) to text (e.g., 'Creatinine')
    pivoted_labs = pivoted_labs.rename(columns=id_to_label)
    
    # Clean up column names so they don't break PyTorch later (replace spaces with underscores)
    pivoted_labs.columns = [str(col).replace(' ', '_').replace(',', '') for col in pivoted_labs.columns]
    
    print("5. Merging Demographics and Labs...")
    final_df = pd.merge(patients_df, pivoted_labs, on='subject_id', how='inner')
    
    # Deep Learning hates missing values (NaN). We will fill missing lab results with the column average
    final_df = final_df.fillna(final_df.mean())
    
    print(f"Final Flat Dataset Shape: {final_df.shape} (Patients, Features)")
    
    # Save the flattened file
    final_df.to_csv(output_file, index=False)
    print(f"\nSUCCESS! Saved flattened clinical data to: {output_file}")
    print("You can now feed this file into the Data Aligner script.")

if __name__ == "__main__":
    # --- YOUR TASK ---
    # Update this to the exact path of your 'hosp' folder
    HOSP_DIR = r"C:\Users\Shreni\OneDrive\Desktop\B.Tech-Project\dataset-iv\mimic-iv-clinical-database-demo-2.2\hosp"
    
    # Output file will be saved in the same directory you run this script
    flatten_mimic_data(HOSP_DIR)