import os
import sys
from pathlib import Path
import pandas as pd
import re

print("Script starting...")

# Define input and output directories
project_root = Path("D:/10-Academy/Week4")
processed_data_dir = project_root / "amharic-ecommerce-extractor" / "data" / "processed"
labeled_data_dir = project_root / "amharic-ecommerce-extractor" / "data" / "labeled"

print(f"Project root: {project_root}")
print(f"Processed data dir: {processed_data_dir}")
print(f"Labeled data dir: {labeled_data_dir}")

# Create output directory if it doesn't exist
os.makedirs(labeled_data_dir, exist_ok=True)
print(f"Created directory: {labeled_data_dir}")

# Load the NER-ready data
ner_data_path = processed_data_dir / "ner_ready_data.csv"
print(f"Looking for file: {ner_data_path}")
print(f"File exists: {ner_data_path.exists()}")

if not ner_data_path.exists():
    print(f"NER-ready data not found at {ner_data_path}")
    sys.exit(1)

df = pd.read_csv(ner_data_path)
print(f"Loaded {len(df)} tokens from {ner_data_path}")
print(f"DataFrame columns: {df.columns.tolist()}")
print(f"First few rows: {df.head()}")

# Define the path for the CoNLL file
conll_path = labeled_data_dir / "unlabeled_data.conll"
print(f"Will save to: {conll_path}")

# Simple function to convert CSV to CoNLL format
def csv_to_conll(df, output_path):
    print(f"Converting to CoNLL format...")
    # Group by message_id to get sentences
    grouped = df.groupby('message_id')
    print(f"Found {len(grouped)} message groups")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            for name, group in grouped:
                # Write each token and its entity
                for _, row in group.iterrows():
                    f.write(f"{row['token']} {row['entity']}\n")
                # Empty line between sentences
                f.write("\n")
        print(f"Successfully wrote to {output_path}")
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False

# Convert to CoNLL format
result = csv_to_conll(df, conll_path)

print(f"Conversion result: {result}")
print(f"Converted data to CoNLL format and saved to {conll_path}")
print(f"File exists after conversion: {conll_path.exists()}")
print(f"Script completed.") 