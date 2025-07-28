import pandas as pd
from pathlib import Path
import re

# === CONFIGURATION ===
TENSOR_DIR = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\data\Ind_Train_Tensors")
METADATA_PATH = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\Prepared_English_Videos_Metadata - Fixed_Unstratified.xlsx")
OUTPUT_PATH = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\features\Matched_Metadata_Cleaned.xlsx")

# === SANITIZATION FUNCTION ===
def sanitize(text):
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'\s+', '_', text)  # Replace spaces with underscores
    text = re.sub(r'[^a-z0-9_]', '', text)  # Remove special characters
    return text.strip()

# === LOAD METADATA SHEET ===
df = pd.read_excel(METADATA_PATH, sheet_name='Ind_Train')
df['Sanitized_Title'] = df['Video Title'].apply(sanitize)

# === BUILD TENSOR FILE INDEX ===
tensor_files = list(TENSOR_DIR.glob("*.pt"))
tensor_map = {}  # sanitized_name -> actual filename

for file in tensor_files:
    sanitized_name = sanitize(file.stem)
    tensor_map[sanitized_name] = file.name  # store full name

# === CHECK FOR MATCHES ===
match_status = []
matched_tensor_file = []

for i, row in df.iterrows():
    sanitized_title = row['Sanitized_Title']
    if sanitized_title in tensor_map:
        match_status.append("Yes")
        matched_tensor_file.append(tensor_map[sanitized_title])
    else:
        match_status.append("No")
        matched_tensor_file.append("")

df['Tensor_Exists'] = match_status
df['Matched_Tensor_File'] = matched_tensor_file

# === DEBUG OUTPUT ===
matched_count = df['Tensor_Exists'].value_counts().get("Yes", 0)
unmatched_count = df['Tensor_Exists'].value_counts().get("No", 0)
print(f"✅ Matched {matched_count} out of {len(df)} metadata entries with tensor files.")
print(f"❌ Unmatched metadata entries: {unmatched_count}")
print(f"🔢 Total .pt files in folder: {len(tensor_files)}")

# === SAVE TO EXCEL ===
df.to_excel(OUTPUT_PATH, index=False)
print(f"✅ File match results saved to: {OUTPUT_PATH}")
# === DONE ===
print("Feature matching complete. Check the output Excel file for results.")
# === END OF SCRIPT ===
# This script matches metadata entries with tensor files based on sanitized titles.
# It sanitizes both the metadata titles and tensor filenames to ensure consistent matching. 
