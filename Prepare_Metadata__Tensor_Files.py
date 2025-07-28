import pandas as pd
from pathlib import Path
import re
import os

# === CONFIGURATION FOR SCRIPT 1 ===
# Path to your main metadata Excel file
METADATA_PATH = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets\Prepared_English_Videos_Metadata - tesnor based.xlsx")
# Directory containing your raw .pt tensor files (frames extracted from videos)
TENSOR_DIR = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\data\Ind_Test_Tensors")
# Output path for the new prepared metadata Excel file (will have two sheets)
PREPARED_METADATA_PATH = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\features\prepared_for_Test_feature_extraction.xlsx")

# Ensure the output directory exists
PREPARED_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)

# === SANITIZATION FUNCTION ===
def sanitize_for_matching(text):
    """
    Sanitizes text for robust string matching.
    Converts to lowercase, replaces spaces with underscores, and removes/replaces
    problematic characters commonly found in filenames and titles.
    """
    if pd.isna(text):
        return ''
    text = str(text).lower()
    
    # Replace common problematic characters with underscore or remove
    text = re.sub(r'[\s\W_]+', '_', text) # Replace multiple spaces/non-alphanumeric/underscore with single underscore
    
    # Remove specific unicode characters that might appear (like âœ¨)
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    # Remove leading/trailing underscores
    text = text.strip('_')
    
    return text

# === MAIN SCRIPT 1 LOGIC ===

print("🚀 Starting metadata preparation and tensor file matching process...")

# --- 1. Load Metadata ---
try:
    # Load the 'Ind_Train' sheet as requested
    df_metadata = pd.read_excel(METADATA_PATH, sheet_name='Ind_Test')
    print(f"📖 Loaded metadata from: {METADATA_PATH} (Sheet: Ind_Test)")
    print(f"Initial metadata entries: {len(df_metadata)}")
except Exception as e:
    print(f"❌ Error loading metadata from {METADATA_PATH} or 'Ind_Test' sheet: {e}")
    print("Please ensure the path is correct and the sheet exists.")
    exit()

# Ensure serial number is treated as a unique identifier for matching
if 'serial number' in df_metadata.columns:
    # Drop duplicates based on 'serial number' if any, keeping the first occurrence
    initial_rows = len(df_metadata)
    df_metadata.drop_duplicates(subset=['serial number'], inplace=True)
    if len(df_metadata) < initial_rows:
        print(f"⚠️ Removed {initial_rows - len(df_metadata)} duplicate 'serial number' entries from metadata.")
else:
    print("⚠️ 'serial number' column not found in metadata. Cannot ensure uniqueness by serial number.")

# Sanitize 'Video Title' for matching
df_metadata['Sanitized_Video_Title'] = df_metadata['Video Title'].apply(sanitize_for_matching)
print("✅ Sanitized 'Video Title' column for matching.")

# --- NEW: Check for duplicate sanitized titles in metadata ---
duplicate_sanitized_titles = df_metadata[df_metadata.duplicated(subset=['Sanitized_Video_Title'], keep=False)]
if not duplicate_sanitized_titles.empty:
    print(f"⚠️ Found {len(duplicate_sanitized_titles)} metadata entries with duplicate 'Sanitized_Video_Title'.")
    print("These metadata entries will all match the same unique tensor file if one exists.")
    # You might want to inspect these: print(duplicate_sanitized_titles[['serial number', 'Video Title', 'Sanitized_Video_Title']])


# --- 2. Build an Index of Available Tensor Files ---
tensor_files = list(TENSOR_DIR.glob("*.pt"))
tensor_file_map = {} # sanitized_filename -> original_full_filename
tensor_list_for_sheet = [] # Data for the second sheet

if not tensor_files:
    print(f"⚠️ No .pt files found in {TENSOR_DIR}. Please check the directory.")
else:
    for file_path in tensor_files:
        sanitized_stem = sanitize_for_matching(file_path.stem)
        if sanitized_stem in tensor_file_map:
            print(f"⚠️ Duplicate sanitized tensor filename detected: '{sanitized_stem}'. Keeping first instance.")
        tensor_file_map[sanitized_stem] = file_path.name # Store original filename (e.g., 'video_name.pt')
        
        tensor_list_for_sheet.append({
            'Tensor_Filename': file_path.name,
            'Sanitized_Tensor_Name': sanitized_stem
        })

    print(f"🔎 Indexed {len(tensor_file_map)} unique sanitized tensor file names from {len(tensor_files)} files.")

# Create DataFrame for the second sheet
df_tensor_list = pd.DataFrame(tensor_list_for_sheet)

# --- 3. Match Metadata Entries with Tensor Files ---
df_metadata['Tensor_Found_Status'] = 'Not Found' # Initialize status
df_metadata['Matched_Tensor_Filename'] = ''     # To store the actual filename of the matched .pt file

matched_count = 0 # Counts metadata rows that found a match
unique_tensors_matched = set() # Stores unique sanitized tensor names that found a match

for index, row in df_metadata.iterrows():
    sanitized_title = row['Sanitized_Video_Title']
    if sanitized_title in tensor_file_map:
        df_metadata.at[index, 'Tensor_Found_Status'] = 'Found'
        df_metadata.at[index, 'Matched_Tensor_Filename'] = tensor_file_map[sanitized_title]
        matched_count += 1
        unique_tensors_matched.add(sanitized_title) # Add the sanitized tensor name to the set

print("📊 Matching process complete.")

# --- 4. Report and Save Results to Multiple Sheets ---
found_count = df_metadata['Tensor_Found_Status'].value_counts().get('Found', 0)
not_found_count = df_metadata['Tensor_Found_Status'].value_counts().get('Not Found', 0)
unique_tensors_matched_count = len(unique_tensors_matched) # Count of unique tensor files actually used

print(f"\n✅ Matched {found_count} metadata entries with tensor files.")
print(f"❌ {not_found_count} metadata entries did not find a matching tensor file.")
print(f"Total .pt files found in tensor directory: {len(tensor_files)}")
print(f"✅ Number of UNIQUE .pt files that found a match: {unique_tensors_matched_count}") # NEW REPORTING

# Save to Excel with multiple sheets
try:
    with pd.ExcelWriter(PREPARED_METADATA_PATH, engine='xlsxwriter') as writer:
        df_metadata.to_excel(writer, sheet_name='Metadata_For_Extraction', index=False)
        df_tensor_list.to_excel(writer, sheet_name='Tensor_Files_List', index=False)
    print(f"📝 Prepared metadata saved to: {PREPARED_METADATA_PATH} (Sheets: 'Metadata_For_Extraction', 'Tensor_Files_List')")
except Exception as e:
    print(f"❌ Error saving Excel file to {PREPARED_METADATA_PATH}: {e}")
    print("Please ensure the file is not open and you have 'xlsxwriter' installed ('pip install xlsxwriter').")

print("\n🎉 Metadata preparation complete. Review the output Excel file for results.")
# === DONE ===
# This script prepares metadata for feature extraction by matching video titles with tensor files,
# sanitizing titles for robust matching, and saving results in an Excel file with two sheets.