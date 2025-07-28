import pandas as pd
from pathlib import Path
import re
import torch

# === CONFIGURATION ===
FEATURE_DIR = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\features\Ind_Train_ResNet50")
METADATA_PATH = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\Prepared_English_Videos_Metadata - Fixed_Unstratified.xlsx")
OUTPUT_METADATA_PATH = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\features\Updated_Metadata_With_Extraction_Status.xlsx")

# === SANITIZATION FUNCTION ===
def sanitize_for_matching(text, max_len_for_comparison=50):
    """
    Sanitizes text for matching filenames.
    Converts to lowercase, replaces spaces with underscores, removes most special characters,
    and truncates to a specified length.
    """
    if pd.isna(text):
        return ''
    
    text = str(text)

    text = text.lower()
    text = re.sub(r'\s+', '_', text) 
    text = re.sub(r'[^a-z0-9_]', '', text) # Remove any non-alphanumeric/underscore characters
    
    if max_len_for_comparison is not None and len(text) > max_len_for_comparison:
        text = text[:max_len_for_comparison]
        
    return text.strip()

# === MAIN SCRIPT LOGIC ===

print("🚀 Starting feature file matching process...")

# --- 1. Load Metadata ---
try:
    df = pd.read_excel(METADATA_PATH, sheet_name='Ind_Train')
    print(f"📖 Loaded metadata from: {METADATA_PATH} (Sheet: Ind_Train)")
    print(f"Initial metadata entries: {len(df)}")
except Exception as e:
    print(f"❌ Error loading metadata from {METADATA_PATH}: {e}")
    print("Please ensure the path is correct and 'Ind_Train' sheet exists.")
    exit()

# --- 2. Sanitize Video Titles in Metadata ---
# Use the best `max_len_for_comparison` you found that yielded 53 matches.
# If no value improved beyond 0, maybe try a smaller, more aggressive truncation like 20 or 30.
CURRENT_MAX_LEN = 50 # Set this to the value that gave you 53 matches, or experiment if unsure.
df['Sanitized_Video_Title'] = df['Video Title'].apply(
    lambda x: sanitize_for_matching(x, max_len_for_comparison=CURRENT_MAX_LEN)
)
print(f"✅ Sanitized 'Video Title' column for matching with max_len_for_comparison={CURRENT_MAX_LEN}.")


# --- 3. Build an Index of Available Feature Files and prepare for 2nd sheet ---
feature_files = list(FEATURE_DIR.glob("*.pt"))
extracted_feature_map = {} # sanitized_filename -> full_filename
feature_file_list_for_sheet = [] # For the second sheet

if not feature_files:
    print(f"⚠️ No .pt files found in {FEATURE_DIR}. Please check the directory.")
else:
    try:
        sample_tensor = torch.load(feature_files[0])
        print(f"📐 Sample feature shape from {feature_files[0].name}: {sample_tensor.shape}")
        print(f"📦 Sample feature dtype: {sample_tensor.dtype}")
    except Exception as e:
        print(f"❌ Could not load a sample tensor from {feature_files[0].name}: {e}")

    for file_path in feature_files:
        sanitized_stem = sanitize_for_matching(file_path.stem, max_len_for_comparison=CURRENT_MAX_LEN)
        extracted_feature_map[sanitized_stem] = file_path.name
        
        feature_file_list_for_sheet.append({
            'Feature_Filename': file_path.name,
            'Sanitized_Feature_Name': sanitized_stem
        })

    print(f"🔎 Indexed {len(extracted_feature_map)} unique sanitized feature file names.")

# Create the DataFrame for the second sheet
df_features = pd.DataFrame(feature_file_list_for_sheet)

# --- 4. Match Metadata Entries with Feature Files ---
df['Used_In_ResNet50'] = 'No' 
df['Matched_Feature_Filename'] = '' 
matched_stems_count = 0 

for index, row in df.iterrows():
    sanitized_title = row['Sanitized_Video_Title']
    if sanitized_title in extracted_feature_map:
        df.at[index, 'Used_In_ResNet50'] = 'Yes'
        df.at[index, 'Matched_Feature_Filename'] = extracted_feature_map[sanitized_title]
        matched_stems_count += 1

print("📊 Matching process complete.")

# --- 5. Report and Save Results to Multiple Sheets ---
matched_count = df['Used_In_ResNet50'].value_counts().get('Yes', 0)
unmatched_count = df['Used_In_ResNet50'].value_counts().get('No', 0)

print(f"\n✅ Matched {matched_count} video titles with existing feature files.")
print(f"❌ {unmatched_count} video titles did not find a matching feature file.")
print(f"Total .pt files found in folder: {len(feature_files)}")
print(f"Number of unique feature files successfully mapped to metadata entries: {matched_stems_count}")

# Save to Excel with multiple sheets
try:
    with pd.ExcelWriter(OUTPUT_METADATA_PATH, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Metadata_Status', index=False)
        df_features.to_excel(writer, sheet_name='Feature_Files_List', index=False)
    print(f"📝 Updated metadata and feature list saved to: {OUTPUT_METADATA_PATH} in separate sheets.")
except Exception as e:
    print(f"❌ Error saving Excel file to {OUTPUT_METADATA_PATH}: {e}")
    print("Please ensure the file is not open and you have write permissions.")


print("\n🎉 Feature matching complete. Check the output Excel file for results.")