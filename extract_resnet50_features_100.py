import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
import re

# === CONFIG PATHS ===
TENSOR_DIR = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\data\Ind_Train_Tensors")
FEATURE_DIR = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\features\Ind_Train_ResNet50")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

METADATA_PATH = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\Prepared_English_Videos_Metadata - Fixed_Unstratified.xlsx")
UPDATED_METADATA_PATH = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\features\Updated_Metadata_With_Status.xlsx")

# === SANITIZE FUNCTION ===
def sanitize_filename(name, max_length=100):
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    return name[:max_length]

# === LOAD METADATA ===
df = pd.read_excel(METADATA_PATH)
df['Sanitized Filename'] = df['Sanitized Filename'].astype(str).apply(sanitize_filename)
df['Used_In_ResNet50'] = "No"  # Default value

# === SETUP DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# === LOAD RESNET50 BACKBONE ===
resnet50 = models.resnet50(pretrained=True)
resnet50 = nn.Sequential(*list(resnet50.children())[:-2])  # Remove avgpool + fc
resnet50.eval().to(device)

# === LOAD FIRST 100 TENSORS ===
pt_files = list(TENSOR_DIR.glob("*.pt"))[:100]
print(f"🧠 Processing {len(pt_files)} tensors...")

# === FEATURE EXTRACTION ===
for pt_path in tqdm(pt_files, desc="Extracting features"):
    try:
        tensor = torch.load(pt_path).permute(0, 3, 1, 2).to(device)  # (960, 3, 128, 128)
        base_name = pt_path.stem
        sanitized_name = sanitize_filename(base_name)
        feature_path = FEATURE_DIR / f"{sanitized_name}_resnet50.pt"

        # Extract features
        with torch.no_grad():
            features = resnet50(tensor)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            pooled = pooled.view(960, -1).cpu()

        # Save feature tensor
        torch.save(pooled, feature_path)

        # === Mark metadata row as used ===
        matched = df['Sanitized Filename'] == sanitized_name
        if matched.any():
            df.loc[matched, 'Used_In_ResNet50'] = "Yes"
        else:
            print(f"⚠️ Could not match {sanitized_name} in metadata.")

    except Exception as e:
        print(f"❌ Failed on {pt_path.name}: {e}")

# === SAVE UPDATED METADATA ===
df.to_excel(UPDATED_METADATA_PATH, index=False)
print(f"\n✅ Saved updated metadata to:\n{UPDATED_METADATA_PATH}")
print(f"✅ Extracted features saved in:\n{FEATURE_DIR}")
print("✅ Feature extraction complete.")
# === DONE ===
# This script extracts ResNet50 features from the first 100 tensors in the specified directory,
# saves them in a new directory, and updates the metadata to indicate which videos were processed.
# It sanitizes filenames to ensure compatibility with file systems and avoids issues with special characters.
# The script also handles any exceptions that may occur during the feature extraction process,
# ensuring that the script continues running even if some tensors fail to process.
