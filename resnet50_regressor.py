import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# === CONFIG ===
FEATURE_DIR = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\features\Ind_Train_ResNet50")
METADATA_PATH = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\features\Updated_Metadata_With_Extraction_Status.xlsx")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === LOAD METADATA ===
try:
    df = pd.read_excel(METADATA_PATH, sheet_name='Metadata_Status')
    print(f"📖 Loaded metadata from: {METADATA_PATH} (Sheet: Metadata_Status)")
except Exception as e:
    print(f"❌ Error loading metadata from {METADATA_PATH} or 'Metadata_Status' sheet: {e}")
    print("Please ensure the path is correct and the sheet exists.")
    exit()

# Filter for successfully extracted features AND ensure 'Matched_Feature_Filename' is not NaN/empty
df_filtered = df[
    (df["Used_In_ResNet50"] == "Yes") &
    (df["Matched_Feature_Filename"].notna()) &
    (df["Matched_Feature_Filename"] != '')
].copy() # Use .copy() to avoid SettingWithCopyWarning

print(f"✅ Number of videos in metadata after filtering: {len(df_filtered)}")
print("\nFiltered DataFrame Head (used for dataset):")
print(df_filtered[['serial number', 'Video Title', 'Matched_Feature_Filename', 'Views', 'Used_In_ResNet50']].head())

# === DATASET ===
class ResNet50Dataset(Dataset):
    def __init__(self, df_filtered, feature_dir):
        self.data = df_filtered[['serial number', 'Matched_Feature_Filename', 'Views']].reset_index(drop=True)
        self.feature_dir = feature_dir
        # No need to track problematic_samples here, as we pre-filter to create valid_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        feature_filename = str(row["Matched_Feature_Filename"]) 
        path = self.feature_dir / feature_filename

        try:
            features = torch.load(path)  # shape: [960, 2048]
            if features.dim() == 4: 
                features = features.view(features.size(0), -1)
            mean_feat = features.mean(dim=0)  # [2048]
            
            target = torch.tensor(row["Views"], dtype=torch.float32).item() 
            return mean_feat, torch.tensor(target, dtype=torch.float32)

        except FileNotFoundError:
            # If we reach here, it means Matched_Feature_Filename pointed to a non-existent file
            # despite previous filtering. This indicates a deeper issue or recent file changes.
            # For robust prediction, we must return valid data or skip.
            # Given the previous filtering, this should ideally not happen.
            # If it does, we'll return zeros and let the calling code decide to skip.
            print(f"⚠️ Warning: Feature file not found at {path} for serial {row['serial number']}. Returning dummy data.")
            return torch.zeros(2048), torch.tensor(0.0, dtype=torch.float32)
        except Exception as e:
            print(f"❌ Error loading or processing feature file {path} for serial {row['serial number']}: {e}. Returning dummy data.")
            return torch.zeros(2048), torch.tensor(0.0, dtype=torch.float32)

# === MODEL ===
class SimpleRegressor(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze(1)

# === DATASET/DATALOADER SPLIT ===
# We'll create the main dataset and then filter for truly valid samples
full_dataset = ResNet50Dataset(df_filtered, FEATURE_DIR)

valid_data_for_split = []
valid_original_indices_in_df_filtered = [] # Store indices relative to df_filtered
# Also store the `serial number` to correctly map predictions back to the original df
valid_serial_numbers = [] 

for i in tqdm(range(len(full_dataset)), desc="Checking dataset validity"):
    features, target = full_dataset[i]
    # Check if it's not a dummy tensor (meaning feature load was successful)
    if not torch.equal(features, torch.zeros(2048)): 
        valid_data_for_split.append((features, target))
        # Get the original index from df_filtered (which corresponds to `idx` in ResNet50Dataset)
        valid_original_indices_in_df_filtered.append(df_filtered.index[i])
        valid_serial_numbers.append(full_dataset.data.iloc[i]['serial number'])

if len(valid_data_for_split) == 0:
    print("❌ No valid samples found after dataset creation. Cannot proceed with training.")
    exit()

class ValidDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        return self.data_list[idx]

valid_dataset = ValidDataset(valid_data_for_split)


train_size = int(0.8 * len(valid_dataset))
val_size = len(valid_dataset) - train_size

if train_size == 0: 
    print("⚠️ Not enough valid samples for a meaningful training set. Exiting.")
    exit()
elif val_size == 0: 
    print(f"⚠️ Only {len(valid_dataset)} valid samples. Using all for training, no separate validation.")
    train_set = valid_dataset
    val_set = [] # Empty list as no separate validation
else:
    train_set, val_set = torch.utils.data.random_split(valid_dataset, [train_size, val_size])

batch_size = 8
train_batch_size = min(batch_size, len(train_set)) if len(train_set) > 0 else 1
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True)

val_batch_size = min(batch_size, len(val_set)) if len(val_set) > 0 else 1
val_loader = DataLoader(val_set, batch_size=val_batch_size)

print(f"📊 Training set size: {len(train_set)}, Validation set size: {len(val_set)}")


# === TRAINING ===
model = SimpleRegressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

EPOCHS = 20
if len(train_loader) == 0 or len(train_set) == 0:
    print("⚠️ No batches or samples to train on. Skipping training.")
else:
    for epoch in range(EPOCHS):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print(f"Epoch {epoch+1}, Train Loss: {np.mean(train_losses):.4f}")

# === VALIDATION ===
if len(val_loader) == 0 or len(val_set) == 0:
    print("⚠️ No batches or samples to validate on. Skipping validation.")
    val_preds, val_truths = np.array([]), np.array([]) # Renamed to avoid conflict
else:
    model.eval()
    val_preds_list, val_truths_list = [], [] 
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            val_preds_list.append(pred.cpu().numpy())
            val_truths_list.append(y.cpu().numpy())

    val_preds = np.concatenate(val_preds_list)
    val_truths = np.concatenate(val_truths_list)

if len(val_preds) > 0 and len(val_truths) > 0:
    mse = mean_squared_error(val_truths, val_preds)
    r, _ = pearsonr(val_truths, val_preds)
    print(f"✅ Val MSE: {mse:.2f}, Pearson Correlation: {r:.3f}")
else:
    print("❌ Not enough data to calculate validation metrics.")

# === GENERATE PREDICTIONS FOR ALL VALID SAMPLES ===
# Create a DataLoader for the entire valid_dataset
all_preds_loader = DataLoader(valid_dataset, batch_size=val_batch_size) # Use val_batch_size or 8

model.eval()
all_preds_list = []
with torch.no_grad():
    for x, y in all_preds_loader: # Iterate over the full valid_dataset
        x = x.to(device)
        pred = model(x)
        all_preds_list.append(pred.cpu().numpy())

all_preds = np.concatenate(all_preds_list)

# Verify length match before assignment
if len(all_preds) != len(valid_original_indices_in_df_filtered):
    print(f"Fatal Error: Length of all_preds ({len(all_preds)}) does not match length of valid_original_indices_in_df_filtered ({len(valid_original_indices_in_df_filtered)}).")
    exit()

# === SAVE MODEL ===
model_path = FEATURE_DIR / "resnet50_regressor.pth"
torch.save(model.state_dict(), model_path)
print(f"✅ Model saved to {model_path}")

# === SAVE UPDATED METADATA WITH PREDICTIONS ===
# Re-load the original, unfiltered DataFrame to ensure all rows are present for update
df_full_original = pd.read_excel(METADATA_PATH, sheet_name='Metadata_Status')

# Initialize 'Predicted_Views' column in the full DataFrame with NaN
df_full_original['Predicted_Views'] = np.nan

# Get the actual index labels from the original DataFrame for the filtered rows
# These are the indices that the 'valid_original_indices_in_df_filtered' map to in df_filtered
# and correspond to the rows in df_full_original
actual_original_indices_in_full_df = df_filtered.index[valid_original_indices_in_df_filtered]


# Create a temporary Series with predictions aligned to these specific original DataFrame indices
# This is the correct way to ensure alignment
print(f"🔎 Predictions: {len(all_preds.flatten())}, Indices: {len(actual_original_indices_in_full_df)}")
temp_preds_series = pd.Series(all_preds.flatten(), index=actual_original_indices_in_full_df)

# Use .update() method of pandas Series to assign values to corresponding indices
df_full_original['Predicted_Views'].update(temp_preds_series)

output_metadata_path = FEATURE_DIR / "Updated_Metadata_With_Predictions.xlsx"
df_full_original.to_excel(output_metadata_path, index=False)
print(f"✅ Updated metadata (including all original rows and predictions) saved to {output_metadata_path}")

# Use the filtered DataFrame with predictions
valid_df = df_full_original.loc[actual_original_indices_in_full_df].copy()

# Ensure we drop any rows where prediction or actual view is missing
valid_df = valid_df.dropna(subset=["Predicted_Views", "Views"])

actual = valid_df["Views"].values
predicted = valid_df["Predicted_Views"].values

# === 1. Scatter Plot: Actual vs Predicted Views ===
plt.figure(figsize=(8, 6))
sns.scatterplot(x=actual, y=predicted, alpha=0.7, color="dodgerblue")
plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')  # y = x line
plt.xlabel("Actual View Count")
plt.ylabel("Predicted View Count")
plt.title("ResNet50 Regression: Actual vs. Predicted Views")
plt.grid(True)
plt.tight_layout()
plt.show()

# === 2. Residual Plot (Optional) ===
residuals = predicted - actual
plt.figure(figsize=(8, 6))
sns.histplot(residuals, bins=30, kde=True, color="darkorange")
plt.title("Residuals (Predicted - Actual Views)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
