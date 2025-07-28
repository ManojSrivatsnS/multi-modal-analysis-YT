import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import numpy as np

# === CONFIG ===
TENSOR_DIR = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\data\Ind_Train_Tensors")
FEATURE_DIR = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\features\Ind_Train_ResNet50")
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# === Load pretrained ResNet50 ===
resnet50 = models.resnet50(pretrained=True)
resnet50 = nn.Sequential(*list(resnet50.children())[:-2])  # remove avgpool + fc

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)
resnet50.to(device)
resnet50.eval().to(device)

# Freeze weights
for param in resnet50.parameters():
    param.requires_grad = False

# === Process each tensor file ===
pt_files = list(TENSOR_DIR.glob("*.pt"))[:100]  # Limit to first 100 for testing
print(f"Found {len(pt_files)} video tensors.")

for pt_path in tqdm(pt_files, desc="Extracting features"):
    video_tensor = torch.load(pt_path)  # [960, 128, 128, 3]
    
    # Rearrange to [960, 3, 128, 128]
    video_tensor = video_tensor.permute(0, 3, 1, 2)

    # Send to device
    video_tensor = video_tensor.to(device)

    with torch.no_grad():
        # Feature maps: [960, 2048, 4, 4]
        feats = resnet50(video_tensor)
        
        # Global average pooling → [960, 2048]
        pooled_feats = F.adaptive_avg_pool2d(feats, (1, 1)).squeeze(-1).squeeze(-1)

    # Move to CPU and save
    pooled_feats = pooled_feats.cpu()
    
    # Save
    save_path = FEATURE_DIR / pt_path.name
    torch.save(pooled_feats, save_path)
    print(f"Saved features to {save_path}")
# === Done ===
print("Feature extraction complete.")