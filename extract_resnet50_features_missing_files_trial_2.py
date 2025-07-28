import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import re
from tqdm import tqdm
from pathlib import Path
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# === CONFIG ===
VIDEO_DIR = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets\Ind_Train")
FEATURE_DIR = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\features\Ind_Train_ResNet50")
FEATURE_DIR.mkdir(parents=True, exist_ok=True) # Ensure the directory exists

# The list of 14 required video names (MAKE SURE THESE EXACTLY MATCH YOUR FILE NAMES ON DISK, INCLUDING SPACES/SPECIAL CHARS)
required_names = [
    "10 Min Hip Stretches for Pain  Tight Hips - Stretching for Hip Pain and Sciatica Sitting  Running",
    "10 MIN TONED ARMS - STANDING  burns like fire for all levels just switch size of bottleweight",
    "11 Min Hip Stretches for Pain  Tight Hips - Stretching for Hip Pain and Sciatica Sitting  Running",
    "12 Min Hip Stretches for Pain  Tight Hips - Stretching for Hip Pain and Sciatica Sitting  Running",
    "1300 BC - A Tour Of The Bronze Age Greek World - Mycenaean Greece  History Archaeology Documentary",
    "20 Min Chair Exercises for Seniors Workout at Home - Seated Exercise for Weight Loss - Sitting Down",
    "20 Min Chair Exercises for Seniors Workout at Home - Seated Exercise for Weight Loss - Sitting Down", # Duplicate name, if these are actually distinct files, rename one on disk.
    "20 Min Full Body Beginner Workout at Home - Dumbbell Strength Training for Women  Men Over 50",
    "20 Min HIIT Workout for Beginners for Fat Loss - No Jumping No Repeat No Equipment Easy Low Impact",
    "21 Min Full Body Beginner Workout at Home - Dumbbell Strength Training for Women  Men Over 50",
    "21 Min HIIT Workout for Beginners for Fat Loss - No Jumping No Repeat No Equipment Easy Low Impact",
    "22 Min HIIT Workout for Beginners for Fat Loss - No Jumping No Rep.eat No Equipment Easy Low Impact",
    "30 Min Full Body Dumbbell Workout at Home Strength Training - Weight Training for Weight Loss",
    "a Introducing Aurora Crystal Cut Glass Nails #crazy #nailsnailsnails #nailart #cute #nails #nailart"
]

METADATA_PATH = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\Prepared_English_Videos_Metadata - Fixed_Unstratified.xlsx")
UPDATED_METADATA_PATH = Path(r"C:\Users\manoj\Downloads\University UK docs\University UK docs\classes\PRoject dissertation\downloaded_videos\ind_sets_tensors\features\Updated_Metadata_With_Status.xlsx")


# === HELPERS ===
# We still keep this for the metadata's 'Sanitized Filename' column if needed,
# but it won't be used for the actual feature file names anymore.
def sanitize_filename_for_metadata(name, max_length=100):
    """
    Sanitizes a string for use in the metadata's 'Sanitized Filename' column.
    """
    sanitized = re.sub(r'[<>:"/\\|?*]', '', name)
    sanitized = sanitized.replace(' ', '_')
    sanitized = sanitized.replace('.', '')
    sanitized = sanitized.replace('#', '')
    sanitized = sanitized.replace('-', '_')
    sanitized = sanitized.replace('__', '_')
    return sanitized[:max_length].strip()


def extract_frames_from_video(video_path, target_frames=960):
    """
    Extracts frames from a video, ensuring a consistent number of frames.
    If the video has fewer frames than target_frames, frames are repeated.
    If it has more, frames are randomly sampled.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError("Empty or corrupted video file.")

    # Create indices to extract: random sample if enough frames,
    # else repeat frames until reaching target_frames
    if total_frames < target_frames:
        repeat = int(np.ceil(target_frames / total_frames))
        frame_indices = (list(range(total_frames)) * repeat)[:target_frames]
    else:
        frame_indices = sorted(np.random.choice(range(total_frames), target_frames, replace=False))

    frames = []
    current_frame = 0
    idx_pointer = 0

    while len(frames) < target_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break # No more frames to read
        if current_frame == frame_indices[idx_pointer]:
            frame = cv2.resize(frame, (128, 128))
            frame = frame[:, :, ::-1]  # BGR to RGB
            frames.append(frame)
            idx_pointer += 1
            if idx_pointer >= len(frame_indices):
                break # All target frames collected
        current_frame += 1

    cap.release()

    if len(frames) != target_frames:
        raise RuntimeError(f"Expected {target_frames} frames, but extracted {len(frames)} from {video_path.name}. Video might be truncated or problematic.")

    frames = np.stack(frames)
    frames_tensor = torch.tensor(frames, dtype=torch.float32) / 255.0  # Normalize
    return frames_tensor  # shape: (960, 128, 128, 3)

# === LOAD METADATA ===
try:
    df = pd.read_excel(METADATA_PATH)
    # Ensure 'Original Filename' column exists for mapping
    if 'Original Filename' not in df.columns:
        df['Original Filename'] = df['Video Title'].astype(str) # Assuming 'Video Filename' holds original names
    # Ensure 'Sanitized Filename' column exists for consistency, but it's not used for saving feature files now
    if 'Sanitized Filename' not in df.columns:
        df['Sanitized Filename'] = df['Original Filename'].apply(sanitize_filename_for_metadata)
    else:
        df['Sanitized Filename'] = df['Sanitized Filename'].astype(str).apply(sanitize_filename_for_metadata)

    df['Feature_File_ID'] = pd.NA # New column to store the numerical ID
    df['Used_In_ResNet50'] = "No" # Initialize column for tracking
except FileNotFoundError:
    print(f"Metadata file not found at: {METADATA_PATH}. Creating a dummy DataFrame for tracking.")
    df = pd.DataFrame(columns=['Original Filename', 'Sanitized Filename', 'Feature_File_ID', 'Used_In_ResNet50'])
    # Add rows for the required names
    for name in required_names:
        sanitized_name = sanitize_filename_for_metadata(name)
        if not (df['Original Filename'] == name).any(): # Check against original name
            df.loc[len(df)] = {'Original Filename': name, 'Sanitized Filename': sanitized_name, 'Feature_File_ID': pd.NA, 'Used_In_ResNet50': "No"}


# === MODEL SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet50 = nn.Sequential(*list(resnet50.children())[:-2])  # Remove avgpool + fc
resnet50.to(device)
resnet50.eval()

# === MAIN PROCESSING LOOP ===
processed_count = 0
# Initialize an ID counter for numerical feature filenames
feature_id_counter = 1

for name in tqdm(required_names, desc="🧠 Extracting Features"):
    # Construct video_path using the ORIGINAL name, as it exists on disk
    video_path = VIDEO_DIR / f"{name}.mp4"

    # Determine the numerical ID for the feature file
    # First, try to find an existing ID if this video name was already processed in a previous run
    current_feature_id = df.loc[df['Original Filename'] == name, 'Feature_File_ID'].iloc[0] if (df['Original Filename'] == name).any() and not pd.isna(df.loc[df['Original Filename'] == name, 'Feature_File_ID'].iloc[0]) else None

    if current_feature_id is None:
        # If no existing ID, assign a new one
        current_feature_id = feature_id_counter
        feature_id_counter += 1

    feature_path = FEATURE_DIR / f"{current_feature_id}.pt"

    if video_path.exists():
        print(f"🎞️  Processing video: {name}")
        try:
            frames_tensor = extract_frames_from_video(video_path)  # (960, 128, 128, 3)
            frames_tensor = frames_tensor.permute(0, 3, 1, 2).to(device)  # (960, 3, 128, 128)

            with torch.no_grad():
                features = resnet50(frames_tensor)  # (960, 2048, 4, 4)
                pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                pooled = pooled.view(960, -1).cpu()  # (960, 2048)

            torch.save(pooled, feature_path)
            print(f"✅ Features saved for: {name} to {feature_path}")
            processed_count += 1

            # Update metadata DataFrame
            if (df['Original Filename'] == name).any():
                df.loc[df['Original Filename'] == name, 'Feature_File_ID'] = current_feature_id
                df.loc[df['Original Filename'] == name, 'Used_In_ResNet50'] = "Yes"
            else:
                # This block should ideally not be hit if required_names are truly from your metadata
                sanitized_for_meta = sanitize_filename_for_metadata(name)
                new_row = pd.DataFrame([{'Original Filename': name, 'Sanitized Filename': sanitized_for_meta, 'Feature_File_ID': current_feature_id, 'Used_In_ResNet50': "Yes"}])
                df = pd.concat([df, new_row], ignore_index=True)
                print(f"⚠️ Added new entry to metadata for: {name}")

        except Exception as e:
            print(f"❌ Failed to process {name}: {e}")
            # Ensure status is 'No' if processing failed
            if (df['Original Filename'] == name).any():
                df.loc[df['Original Filename'] == name, 'Used_In_ResNet50'] = "No"
                # Keep Feature_File_ID as NA or previous if failure
            else:
                sanitized_for_meta = sanitize_filename_for_metadata(name)
                new_row = pd.DataFrame([{'Original Filename': name, 'Sanitized Filename': sanitized_for_meta, 'Feature_File_ID': pd.NA, 'Used_In_ResNet50': "No"}])
                df = pd.concat([df, new_row], ignore_index=True)

    else:
        print(f"❌ Video file not found: {video_path}")
        # Mark as 'No' if video file doesn't exist
        if (df['Original Filename'] == name).any():
            df.loc[df['Original Filename'] == name, 'Used_In_ResNet50'] = "No"
            df.loc[df['Original Filename'] == name, 'Feature_File_ID'] = pd.NA # No feature file created
        else:
            sanitized_for_meta = sanitize_filename_for_metadata(name)
            new_row = pd.DataFrame([{'Original Filename': name, 'Sanitized Filename': sanitized_for_meta, 'Feature_File_ID': pd.NA, 'Used_In_ResNet50': "No"}])
            df = pd.concat([df, new_row], ignore_index=True)


# === SAVE UPDATED METADATA ===
OUTPUT_METADATA_PATH = FEATURE_DIR / "Updated_Metadata_With_Status.xlsx"
df.to_excel(OUTPUT_METADATA_PATH, index=False)
print(f"\n✅ Processed {processed_count} out of {len(required_names)} videos.")
print(f"✅ Features saved to: {FEATURE_DIR}")
print(f"✅ Metadata updated and saved to: {OUTPUT_METADATA_PATH}")
print("\n✅ All done.")