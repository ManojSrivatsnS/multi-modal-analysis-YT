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
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

# The list of 14 required video names (same as before)
required_names = [
    "10 Min Hip Stretches for Pain  Tight Hips - Stretching for Hip Pain and Sciatica Sitting  Running",
    "10 MIN TONED ARMS - STANDING  burns like fire for all levels just switch size of bottleweight",
    "11 Min Hip Stretches for Pain  Tight Hips - Stretching for Hip Pain and Sciatica Sitting  Running",
    "12 Min Hip Stretches for Pain  Tight Hips - Stretching for Hip Pain and Sciatica Sitting  Running",
    "1300 BC - A Tour Of The Bronze Age Greek World - Mycenaean Greece  History Archaeology Documentary",
    "20 Min Chair Exercises for Seniors Workout at Home - Seated Exercise for Weight Loss - Sitting Down",
    "20 Min Chair Exercises for Seniors Workout at Home - Seated Exercise for Weight Loss - Sitting Down",
    "20 Min Full Body Beginner Workout at Home - Dumbbell Strength Training for Women  Men Over 50",
    "20 Min HIIT Workout for Beginners for Fat Loss - No Jumping No Repeat No Equipment Easy Low Impact",
    "21 Min Full Body Beginner Workout at Home - Dumbbell Strength Training for Women  Men Over 50",
    "21 Min HIIT Workout for Beginners for Fat Loss - No Jumping No Repeat No Equipment Easy Low Impact",
    "22 Min HIIT Workout for Beginners for Fat Loss - No Jumping No Repeat No Equipment Easy Low Impact",
    "30 Min Full Body Dumbbell Workout at Home Strength Training - Weight Training for Weight Loss",
    "a Introducing Aurora Crystal Cut Glass Nails #crazy #nailsnailsnails #nailart #cute #nails #nailart"
]

# === HELPERS ===
def sanitize_filename(name, max_length=100):
    return re.sub(r'[<>:"/\\|?*]', '', name)[:max_length]

def extract_frames_from_video(video_path, target_frames=960):
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
    selected_set = set(frame_indices)
    current_frame = 0
    idx_pointer = 0
    # To speed up, read frame by frame and add frames at needed indices only
    while len(frames) < target_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if current_frame == frame_indices[idx_pointer]:
            frame = cv2.resize(frame, (128, 128))
            frame = frame[:, :, ::-1]  # BGR to RGB
            frames.append(frame)
            idx_pointer += 1
            if idx_pointer >= len(frame_indices):
                break
        current_frame += 1

    cap.release()

    if len(frames) != target_frames:
        raise RuntimeError(f"Expected {target_frames} frames, got {len(frames)}")

    frames = np.stack(frames)
    frames_tensor = torch.tensor(frames, dtype=torch.float32) / 255.0  # Normalize
    return frames_tensor  # shape: (960, 128, 128, 3)

# === MODEL SETUP ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet50 = nn.Sequential(*list(resnet50.children())[:-2])  # Remove avgpool + fc
resnet50.to(device)
resnet50.eval()

# === MAIN PROCESSING LOOP ===
for name in tqdm(required_names, desc="🧠 Extracting Features"):
    sanitized_name = sanitize_filename(name)
    feature_path = FEATURE_DIR / f"{sanitized_name}_resnet50.pt"
    video_path = VIDEO_DIR / f"{sanitized_name}.mp4"

    try:
        if not video_path.exists():
            raise FileNotFoundError(f"❌ Video file not found: {video_path}")

        print(f"🎞️  Processing video: {name}")
        frames_tensor = extract_frames_from_video(video_path)  # (960, 128, 128, 3)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2).to(device)  # (960, 3, 128, 128)

        with torch.no_grad():
            features = resnet50(frames_tensor)  # (960, 2048, 4, 4)
            pooled = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))  # (960, 2048, 1, 1)
            pooled = pooled.view(960, -1).cpu()  # (960, 2048)

        torch.save(pooled, feature_path)

        if (df['Sanitized Filename'] == sanitized_name).any():
            df.loc[df['Sanitized Filename'] == sanitized_name, 'Used_In_ResNet50'] = "Yes"
        else:
            print(f"⚠️ Metadata not matched for: {sanitized_name}")

    except Exception as e:
        print(f"❌ Failed for {sanitized_name}: {e}")
        continue
# === SAVE UPDATED METADATA ===
print("\n✅ All done.")
# This script extracts ResNet50 features from specified videos,
# saves them in a specified directory, and handles errors gracefully.