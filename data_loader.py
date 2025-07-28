import torch
from torch.utils.data import Dataset
import os
from pathlib import Path

class VideoFrameDataset(Dataset):
    def __init__(self, root_dir, labels_dict):
        self.root_dir = Path(root_dir)
        self.files = sorted(list(self.root_dir.glob("*.pt")))
        self.labels = labels_dict  # Dict: {filename: view_count}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        x = torch.load(file)  # shape: [960, 3, 128, 128]
        video_id = file.stem
        y = self.labels.get(video_id, 0.0)  # Regression label
        return x.float(), torch.tensor(y, dtype=torch.float32)

    def get_video_id(self, idx):    
        file = self.files[idx]
        return file.stem

    def get_view_count(self, idx):
        video_id = self.get_video_id(idx)
        return self.labels.get(video_id, 0.0)   
    
    def get_all_video_ids(self):
        return [file.stem for file in self.files]   
    
    def get_all_view_counts(self):
        return [self.labels.get(file.stem, 0.0) for file in self.files]
    
def create_labels_dict(labels_file):
    labels_dict = {}
    with open(labels_file, 'r') as f:
        for line in f:
            video_id, view_count = line.strip().split(',')
            labels_dict[video_id] = float(view_count)
    return labels_dict
    
def load_dataset(root_dir, labels_file):
    labels_dict = VideoFrameDataset.create_labels_dict(labels_file)
    return VideoFrameDataset(root_dir, labels_dict)

def create_dataset(root_dir, labels_file):
    labels_dict = create_labels_dict(labels_file)
    return VideoFrameDataset(root_dir, labels_dict)

def get_dataset_info(dataset):
    num_videos = len(dataset)
    video_ids = dataset.get_all_video_ids()
    view_counts = dataset.get_all_view_counts()
    return {
        "num_videos": num_videos,
        "video_ids": video_ids,
        "view_counts": view_counts
    }

def get_video_info(dataset, idx):
    video_id = dataset.get_video_id(idx)
    view_count = dataset.get_view_count(idx)
    return {
        "video_id": video_id,
        "view_count": view_count
    }

def get_video_tensor(dataset, idx):
    video_tensor, view_count = dataset[idx]
    return {
        "video_tensor": video_tensor,
        "view_count": view_count
    }

def get_all_video_tensors(dataset):
    video_tensors = []
    view_counts = []
    for idx in range(len(dataset)):
        video_tensor, view_count = dataset[idx]
        video_tensors.append(video_tensor)
        view_counts.append(view_count)
    return {
        "video_tensors": video_tensors,
        "view_counts": view_counts
    }

def get_video_tensor_by_id(dataset, video_id):
    for idx in range(len(dataset)):
        if dataset.get_video_id(idx) == video_id:
            return dataset[idx]
    raise ValueError(f"Video ID {video_id} not found in dataset.")

def get_view_count_by_id(dataset, video_id):
    for idx in range(len(dataset)):
        if dataset.get_video_id(idx) == video_id:
            return dataset.get_view_count(idx)
    raise ValueError(f"Video ID {video_id} not found in dataset.")

def get_all_video_ids(dataset):
    return dataset.get_all_video_ids()

def get_all_view_counts(dataset):
    return dataset.get_all_view_counts()

def get_video_ids_and_view_counts(dataset):
    video_ids = dataset.get_all_video_ids()
    view_counts = dataset.get_all_view_counts()
    return list(zip(video_ids, view_counts))

def get_video_ids_and_tensors(dataset):
    video_ids = dataset.get_all_video_ids()
    video_tensors = [dataset[idx][0] for idx in range(len(dataset))]
    return list(zip(video_ids, video_tensors))

def get_video_ids_and_view_counts_as_dict(dataset):
    video_ids = dataset.get_all_video_ids()
    view_counts = dataset.get_all_view_counts()
    return {video_id: view_count for video_id, view_count in zip(video_ids, view_counts)}

def get_video_tensors_as_dict(dataset):
    video_ids = dataset.get_all_video_ids()
    video_tensors = [dataset[idx][0] for idx in range(len(dataset))]
    return {video_id: video_tensor for video_id, video_tensor in zip(video_ids, video_tensors)}

def get_view_counts_as_dict(dataset):
    video_ids = dataset.get_all_video_ids()
    view_counts = dataset.get_all_view_counts()
    return {video_id: view_count for video_id, view_count in zip(video_ids, view_counts)}   

def get_video_tensor_and_view_count_as_dict(dataset, video_id):
    for idx in range(len(dataset)):
        if dataset.get_video_id(idx) == video_id:
            video_tensor, view_count = dataset[idx]
            return {video_id: {"video_tensor": video_tensor, "view_count": view_count}}
    raise ValueError(f"Video ID {video_id} not found in dataset.")

def get_all_video_tensors_as_dict(dataset):
    video_ids = dataset.get_all_video_ids()
    video_tensors = [dataset[idx][0] for idx in range(len(dataset))]
    return {video_id: video_tensor for video_id, video_tensor in zip(video_ids, video_tensors)}
