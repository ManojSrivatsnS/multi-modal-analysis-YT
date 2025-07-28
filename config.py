from pathlib import Path

CONFIG = {
    "train_dir": Path("data/Ind_Train"),
    "val_dir": Path("data/Ind_Validation"),
    "test_dir": Path("data/Ind_Test"),
    "batch_size": 5,
    "sequence_length": 960,
    "learning_rate": 0.001,
    "num_epochs": 30,
    "device": "cuda",  # or "cpu"
}
def get_config():
    """
    Returns the configuration dictionary.
    
    Returns:
        dict: Configuration settings for the training process.
    """
    return CONFIG