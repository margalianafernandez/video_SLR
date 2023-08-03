
import os
from os.path import join


# Parameters
NUM_LABELS = 20
TARGET_SIZE = 256
TARGET_CHANNELS = 3
LEFT_HAND_LABEL = "Left"

# Folders
TEST = "test"
TRAIN = "train"
VALIDATION = "val"
FILES_EXTENSION = ".mp4"

CURRENT_PATH = os.getcwd()
CONFIG_PATH = join(CURRENT_PATH, "config")
VIDEOS_FOLDER = join(CURRENT_PATH, "videos")
DATASET_FILE = join(CONFIG_PATH, "dataset.json")
WLASL_FILE = join(CONFIG_PATH, "WLASL_v0.3.json")
PROCESSED_VIDEO_FOLDER = os.path.join(CURRENT_PATH, "processed_data")
