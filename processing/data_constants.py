import os
from enum import Enum
from os.path import join


class ProcessingType(Enum):
    HANDS = "hands"
    FACE_HANDS = "face_and_hands"
    BODY_HANDS = "body_and_hands"
    ALL = "all"


# Parameters
NUM_LABELS = 20
TARGET_SIZE = 256
FRAME_SIZE = TARGET_SIZE*2
TARGET_CHANNELS = 3
LEFT_HAND_LABEL = "Left"
MAX_NUM_HANDS = 2
MIN_DETECTION_COFIDENCE = 0.5
MIN_TRACKING_COFIDENCE = 0.5

# Folders
TEST = "test"
TRAIN = "train"
VALIDATION = "val"
FILES_EXTENSION = ".mp4"

# Paths
CURRENT_PATH = os.getcwd()
CONFIG_PATH = join(CURRENT_PATH, "config")
VIDEOS_FOLDER = join(CURRENT_PATH, "videos")
DATASET_FILE = join(CONFIG_PATH, "dataset.json")
WLASL_FILE = join(CONFIG_PATH, "WLASL_v0.3.json")
PROCESSED_VIDEO_FOLDER = os.path.join(CURRENT_PATH, "processed_data")
