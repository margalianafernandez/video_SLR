import os
from enum import Enum
from os.path import join
from labels import face_motion_labels, hands_motion_labels, body_motion_labels


class ProcessingType(Enum):
    HANDS = "hands"
    FACE_HANDS = "face_and_hands"
    BODY_HANDS = "body_and_hands"
    ALL = "all"


# Parameters
MIN_SAMPLES_LABEL = 8
MAX_SAMPLES_LABEL = 15
NUM_LABELS = 50
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
#VIDEOS_FOLDER = join(CURRENT_PATH, "videos_WLASL")
VIDEOS_FOLDER = join(CURRENT_PATH, "videos_MSASL")
DATASET_FILE = join(CONFIG_PATH, "dataset_MSASL.json")
#DATASET_FILE = join(CONFIG_PATH, "dataset.json")
#WLASL_FILE = join(CONFIG_PATH, "WLASL_v0.3.json")
WLASL_FILE = join(CONFIG_PATH, "MSASL.json")
PROCESSED_VIDEO_FOLDER = os.path.join(CURRENT_PATH, "processed_data")


LABELS = face_motion_labels + hands_motion_labels + body_motion_labels
