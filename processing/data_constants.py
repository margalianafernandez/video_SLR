import os
from enum import Enum
from os.path import join
from processing.labels import face_motion_labels, hands_motion_labels, body_motion_labels


class DatasetSelected(Enum):
    WLASL = "wlasl"
    MSASL = "msasl"

class ProcessingType(Enum):
    HANDS = "hands"
    FACE_HANDS = "face_and_hands"
    BODY_HANDS = "body_and_hands"
    ALL = "all"


DATASET_SELECTED = DatasetSelected.MSASL

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

SETS = [TRAIN, TEST, VALIDATION]

# Paths
CURRENT_PATH = os.getcwd()
CONFIG_PATH = join(CURRENT_PATH, "config")
VIDEOS_FOLDER = join(CURRENT_PATH, "videos_{}".format(DATASET_SELECTED.value))
DATASET_FILE = join(CONFIG_PATH, "dataset_{}.json".format(DATASET_SELECTED.value))
PROCESSED_VIDEO_FOLDER = os.path.join(CURRENT_PATH, "processed_data")

if DATASET_SELECTED == DatasetSelected.WLASL:
    DATASET_ORIGINAL_FILE = join(CONFIG_PATH, "WLASL_v0.3.json")
else:
    DATASET_ORIGINAL_FILE = join(CONFIG_PATH, "MSASL.json")


LABELS = face_motion_labels + hands_motion_labels + body_motion_labels
