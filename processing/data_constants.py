import os
from enum import Enum
from os.path import join

# Import labels from appropriate source
try:
    from processing.labels import face_motion_labels, hands_motion_labels, body_motion_labels
except:
    from labels import face_motion_labels, hands_motion_labels, body_motion_labels

# Enumerations for selected datasets and processing types
class DatasetSelected(Enum):
    WLASL = "wlasl"
    MSASL = "msasl"

class ProcessingType(Enum):
    HANDS = "hands"
    FACE_HANDS = "face_and_hands"
    BODY_HANDS = "body_and_hands"
    ALL = "all"

# Select the dataset (MSASL or WLASL)
DATASET_SELECTED = DatasetSelected.MSASL

# Define split rates for train, validation, and test sets
TRAIN_RATE = 0.6
VALIDATION_RATE = 0.2
TEST_RATE = 0.2

# Labels
# Combine labels for face, hands, and body motion
LABELS = face_motion_labels + hands_motion_labels + body_motion_labels
NUM_LABELS = len(LABELS)

# Parameters
MIN_SAMPLES_LABEL = 8
MAX_SAMPLES_LABEL = 15
TARGET_SIZE = 256
FRAME_SIZE = TARGET_SIZE * 2
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
START_PROCESSED_VIDEO_FOLDER =  os.path.join(CURRENT_PATH, "processed_data_hands_{}")
PROCESSED_VIDEO_FOLDER = START_PROCESSED_VIDEO_FOLDER.format(DATASET_SELECTED.value)

# Define the path for the original dataset file based on the selected dataset
if DATASET_SELECTED == DatasetSelected.WLASL:
    DATASET_ORIGINAL_FILE = join(CONFIG_PATH, "WLASL_v0.3.json")
else:
    DATASET_ORIGINAL_FILE = join(CONFIG_PATH, "MSASL.json")

# Define paths for dataset splitting and joining
START_SPLIT_DATASET_FILE_PATH = join(CONFIG_PATH, "{}_test_val_dataset_split.json")
SPLIT_DATASET_FILE_PATH = START_SPLIT_DATASET_FILE_PATH.format(DATASET_SELECTED.value)
JOIN_DATASET_FILE_PATH = join(CONFIG_PATH, "join_dataset_split.json")
