import datetime
from enum import Enum
from os.path import join


# MODELS
class Models(Enum):
    SLOWFAST = "slowfast"
    CNN_3D = "3dcnn"


CURRENT_MODEL = Models.SLOWFAST

# DATA PROCESSING PARAMETERS

SIDE_SIZE = 336
MAX_SIZE = 368
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]
CROP_SIZE = 336
NUM_FRAMES = 32
SAMPLING_RATE = 1
FPS = 32/5
CLIP_DURATION = (NUM_FRAMES * SAMPLING_RATE) / FPS


# MODEL PARAMETERS

EPOCHS = 50
MOMENTUM = 0.9
WEIGHT_DECAY=0.001
BATCH_SIZE = 5 # 10
LEARNING_RATE = 0.02


# PATHS

ROOT_PATH = join("/dcs", "pg22", "u2288875", "Documents", "TFM")
DATA_PATH = join(ROOT_PATH, "processed_data")
CHECKPOINTS_PATH = join(ROOT_PATH, "check_points") 


current_datetime = datetime.datetime.now()
current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
