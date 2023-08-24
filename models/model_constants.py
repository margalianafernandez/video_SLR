import os
import torch
from enum import Enum
from os.path import join

# Type of models

class Models(Enum):
    SLOWFAST = "slowfast"
    CNN_3D = "3dcnn"


# ENVIRONMMENT PARAMETERS
CUDA_ACTIVATED = torch.cuda.is_available()

# DATA PROCESSING PARAMMETERS
MAX_SIZE = 368
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]
CROP_SIZE = 336
NUM_FRAMES = 32
SAMPLING_RATE = 1
FPS = NUM_FRAMES/5
CLIP_DURATION = (NUM_FRAMES * SAMPLING_RATE) / FPS
NUM_WORKERS = 3

# MODEL PARAMETERS
CURRENT_MODEL = Models.SLOWFAST
BATCH_SIZE = 5 
SIDE_SIZE =  336
EPOCHS = 30
MOMENTUM = 0.3
WEIGHT_DECAY = 0.01
BATCH_SIZE = 5
LEARNING_RATE = 0.1

# PATHS
CURRENT_PATH = os.getcwd()
ROOT_PATH = join("/dcs", "pg22", "u2288875", "Documents", "TFM")
CHECKPOINTS_PATH = join(ROOT_PATH, "check_points")
PROCESSED_VIDEO_FOLDER = join(CURRENT_PATH, "processed_data")
