# Import necessary libraries and modules
import os
import torch
from enum import Enum
from os.path import join

# Define an enumeration for different types of models


class Models(Enum):
    SLOWFAST = "slowfast"
    CNN_3D = "3dcnn"

# ENVIRONMENT PARAMETERS


# Check if CUDA is available for GPU acceleration
CUDA_ACTIVATED = torch.cuda.is_available()

# DATA PROCESSING PARAMETERS

# Maximum size for data processing
MAX_SIZE = 368

# Mean and standard deviation values for data normalization
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]

# Size for cropping data
CROP_SIZE = 336

# Number of frames in video clips
NUM_FRAMES = 32

# Video frame sampling rate
SAMPLING_RATE = 1

# Frames per second (FPS) for video clips
FPS = NUM_FRAMES / 5

# Duration of video clips in seconds
CLIP_DURATION = (NUM_FRAMES * SAMPLING_RATE) / FPS

# Number of worker processes for data loading
NUM_WORKERS = 3

# MODEL PARAMETERS

# Batch size for training
BATCH_SIZE = 5

# Current model type (SlowFast or 3DCNN)
CURRENT_MODEL = Models.SLOWFAST

# Number of training epochs
EPOCHS = 30

# Parameters for SlowFast model
MOMENTUM_SLOWFAST = 0.9
WEIGHT_DECAY_SLOWFAST = 0.001
LEARNING_RATE_SLOWFAST = 0.01
SIDE_SIZE_SLOWFAST = 336

# Parameters for 3DCNN model
MOMENTUM_3DCNN = 0.6
WEIGHT_DECAY_3DCNN = 0.01
LEARNING_RATE_3DCNN = 0.01
SIDE_SIZE_3DCNN = 256

# Parameters for MLP (Multi-Layer Perceptron)
MOMENTUM_MLP = 0.9
WEIGHT_DECAY_MLP = 0.01
LEARNING_RATE_MLP = 0.01

# PATHS

# Get the current working directory
CURRENT_PATH = os.getcwd()

# Define the root path and checkpoints path
ROOT_PATH = join("/dcs", "pg22", "u2288875", "Documents", "TFM")
CHECKPOINTS_PATH = join(ROOT_PATH, "check_points")

# Define the folder path for processed video data
PROCESSED_VIDEO_FOLDER = join(CURRENT_PATH, "processed_data")

# Define specific data folder paths for different processing types
PROCESSED_VIDEO_FOLDER_ALL = join(CURRENT_PATH, "data_all")
PROCESSED_VIDEO_FOLDER_FACE_AND_HANDS = join(
    CURRENT_PATH, "data_face_and_hands")
PROCESSED_VIDEO_FOLDER_BODY_AND_HANDS = join(
    CURRENT_PATH, "data_body_and_hands")
