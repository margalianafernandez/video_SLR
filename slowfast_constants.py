from os.path import join
from data_constants import NUM_LABELS

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

BATCH_SIZE = 5 # 10
EPOCHS = 50
LEARNING_RATE = 0.02
MOMENTUM = 0.9
WEIGHT_DECAY=0.001


# PATHS
  
#checkpoint_path = '/kaggle/working/SLOWFAST_8x8_R50.pyth'

ROOT_PATH = join("/dcs", "pg22", "u2288875", "Documents", "TFM")
DATA_PATH = join(ROOT_PATH, "processed_data")
CHECKPOINTS_PATH = join(ROOT_PATH, "check_points") 
