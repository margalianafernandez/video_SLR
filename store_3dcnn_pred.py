import os
import torch
torch.cuda.empty_cache()
from os.path import  join
import math
import random
import numpy as np
from tqdm import tqdm
from models.ensemble import *
from models.model_constants import *
from processing.data_constants import ProcessingType, CONFIG_PATH


PROCESSING_TYPE = ProcessingType.ALL

def get_prev_models(prev_model_type):
    # Load the pre-trained 3DCNN models
    
    if prev_model_type == Models.SLOWFAST:
        model_1_file = "check_points_all__all_opt/slowfast_model_2023-08-30_12:14:49.pth"
        model_2_file = "check_points_bah_opt/slowfast_model_2023-08-30_21:41:28.pth"
        model_3_file = "check_points_fah_opt/slowfast_model_2023-08-31_00:03:33.pth"

    else:
        model_1_file = "models_3dcnn/3dcnn_model_all.pth"
        model_2_file = "models_3dcnn/3dcnn_model_body_and_hands.pth"
        model_3_file = "models_3dcnn/3dcnn_model_face_and_hands.pth"

    
    model_1 = torch.load(join(ROOT_PATH, model_1_file), map_location=torch.device('cpu'))
    model_2 = torch.load(join(ROOT_PATH, model_2_file), map_location=torch.device('cpu'))
    model_3 = torch.load(join(ROOT_PATH, model_3_file), map_location=torch.device('cpu'))
    
    return [model_1, model_2, model_3]


def enable_cuda_launch_blocking():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # for reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)


def get_epoch(model, loader):

    num_videos = loader[PROCESSING_TYPE].dataset.num_videos
    total = math.ceil(num_videos / BATCH_SIZE)
    val_bars = tqdm(loader[PROCESSING_TYPE], total=total, dynamic_ncols=False)

    total_predictions = []
    total_labels = []
    
    # Iterate using a common index
    for batch in val_bars:
        
        outputs_3dcnn = []
        total_labels += [batch['label']]

        out = model(batch['video'])
        outputs_3dcnn += [out]

        outputs_3dcnn = torch.stack(outputs_3dcnn, dim=1)
        outputs_3dcnn = outputs_3dcnn.reshape(outputs_3dcnn.shape[0], outputs_3dcnn.shape[1]*outputs_3dcnn.shape[2]) 
        total_predictions += [outputs_3dcnn]

    return torch.stack(total_predictions), torch.stack(total_labels)


def store_ensembled_predictions(train_loader, val_loader, test_loader, models_type, epochs=EPOCHS):

    
    prev_models = get_prev_models(models_type)
    
    total_train_pred, total_train_labels = [], []
    total_val_pred, total_val_labels = [], []
    total_test_pred, total_test_labels = [], []

    for epoch in range(1, epochs + 1):
        
        print("Epoch:", epoch)

        train_pred, train_labels = get_epoch(prev_models, train_loader)
        val_pred, val_labels = get_epoch(prev_models, val_loader)
        test_pred, test_labels = get_epoch(prev_models, test_loader)

        total_train_pred += [train_pred]
        total_val_pred += [val_pred]
        total_test_pred += [test_pred]

        total_train_labels += [train_labels]
        total_val_labels += [val_labels]
        total_test_labels += [test_labels]

    total_train_pred = torch.stack(total_train_pred)
    total_val_pred = torch.stack(total_val_pred)
    total_test_pred = torch.stack(total_test_pred)
    total_train_labels = torch.stack(total_train_labels)
    total_val_labels = torch.stack(total_val_labels)
    total_test_labels = torch.stack(total_test_labels)

    torch.save(total_train_pred, 'config/train_pred_{}.pt'.format(PROCESSING_TYPE.value))
    torch.save(total_val_pred, 'config/val_pred_{}.pt'.format(PROCESSING_TYPE.value))
    torch.save(total_test_pred, 'config/test_pred_{}.pt'.format(PROCESSING_TYPE.value))
    torch.save(total_train_labels, 'config/train_labels_{}.pt'.format(PROCESSING_TYPE.value))
    torch.save(total_val_labels, 'config/val_labels_{}.pt'.format(PROCESSING_TYPE.value))
    torch.save(total_test_labels, 'config/test_labels_{}.pt'.format(PROCESSING_TYPE.value))

 
if __name__ == "__main__":

    if CUDA_ACTIVATED:
        torch.cuda.empty_cache()
    
    enable_cuda_launch_blocking()

    # args = parse_arguments()
    model_name = Models.CNN_3D
    print("TRAINING MODEL ENSEMBLE:",  model_name.value, "+ MLP")

    train_loader, val_loader = get_train_val_data_loaders(model_name)
    test_loader = get_test_loaders(model=model_name)
    store_ensembled_predictions(train_loader, val_loader, test_loader, model_name)
