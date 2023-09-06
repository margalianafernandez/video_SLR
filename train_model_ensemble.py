# Import necessary libraries
import os
import torch
torch.cuda.empty_cache()  # Clear GPU memory

import math
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.backends import cudnn
from models.ensemble import *
from models.model_constants import *
from evaluate_ensemble_model import evaluate_model
from processing.data_constants import NUM_LABELS, ProcessingType

# Get the current date and time for creating filenames
current_datetime = datetime.datetime.now()
current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S').replace(" ", "_")

# Define filenames for storing metrics, loss function plot, and model
METRICS_FILENAME = f"{CHECKPOINTS_PATH}/ensemble_metrics_{current_datetime_str}.csv"
LOSS_FUNC_FILENAME = f"{CHECKPOINTS_PATH}/ensemble_loss_func_{current_datetime_str}.png"
MODEL_FILENAME = f"{CHECKPOINTS_PATH}/ensemble_model_{current_datetime_str}.pth"

def get_prev_models(prev_model_type):
    """
    Load pre-trained 3DCNN models based on the specified type.
    Args:
        prev_model_type: The type of the pre-trained models (e.g., Models.SLOWFAST).

    Returns:
        A list of loaded pre-trained models.
    """
    # Define file paths for pre-trained models based on the type
    if prev_model_type == Models.SLOWFAST:
        model_1_file = "check_points_all__all_opt/slowfast_model_2023-08-30_12:14:49.pth"
        model_2_file = "check_points_bah_opt/slowfast_model_2023-08-30_21:41:28.pth"
        model_3_file = "check_points_fah_opt/slowfast_model_2023-08-31_00:03:33.pth"
    else:
        model_1_file = "models_3dcnn/3dcnn_model_all.pth"
        model_2_file = "models_3dcnn/3dcnn_model_body_and_hands.pth"
        model_3_file = "models_3dcnn/3dcnn_model_face_and_hands.pth"

    # Load and return the pre-trained models
    model_1 = torch.load(join(ROOT_PATH, model_1_file), map_location=torch.device('cpu'))
    model_2 = torch.load(join(ROOT_PATH, model_2_file), map_location=torch.device('cpu'))
    model_3 = torch.load(join(ROOT_PATH, model_3_file), map_location=torch.device('cpu'))
    
    return [model_1, model_2, model_3]

def enable_cuda_launch_blocking():
    """
    Enable CUDA launch blocking for reproducibility and set random seeds.
    """
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # Set random seeds for reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    
    if CUDA_ACTIVATED:
        cudnn.deterministic = True
        cudnn.benchmark = True

def store_loss_function(train_losses, val_losses):
    """
    Plot and save the training and validation loss curves.
    Args:
        train_losses: List of training losses.
        val_losses: List of validation losses.
    """
    # Plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Save the plot as an image
    plt.savefig(LOSS_FUNC_FILENAME)

def train_one_epoch(model, prev_models, train_loaders, optimizer, loss_criterion, epoch, epochs=EPOCHS):
    """
    Train the model for one epoch.
    Args:
        model: The ensemble model to train.
        prev_models: List of pre-trained 3DCNN models.
        train_loaders: Data loaders for training data.
        optimizer: The optimizer for updating model parameters.
        loss_criterion: The loss criterion for calculating the loss.
        epoch: The current epoch number.
        epochs: The total number of epochs (default is EPOCHS).
    
    Returns:
        acc: The accuracy for the epoch.
        train_losses: List of training losses for each batch.
    """
    train_losses = []

    model.train()
    total_loss, total_acc, total_num = 0.0, 0, 0
    
    num_videos = train_loaders[ProcessingType.ALL].dataset.num_videos
    total = math.ceil(num_videos / BATCH_SIZE)
    train_bars_all = tqdm(train_loaders[ProcessingType.ALL], total=total, dynamic_ncols=False)

    iterator = zip(train_bars_all, train_loaders[ProcessingType.BODY_HANDS], train_loaders[ProcessingType.FACE_HANDS])

    predictions = torch.Tensor()
    for batches_all, batches_fah, batches_bah in iterator:
   
        batches = [batches_all, batches_bah, batches_fah]
        outputs_3dcnn = []
        labels = batches_all['label']
        
        for loader_batch, model_3dcnn in zip(batches, prev_models):
            out = model_3dcnn(loader_batch['video'])
            outputs_3dcnn += [out]
        
        outputs_3dcnn = torch.stack(outputs_3dcnn, dim=1)
        outputs_3dcnn = outputs_3dcnn.reshape(outputs_3dcnn.shape[0], outputs_3dcnn.shape[1]*outputs_3dcnn.shape[2]) 
        
        predictions = torch.cat((predictions, outputs_3dcnn), dim=0)
        
        if CUDA_ACTIVATED:
            outputs_3dcnn = outputs_3dcnn.cuda()
            labels = labels.cuda()
            model = model.cuda()

        optimizer.zero_grad()

        pred = model(outputs_3dcnn)
        loss = loss_criterion(pred, labels)
        
        loss.backward()
        optimizer.step()

        total_num += len(labels)

        total_loss += loss.item() * total_num
        total_acc += (torch.eq(pred.argmax(dim=-1), labels)).sum().item()
        train_losses += [loss.item()]
        
        train_bars_all.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}%'
                                .format(epoch, epochs, total_loss / total_num, total_acc * 100 / total_num))

    # Calculate and print the average loss and accuracy for the epoch
    acc = total_acc / total_num

    return acc, np.array(train_losses).mean()

def validate_one_epoch(model, prev_models, val_loader, loss_criterion, epoch, epochs=EPOCHS):
    """
    Validate the model for one epoch.
    Args:
        model: The ensemble model to validate.
        prev_models: List of pre-trained 3DCNN models.
        val_loader: Data loader for validation data.
        loss_criterion: The loss criterion for calculating the loss.
        epoch: The current epoch number.
        epochs: The total number of epochs (default is EPOCHS).
    
    Returns:
        acc: The accuracy for the epoch.
        val_losses: List of validation losses for each batch.
    """
    val_losses = []
    total_acc, total_num = 0, 0

    model.eval()
    num_videos = val_loader[ProcessingType.ALL].dataset.num_videos
    total = math.ceil(num_videos / BATCH_SIZE)
    val_bars_all = tqdm(val_loader[ProcessingType.ALL], total=total, dynamic_ncols=False)

    predictions = torch.Tensor()
    
    with torch.no_grad():

        iterator = zip(val_bars_all, val_loader[ProcessingType.FACE_HANDS], val_loader[ProcessingType.BODY_HANDS])
        
        for batches_all, batches_fah, batches_bah in iterator:
            batches = [batches_all, batches_bah, batches_fah]
            outputs_3dcnn = []
            labels = batches_all['label']

            for loader_batch, model_3dcnn in zip(batches, prev_models):
                out = model_3dcnn(loader_batch['video'])
                outputs_3dcnn += [out]

            outputs_3dcnn = torch.stack(outputs_3dcnn, dim=1)
            outputs_3dcnn = outputs_3dcnn.reshape(outputs_3dcnn.shape[0], outputs_3dcnn.shape[1]*outputs_3dcnn.shape[2]) 
            
            predictions = torch.cat((predictions, outputs_3dcnn), dim=0)

            if CUDA_ACTIVATED:
                outputs_3dcnn = outputs_3dcnn.cuda()
                labels = labels.cuda()
                model = model.cuda()

            preds = model(outputs_3dcnn)
            loss = loss_criterion(preds, labels)
            val_losses += [loss.item()]
            total_num += len(labels)
            total_acc += (torch.eq(preds.argmax(dim=-1), labels)).sum().item()

            val_bars_all.set_description('Test Epoch: [{}/{}] | Top-1:{:.2f}%'
                                     .format(epoch, epochs, total_acc * 100 / total_num))
    

    # Calculate and print the average loss and accuracy for the epoch
    acc = total_acc / total_num
    
    print("Accuracy while training:", acc)
    return acc, np.array(val_losses).mean()

def train_model(train_loader, val_loader, prev_model_type, model, loss_criterion, \
                optimizer, epochs=EPOCHS, store_files=True):
    """
    Train the ensemble model.
    Args:
        train_loader: Data loader for training data.
        val_loader: Data loader for validation data.
        prev_model_type: The type of pre-trained models to use.
        model: The ensemble model to train.
        loss_criterion: The loss criterion for calculating the loss.
        optimizer: The optimizer for updating model parameters.
        epochs: The total number of epochs (default is EPOCHS).
        store_files: Whether to store loss and model files (default is True).
    
    Returns:
        best_model: The best model based on validation accuracy.
    """
    if not os.path.exists(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH)

    prev_models = get_prev_models(prev_model_type)
    best_acc, best_model = 0.0, None
    train_epoch_loss, val_epoch_loss = [], []

    for epoch in range(1, epochs + 1):

        train_acc, train_losses = train_one_epoch(
            model, prev_models, train_loader, optimizer, loss_criterion, epoch, epochs=epochs)

        val_acc, val_losses = validate_one_epoch(
            model, prev_models, val_loader, loss_criterion, epoch, epochs=epochs)            

        train_epoch_loss += [train_losses]
        val_epoch_loss += [val_losses]

        if val_acc > best_acc:
            best_acc = val_acc
            best_model = model

    torch.save(best_model, MODEL_FILENAME)

    if store_files:
        store_loss_function(train_epoch_loss, val_epoch_loss)

    return best_model

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train video classification model.")
    parser.add_argument("--eval", type=bool, default=True,
                        help="True if the evaluate_model script also needs to be executed, otherwise False.")
    return parser.parse_args()

if __name__ == "__main__":
    # Initialize and configure GPU, if available
    if CUDA_ACTIVATED:
        torch.cuda.empty_cache()
    
    enable_cuda_launch_blocking()

    # Define the model name and print information
    model_name = Models.CNN_3D
    print("TRAINING MODEL ENSEMBLE:",  model_name.value, "+ MLP")

    # Get data loaders for training and validation
    train_loaders, val_loaders = get_train_val_data_loaders(model_name)

    # Create the ensemble model, loss criterion, and optimizer
    model, loss_criterion, optimizer = get_ensemble_model(NUM_LABELS, train_loaders[ProcessingType.ALL])

    # Train the model
    train_model(train_loaders, val_loaders, model_name, model, loss_criterion, optimizer)

    # Evaluate the model if needed
    if True:
        evaluate_model(MODEL_FILENAME, model_name)
