# Import necessary libraries
import os
import torch
torch.cuda.empty_cache()

# Import components for mixed-precision training
from torch.cuda.amp import autocast, GradScaler

# Import other required libraries
import math
import random
import argparse
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import custom modules and functions
from torch.backends import cudnn
from models.model_constants import *
from evaluate_model import evaluate_model
from processing.data_constants import NUM_LABELS
from models.cnn3d_model import get_3dcnn_model, get_3dcnn_data_loaders
from models.slowfast_model import get_slowfast_model, get_slowfast_data_loaders


# Function to define file names for storing model-related data
def define_file_names(model_name, checkpoint_path=CHECKPOINTS_PATH, model_filename=None):
    global METRICS_FILENAME, LOSS_FUNC_FILENAME, MODEL_FILENAME

    # If a custom model filename is not provided, generate filenames with timestamps
    if model_filename == None:
        current_datetime = datetime.datetime.now()
        current_datetime_str = current_datetime.strftime(
            '%Y-%m-%d %H:%M:%S').replace(" ", "_")

        METRICS_FILENAME = f"{checkpoint_path}/{model_name}_metrics_{current_datetime_str}.csv"
        LOSS_FUNC_FILENAME = f"{checkpoint_path}/{model_name}_loss_func_{current_datetime_str}.png"
        MODEL_FILENAME = f"{checkpoint_path}/{model_name}_model_{current_datetime_str}.pth"

    # If a custom model filename is provided, use it
    else:
        MODEL_FILENAME = model_filename

# Function to enable CUDA launch blocking and set random seed for reproducibility


def enable_cuda_launch_blocking():
    # Set environment variables for CUDA launch blocking and CUDA-Direct Storage Access
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # For reproducibility, seed the random number generators
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    cudnn.deterministic = True
    cudnn.benchmark = True

# Function to store and visualize loss function


def store_loss_function(train_losses, val_losses):
    # Plot the training and validation loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Save the loss curve plot as an image
    plt.savefig(LOSS_FUNC_FILENAME)

# Function to retrieve batch data and move it to GPU if available


def get_sample_batch_data(batch, model_type):
    # Check if CUDA is activated and move data to GPU if necessary
    if CUDA_ACTIVATED:
        if model_type == Models.SLOWFAST:
            video, label = [i.cuda()
                            for i in batch['video']], batch['label'].cuda()
        else:
            video, label = batch['video'].cuda(), batch['label'].cuda()
    else:
        video, label = batch['video'], batch['label']

    return video, label

# Function to train the model for one epoch


def train_one_epoch(model, model_type, train_loader, optimizer, loss_criterion, epoch, epochs=EPOCHS):
    train_losses = []

    # Set the model to training mode
    model.train()
    total_loss, total_acc, total_num = 0.0, 0, 0
    num_videos = train_loader.dataset.num_videos

    # Create a progress bar for training
    train_bar = tqdm(train_loader, total=math.ceil(
        num_videos / BATCH_SIZE), dynamic_ncols=True)

    # Create a gradient scaler for mixed-precision training
    scaler = GradScaler()

    for batch in train_bar:
        video, label = get_sample_batch_data(batch, model_type)

        # Zero out the gradients
        optimizer.zero_grad()

        # Use autocast to perform mixed-precision training
        with autocast():
            pred = model(video)
            loss = loss_criterion(pred, label)

        # Scale the loss and backpropagate
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Track total loss, accuracy, and individual batch losses
        total_loss += loss.item() * video[0].size(0)
        total_acc += (torch.eq(pred.argmax(dim=-1), label)).sum().item()
        train_losses += [loss.item()]

        total_num += video[0].size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}%'
                                  .format(epoch, epochs, total_loss / total_num, total_acc * 100 / total_num))

    # Calculate average loss and accuracy for the epoch
    loss = total_loss / total_num
    acc = total_acc / total_num

    return loss, acc, np.array(train_losses).mean()

# Function to validate the model for one epoch


def validate_one_epoch(model, model_type, val_loader, loss_criterion, epoch, epochs=EPOCHS):
    val_losses = []

    # Set the model to evaluation mode (no gradients)
    model.eval()

    with torch.no_grad():
        total_top_1, total_top_5, total_num = 0, 0, 0
        num_videos = val_loader.dataset.num_videos

        # Create a progress bar for validation
        test_bar = tqdm(val_loader, total=math.ceil(
            num_videos / BATCH_SIZE), dynamic_ncols=True)

        for batch in test_bar:
            video, label = get_sample_batch_data(batch, model_type)

            preds = model(video)
            loss = loss_criterion(preds, label)
            val_losses += [loss.item()]

            # Track top-1 and top-5 accuracy
            total_top_1 += (torch.eq(preds.argmax(dim=-1),
                            label)).sum().item()
            total_top_5 += torch.any(torch.eq(preds.topk(k=2, dim=-1).indices, label.unsqueeze(dim=-1)),
                                     dim=-1).sum().item()
            total_num += video[0].size(0)

            test_bar.set_description('Test Epoch: [{}/{}] | Top-1:{:.2f}% | Top-5:{:.2f}%'
                                     .format(epoch, epochs, total_top_1 * 100 / total_num,
                                             total_top_5 * 100 / total_num))

    # Calculate top-1 and top-5 accuracy for the epoch
    top_1, top_5 = total_top_1 / total_num, total_top_5 / total_num

    return top_1, top_5, np.array(val_losses).mean()


# Function to train the model
def train_model(train_loader, val_loader, model, model_type, loss_criterion, optimizer, epochs=EPOCHS, store_files=True):
    # Initialize a dictionary to store training metrics
    results = {'loss': [], 'acc': [], 'top-1': [], 'top-5': []}

    # Create the checkpoint directory if it doesn't exist
    if not os.path.exists(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH)

    best_acc, best_model = 0.0, None
    train_epoch_loss, val_epoch_loss = [], []

    for epoch in range(1, epochs + 1):

        # Train the model for one epoch and get training metrics
        loss, acc, train_losses = train_one_epoch(
            model, model_type, train_loader, optimizer, loss_criterion, epoch, epochs=epochs)

        # Validate the model for one epoch and get validation metrics
        top_1, top_5, val_losses = validate_one_epoch(
            model, model_type, val_loader, loss_criterion, epoch, epochs=epochs)

        if store_files:
            # Store training metrics in the results dictionary
            results['loss'].append(loss)
            results['acc'].append(acc * 100)
            results['top-1'].append(top_1 * 100)
            results['top-5'].append(top_5 * 100)

            # Save the training metrics to a CSV file
            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv(METRICS_FILENAME, index_label='epoch')

        # Append training and validation loss for plotting
        train_epoch_loss += [train_losses]
        val_epoch_loss += [val_losses]

        # Update the best model if current top-1 accuracy is higher
        if top_1 > best_acc:
            best_acc = top_1
            best_model = model

    # Save the best model to a file
    torch.save(best_model, MODEL_FILENAME)

    if store_files:
        # Store and visualize the training and validation loss function
        store_loss_function(train_epoch_loss, val_epoch_loss)

    return best_model

# Function to parse command-line arguments


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train video classification model.")
    parser.add_argument("--model", type=Models, default=Models.SLOWFAST,
                        help="Name of the model to train: " + Models.SLOWFAST.value + " or " +
                        Models.CNN_3D.value)
    parser.add_argument("--eval", type=bool, default=False,
                        help="True if the evaluate_model script also needs to be executed, otherwise False.")
    parser.add_argument("--data", type=str, default=PROCESSED_VIDEO_FOLDER,
                        help="Path to the dataset folder for testing, training, and validation.")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINTS_PATH,
                        help="Path to the checkpoint to store the output files.")
    return parser.parse_args()


# Main program entry point
if __name__ == "__main__":

    if CUDA_ACTIVATED:
        enable_cuda_launch_blocking()
        torch.cuda.empty_cache()

    # Parse command-line arguments
    args = parse_arguments()

    # Set the checkpoint path and define model-related file names
    CHECKPOINTS_PATH = args.checkpoint
    define_file_names(args.model.value, checkpoint_path=args.checkpoint)

    print("TRAINING MODEL", args.model.value.upper())

    # Load data loaders and model based on the selected model type
    if args.model == Models.SLOWFAST:
        train_loader, val_loader = get_slowfast_data_loaders(
            data_folder=args.data)
        model, loss_criterion, optimizer = get_slowfast_model(NUM_LABELS)
    else:
        train_loader, val_loader = get_3dcnn_data_loaders(
            data_folder=args.data)
        model, loss_criterion, optimizer = get_3dcnn_model(NUM_LABELS)

    # Train the model
    train_model(train_loader, val_loader, model,
                args.model, loss_criterion, optimizer)

    # If specified, evaluate the trained model
    if args.eval:
        evaluate_model(args.model, MODEL_FILENAME,
                       checkpoint_path=args.checkpoint, data_folder=args.data)
