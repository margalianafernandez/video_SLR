import os
import torch
torch.cuda.empty_cache()
from torch.utils.data import DataLoader

import math
import random
import argparse
import datetime
import numpy as np
from tqdm import tqdm
from models.ensamble import *
import matplotlib.pyplot as plt
from torch.backends import cudnn
from models.model_constants import *
from evaluate_ensamble_model import evaluate_model
from processing.data_constants import NUM_LABELS, ProcessingType

current_datetime = datetime.datetime.now()
current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S').replace(" ", "_")

METRICS_FILENAME = f"{CHECKPOINTS_PATH}/ensamble_metrics_{current_datetime_str}.csv"
LOSS_FUNC_FILENAME = f"{CHECKPOINTS_PATH}/ensamble_loss_func_{current_datetime_str}.png"
MODEL_FILENAME = f"{CHECKPOINTS_PATH}/ensamble_model_{current_datetime_str}.pth"


def get_3dcnn_models():
    # Load the pre-trained 3DCNN models
    # model_1 = torch.load(join(ROOT_PATH, "models_3dcnn/3dcnn_model_all.pth"), map_location=torch.device('cpu'))
    # model_2 = torch.load(join(ROOT_PATH, "models_3dcnn/3dcnn_model_body_and_hands.pth"), map_location=torch.device('cpu'))
    # model_3 = torch.load(join(ROOT_PATH, "models_3dcnn/3dcnn_model_face_and_hands.pth"), map_location=torch.device('cpu'))
    model_1_file = "check_points_all__all_opt/3dcnn_model_2023-08-30_14:44:21.pth"
    model_2_file = "check_points_bah_opt/3dcnn_model_2023-08-30_17:33:47.pth"
    model_3_file = "check_points_fah_opt/3dcnn_model_2023-08-30_19:48:25.pth"

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
    
    if CUDA_ACTIVATED:
        cudnn.deterministic = True
        cudnn.benchmark = True


def store_loss_function(train_losses, val_losses):

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


def get_sample_batch_data(batch):

    if CUDA_ACTIVATED:
        video, label = batch['video'].cuda(), batch['label'].cuda()
    else:
        video, label = batch['video'], batch['label']

    return video, label


def train_one_epoch(model, models_3dcnn, train_loaders, optimizer, loss_criterion, epoch, epochs=EPOCHS):
    train_losses = []

    model.train()
    total_loss, total_acc, total_num = 0.0, 0, 0
    
    num_videos = train_loaders[ProcessingType.ALL].dataset.num_videos
    total = math.ceil(num_videos / BATCH_SIZE)
    train_bars_all = tqdm(train_loaders[ProcessingType.ALL], total=total, dynamic_ncols=False)

    iterator = zip(train_bars_all, train_loaders[ProcessingType.FACE_HANDS], train_loaders[ProcessingType.BODY_HANDS])

    for batches_all, batches_fah, batches_bah in iterator:
   
        batches = [batches_all, batches_bah, batches_fah]
        outputs_3dcnn = []
        labels = batches_all['label']
        
        for loader_batch, model_3dcnn in zip(batches, models_3dcnn):
            video, label = get_sample_batch_data(loader_batch)
            out = model_3dcnn(video)
            outputs_3dcnn += [out.argmax(dim=-1)]
        
        outputs_3dcnn = torch.stack(outputs_3dcnn, dim=1)

        optimizer.zero_grad()

        pred = model(outputs_3dcnn)
        loss = loss_criterion(pred, labels)
        
        loss.backward()
        optimizer.step()

        total_num += len(labels)

        total_loss += loss.item() * total_num
        total_acc += (torch.eq(pred.argmax(dim=-1), label)).sum().item()
        train_losses += [loss.item()]
        
        train_bars_all.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}%'
                                .format(epoch, epochs, total_loss / total_num, total_acc * 100 / total_num))

    # Calculate and print the average loss and accuracy for the epoch
    loss = total_loss / total_num
    acc = total_acc / total_num

    return acc, np.array(train_losses).mean()


def validate_one_epoch(model, models_3dcnn, val_loader, loss_criterion, epoch, epochs=EPOCHS):
    val_losses = []
    total_acc, total_num = 0, 0

    model.eval()
    num_videos = val_loader[ProcessingType.ALL].dataset.num_videos
    total = math.ceil(num_videos / BATCH_SIZE)
    val_bars_all = tqdm(val_loader[ProcessingType.ALL], total=total, dynamic_ncols=False)

    with torch.no_grad():

        iterator = zip(val_bars_all, val_loader[ProcessingType.FACE_HANDS], val_loader[ProcessingType.BODY_HANDS])
        
        # Iterate using a common index
        for batches_all, batches_fah, batches_bah in iterator:
            batches = [batches_all, batches_bah, batches_fah]
            outputs_3dcnn = []
            labels = batches_all['label']

            for loader_batch, model_3dcnn in zip(batches, models_3dcnn):
                video, __ = get_sample_batch_data(loader_batch)
                out = model_3dcnn(video)
                outputs_3dcnn += [out.argmax(dim=-1)]

            outputs_3dcnn = torch.stack(outputs_3dcnn, dim=1)

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



def train_model(train_loader, val_loader, model, loss_criterion, optimizer, epochs=EPOCHS, store_files=True):

    if not os.path.exists(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH)

    models_3dcnn = get_3dcnn_models()
    best_acc, best_model = 0.0, None
    train_epoch_loss, val_epoch_loss = [], []

    for epoch in range(1, epochs + 1):

        train_acc, train_losses = train_one_epoch(
            model, models_3dcnn, train_loader, optimizer, loss_criterion, epoch, epochs=epochs)

        val_acc, val_losses = validate_one_epoch(
            model, models_3dcnn, val_loader, loss_criterion, epoch, epochs=epochs)            

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
    parser.add_argument("--eval", type=bool, default=False,
                        help="True if the evaluate_model script also needs to be executed, otherwise False.")
    return parser.parse_args()

 
if __name__ == "__main__":

    if CUDA_ACTIVATED:
        torch.cuda.empty_cache()
    
    enable_cuda_launch_blocking()

    args = parse_arguments()

    print("TRAINING MODEL ENSAMBLE")

    train_loaders, val_loaders = get_train_val_data_loaders()
    model, loss_criterion, optimizer = get_ensemble_model(NUM_LABELS)

    train_model(train_loaders, val_loaders, model, loss_criterion, optimizer)

    if args.eval:
        evaluate_model(MODEL_FILENAME)
