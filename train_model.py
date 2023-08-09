# https://github.com/kailliang/Lane-Change-Classification-and-Prediction-with-Action-Recognition-Networks/tree/main

import os
import math
import torch
import random
import argparse
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.backends import cudnn
from models.model_constants import *
from evaluate_model import evaluate_model
from processing.data_constants import NUM_LABELS
from models.cnn3d_model import get_3dcnn_model, get_3dcnn_data_loaders
from models.slowfast_model import get_slowfast_model, get_slowfast_data_loaders


def define_file_names(model_name):
    global METRICS_FILENAME, LOSS_FUNC_FILENAME, MODEL_FILENAME

    current_datetime = datetime.datetime.now()
    current_datetime_str = current_datetime.strftime(
        '%Y-%m-%d %H:%M:%S').replace(" ", "_")

    METRICS_FILENAME = f"{CHECKPOINTS_PATH}/{model_name}_metrics_{current_datetime_str}.csv"
    LOSS_FUNC_FILENAME = f"{CHECKPOINTS_PATH}/{model_name}_loss_func_{current_datetime_str}.png"
    MODEL_FILENAME = f"{CHECKPOINTS_PATH}/{model_name}_model_{current_datetime_str}.pth"


def enable_cuda_launch_blocking():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # for reproducibility
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
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


def get_sample_batch_data(batch, model_type):

    if CUDA_ACTIVATED:
        if model_type == Models.SLOWFAST:
            video, label = [i.cuda()
                            for i in batch['video']], batch['label'].cuda()
        else:
            video, label = batch['video'].cuda(), batch['label'].cuda()
    else:
        video, label = batch['video'], batch['label']

    return video, label


def train_one_epoch(model, model_type, train_loader, optimizer, loss_criterion, epoch):
    train_losses = []

    model.train()
    total_loss, total_acc, total_num = 0.0, 0, 0
    num_videos = train_loader.dataset.num_videos

    train_bar = tqdm(train_loader, total=math.ceil(
        num_videos / BATCH_SIZE), dynamic_ncols=True)

    for batch in train_bar:
        video, label = get_sample_batch_data(batch, model_type)

        optimizer.zero_grad()
        pred = model(video)
        loss = loss_criterion(pred, label)
        total_loss += loss.item() * video[0].size(0)
        total_acc += (torch.eq(pred.argmax(dim=-1), label)).sum().item()
        loss.backward()
        optimizer.step()

        train_losses += [loss.item()]

        total_num += video[0].size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}%'
                                  .format(epoch, EPOCHS, total_loss / total_num, total_acc * 100 / total_num))

    loss = total_loss / total_num
    acc = total_acc / total_num

    return loss, acc, np.array(train_losses).mean()


def validate_one_epoch(model, model_type, val_loader, loss_criterion, epoch):
    val_losses = []

    model.eval()

    with torch.no_grad():
        total_top_1, total_top_5, total_num = 0, 0, 0
        num_videos = val_loader.dataset.num_videos

        test_bar = tqdm(val_loader, total=math.ceil(
            num_videos / BATCH_SIZE), dynamic_ncols=True)

        for batch in test_bar:
            video, label = get_sample_batch_data(batch, model_type)

            preds = model(video)
            loss = loss_criterion(preds, label)
            val_losses += [loss.item()]

            total_top_1 += (torch.eq(preds.argmax(dim=-1),
                            label)).sum().item()
            total_top_5 += torch.any(torch.eq(preds.topk(k=2, dim=-1).indices, label.unsqueeze(dim=-1)),
                                     dim=-1).sum().item()
            total_num += video[0].size(0)

            test_bar.set_description('Test Epoch: [{}/{}] | Top-1:{:.2f}% | Top-5:{:.2f}%'
                                     .format(epoch, EPOCHS, total_top_1 * 100 / total_num,
                                             total_top_5 * 100 / total_num))

    top_1, top_5 = total_top_1 / total_num, total_top_5 / total_num

    return top_1, top_5, np.array(val_losses).mean()


def train_model(train_loader, val_loader, model, model_type, loss_criterion, optimizer):

    # training loop
    results = {'loss': [], 'acc': [], 'top-1': [], 'top-5': []}

    if not os.path.exists(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH)

    best_acc, best_model = 0.0, None
    train_epoch_loss, val_epoch_loss = [], []

    for epoch in range(1, EPOCHS + 1):

        loss, acc, train_losses = train_one_epoch(
            model, model_type, train_loader, optimizer, loss_criterion, epoch)

        results['loss'].append(loss)
        results['acc'].append(acc * 100)

        top_1, top_5, val_losses = validate_one_epoch(
            model, model_type, val_loader, loss_criterion, epoch)

        results['top-1'].append(top_1 * 100)
        results['top-5'].append(top_5 * 100)

        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(METRICS_FILENAME, index_label='epoch')

        train_epoch_loss += [train_losses]
        val_epoch_loss += [val_losses]

        if top_1 > best_acc:
            best_acc = top_1
            best_model = model

    torch.save(best_model, MODEL_FILENAME)

    store_loss_function(train_epoch_loss, val_epoch_loss)

    print("Best accuracy: {:.2f}".format(top_1 * 100))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train video classification model.")
    parser.add_argument("--model", type=Models, default=Models.SLOWFAST,
                        help="Name of the model to train: " + Models.SLOWFAST.value + " or " +
                        Models.CNN_3D.value)
    parser.add_argument("--eval", type=bool, default=False,
                        help="True if the evaluate_model script also needs to be executed, otherwise False.")

    return parser.parse_args()


if __name__ == "__main__":

    if CUDA_ACTIVATED:
        enable_cuda_launch_blocking()
        torch.cuda.empty_cache()

    args = parse_arguments()

    define_file_names(args.model.value)

    print("TRAINING MODEL", args.model.value.upper())

    if args.model == Models.SLOWFAST:
        train_loader, val_loader = get_slowfast_data_loaders()
        model, loss_criterion, optimizer = get_slowfast_model(NUM_LABELS)
    else:
        train_loader, val_loader = get_3dcnn_data_loaders()
        model, loss_criterion, optimizer = get_3dcnn_model(NUM_LABELS)

    train_model(train_loader, val_loader, model,
                args.model, loss_criterion, optimizer)

    if args.eval:
        evaluate_model(args.model, MODEL_FILENAME)
