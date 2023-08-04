# https://github.com/kailliang/Lane-Change-Classification-and-Prediction-with-Action-Recognition-Networks/tree/main

import os

def enable_cuda_launch_blocking():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"


# Call the function to enable CUDA launch blocking
enable_cuda_launch_blocking()


import math
import torch
import random
import datetime
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.backends import cudnn
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from slowfast_constants import *
from sklearn.metrics import confusion_matrix
from pytorchvideo.models import create_slowfast
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset

from torchvision.transforms import Compose, Lambda, RandomCrop, CenterCrop
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, \
    RandomShortSideScale, ShortSideScale, Normalize

torch.cuda.empty_cache()


# for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = True

current_datetime = datetime.datetime.now()
current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

metrics_filename = "{}/metrics_{}.csv".format(CHECKPOINTS_PATH, current_datetime_str)
model_filename = '{}/slow_fast_{}.pth'.format(CHECKPOINTS_PATH, current_datetime_str)


class PackPathway(nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """
    def __init__(self, alpha=4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames):
        fast_pathway = frames
        # perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames, 
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long())
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


def get_train_transform():
    
    return ApplyTransformToKey(
        key = "video", 
        transform = Compose(
            [
                UniformTemporalSubsample(NUM_FRAMES), 
                Lambda(lambda x: x / 255.0), 
                Normalize(MEAN, STD), 
                ShortSideScale(size=SIDE_SIZE), 
                PackPathway()
            ]
        ),
    )


def get_test_transform():
    
    return ApplyTransformToKey(
        key = "video", 
        transform = Compose(
            [
                UniformTemporalSubsample(NUM_FRAMES), 
                Lambda(lambda x: x / 255.0), 
                Normalize(MEAN, STD), 
                ShortSideScale(size=SIDE_SIZE), 
                PackPathway()
            ]
        ),
    )


def get_data_loaders():
    
    train_transform = get_train_transform()
    test_transform = get_test_transform()

    # data prepare
    train_data = labeled_video_dataset('{}/train'.format(DATA_PATH), make_clip_sampler('random', CLIP_DURATION),
                                    transform=train_transform, decode_audio=False)

    val_data = labeled_video_dataset('{}/val'.format(DATA_PATH),
                                    make_clip_sampler('constant_clips_per_video', CLIP_DURATION, 1),
                                    transform=test_transform, decode_audio=False)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=8)

    return train_loader, val_loader, train_data, val_data


def get_slowfast_model():
    
    # model define, loss setup and optimizer config
    slow_fast = torch.hub.load('facebookresearch/pytorchvideo:main', model='slowfast_r50', pretrained=True).cuda()
    slow_fast.blocks[6].proj = torch.nn.Linear(in_features=2304, out_features=NUM_LABELS, bias=True).cuda()

    loss_criterion = CrossEntropyLoss()
    optimizer = SGD(slow_fast.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    return slow_fast, loss_criterion, optimizer


def train_one_epoch(model, train_loader, train_data, optimizer, loss_criterion, epoch):
    
    model.train()
    total_loss, total_acc, total_num = 0.0, 0, 0
    train_bar = tqdm(train_loader, total=math.ceil(train_data.num_videos / BATCH_SIZE), dynamic_ncols=True)
    
    for batch in train_bar:
        video, label = [i.cuda() for i in batch['video']], batch['label'].cuda()

        optimizer.zero_grad()
        pred = model(video)
        loss = loss_criterion(pred, label)
        total_loss += loss.item() * video[0].size(0)
        total_acc += (torch.eq(pred.argmax(dim=-1), label)).sum().item()
        loss.backward()
        optimizer.step()

        total_num += video[0].size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}%'
                                  .format(epoch, EPOCHS, total_loss / total_num, total_acc * 100 / total_num))

    return total_loss / total_num, total_acc / total_num


# test for one epoch
def validate(model, val_loader, val_data, epoch):

    model.eval()
    
    with torch.no_grad():
        total_top_1, total_top_5, total_num = 0, 0, 0
        test_bar = tqdm(val_loader, total=math.ceil(val_data.num_videos / BATCH_SIZE), dynamic_ncols=True)
        
        for batch in test_bar:
            video, labels = [i.cuda() for i in batch['video']], batch['label'].cuda()

            preds = model(video)
            total_top_1 += (torch.eq(preds.argmax(dim=-1), labels)).sum().item()
            total_top_5 += torch.any(torch.eq(preds.topk(k=2, dim=-1).indices, labels.unsqueeze(dim=-1)),
                                     dim=-1).sum().item()
            total_num += video[0].size(0)
            test_bar.set_description('Test Epoch: [{}/{}] | Top-1:{:.2f}% | Top-5:{:.2f}%'
                                     .format(epoch, EPOCHS, total_top_1 * 100 / total_num,
                                             total_top_5 * 100 / total_num))
    
    top_1, top_5 = total_top_1 / total_num, total_top_5 / total_num

    return top_1, top_5


def train_model(train_loader, val_loader, train_data, val_data, slow_fast, loss_criterion, optimizer):
    
    # training loop
    results = {'loss': [], 'acc': [], 'top-1': [], 'top-5': []}
    
    if not os.path.exists(CHECKPOINTS_PATH):
        os.makedirs(CHECKPOINTS_PATH)

    best_acc, best_model = 0.0, None
    
    for epoch in range(1, EPOCHS + 1):
        
        loss, acc = train_one_epoch(slow_fast, train_loader, train_data, optimizer, loss_criterion, epoch)
        
        results['loss'].append(loss)
        results['acc'].append(acc * 100)
        
        top_1, top_5 = validate(slow_fast, val_loader, val_data, epoch)
        
        results['top-1'].append(top_1 * 100)
        results['top-5'].append(top_5 * 100)
        
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv(metrics_filename, index_label='epoch')

        if top_1 > best_acc:
            best_acc = top_1
            best_model = slow_fast
    
    torch.save(best_model, model_filename) 

    print("Best accuracy: {:.2f}".format(top_1 * 100))


def main():
    
    train_loader, val_loader, train_data, val_data = get_data_loaders()
    slow_fast, loss_criterion, optimizer = get_slowfast_model()
    
    train_model(train_loader, val_loader, train_data, val_data, slow_fast, loss_criterion, optimizer)


if __name__ == "__main__":
    
    main()   
