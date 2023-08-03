# https://github.com/kailliang/Lane-Change-Classification-and-Prediction-with-Action-Recognition-Networks/tree/main

import os
import datetime

def enable_cuda_launch_blocking():
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["TORCH_USE_CUDA_DSA"] = "1"

# Call the function to enable CUDA launch blocking
enable_cuda_launch_blocking()


import math
import torch
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.backends import cudnn
from torch.optim import Adam, SGD
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from pytorchvideo.models import create_slowfast
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset


#from utils import train_transform, test_transform, clip_duration, num_classes

from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, RandomShortSideScale, \
    ShortSideScale, Normalize
from torch import nn
from torchvision.transforms import Compose, Lambda, RandomCrop, CenterCrop


torch.cuda.empty_cache()

side_size = 336
max_size = 368
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 336
num_frames = 32
sampling_rate = 1
frames_per_second = 32/5
clip_duration = (num_frames * sampling_rate) / frames_per_second
n_classes = 20
#checkpoint_path = '/kaggle/working/SLOWFAST_8x8_R50.pyth'

data_root = "/dcs/pg22/u2288875/Documents/TFM/processed_data"
batch_size = 5 # 10
epochs = 50
save_root = '/dcs/pg22/u2288875/Documents/TFM/check_points'

# for reproducibility
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
cudnn.deterministic = True
cudnn.benchmark = True

current_datetime = datetime.datetime.now()
current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

metrics_filename = "{}/metrics_{}.csv".format(save_root, current_datetime_str)
model_filename = '{}/slow_fast_{}.pth'.format(save_root, current_datetime_str)
conf_matrix_filename = "{}/confusion_matrix_{}.csv".format(save_root, current_datetime_str)


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


train_transform = ApplyTransformToKey(
    key = "video", 
    transform = Compose(
        [
            UniformTemporalSubsample(num_frames), 
            Lambda(lambda x: x / 255.0), 
            Normalize(mean, std), 
            ShortSideScale(size=side_size), 
            PackPathway()
        ]
    ),
)

test_transform = ApplyTransformToKey(
    key = "video", 
    transform = Compose(
        [
            UniformTemporalSubsample(num_frames), 
            Lambda(lambda x: x / 255.0), 
            Normalize(mean, std), 
            ShortSideScale(size=side_size), 
            PackPathway()
        ]
    ),
)

# train for one epoch
def train(model, data_loader, train_optimizer):
    model.train()
    total_loss, total_acc, total_num = 0.0, 0, 0
    train_bar = tqdm(data_loader, total=math.ceil(train_data.num_videos / batch_size), dynamic_ncols=True)
    
    for batch in train_bar:
        video, label = [i.cuda() for i in batch['video']], batch['label'].cuda()

        train_optimizer.zero_grad()
        pred = model(video)
        loss = loss_criterion(pred, label)
        total_loss += loss.item() * video[0].size(0)
        total_acc += (torch.eq(pred.argmax(dim=-1), label)).sum().item()
        loss.backward()
        train_optimizer.step()

        total_num += video[0].size(0)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}%'
                                  .format(epoch, epochs, total_loss / total_num, total_acc * 100 / total_num))

    return total_loss / total_num, total_acc / total_num


# test for one epoch
def val(model, data_loader):
    preds, labels = [], []

    model.eval()
    with torch.no_grad():
        total_top_1, total_top_5, total_num = 0, 0, 0
        test_bar = tqdm(data_loader, total=math.ceil(test_data.num_videos / batch_size), dynamic_ncols=True)
        for batch in test_bar:
            video, labels = [i.cuda() for i in batch['video']], batch['label'].cuda()

            preds = model(video)
            total_top_1 += (torch.eq(preds.argmax(dim=-1), labels)).sum().item()
            total_top_5 += torch.any(torch.eq(preds.topk(k=2, dim=-1).indices, labels.unsqueeze(dim=-1)),
                                     dim=-1).sum().item()
            total_num += video[0].size(0)
            test_bar.set_description('Test Epoch: [{}/{}] | Top-1:{:.2f}% | Top-5:{:.2f}%'
                                     .format(epoch, epochs, total_top_1 * 100 / total_num,
                                             total_top_5 * 100 / total_num))
    
    top_1, top_5 = total_top_1 / total_num, total_top_5 / total_num
    return top_1, top_5, preds, labels


# data prepare
train_data = labeled_video_dataset('{}/train'.format(data_root), make_clip_sampler('random', clip_duration),
                                   transform=train_transform, decode_audio=False)
test_data = labeled_video_dataset('{}/test'.format(data_root),
                                  make_clip_sampler('constant_clips_per_video', clip_duration, 1),
                                  transform=test_transform, decode_audio=False)
train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8)
test_loader = DataLoader(test_data, batch_size=batch_size, num_workers=8)


#------------------------------------------------------------------------------------------------------------

# model define, loss setup and optimizer config
slow_fast = torch.hub.load('facebookresearch/pytorchvideo:main', model='slowfast_r50', pretrained=True).cuda()
slow_fast.blocks[6].proj = torch.nn.Linear(in_features=2304, out_features=n_classes, bias=True).cuda()

#------------------------------------------------------------------------------------------------------------


loss_criterion = CrossEntropyLoss()
optimizer = SGD(slow_fast.parameters(), lr=0.02, momentum=0.9,weight_decay=0.001)

##---------------------------------------------------------------------------------------------------------

# training loop
results = {'loss': [], 'acc': [], 'top-1': [], 'top-5': []}
if not os.path.exists(save_root):
    os.makedirs(save_root)

best_acc = 0.0
best_preds, best_labels = [], []
for epoch in range(1, epochs + 1):
    
    train_loss, train_acc = train(slow_fast, train_loader, optimizer)
    results['loss'].append(train_loss)
    results['acc'].append(train_acc * 100)
    top_1, top_5, preds, labels = val(slow_fast, test_loader)
    results['top-1'].append(top_1 * 100)
    results['top-5'].append(top_5 * 100)
    
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    data_frame.to_csv(metrics_filename, index_label='epoch')

    if top_1 > best_acc:
        best_acc = top_1
        best_preds, best_labels = preds, labels
        torch.save(slow_fast.state_dict(), model_filename)

print("Best accuracy: {:.2f}".format(top_1 * 100))

# Calculate confusion matrix
conf_matrix = confusion_matrix(best_labels, best_preds, num_classes=n_classes)
conf_matrix = conf_matrix.cpu().numpy()

# Save confusion matrix to a CSV file
pd.DataFrame(conf_matrix).to_csv(
    conf_matrix_filename, 
    index_label='True_Label', 
    header=['Predicted_Label_{}'.format(i) for i in range(n_classes)]
)
