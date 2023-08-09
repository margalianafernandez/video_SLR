import math
import torch
import numpy as np
from torch import nn
from torch import optim
from tqdm.notebook import tqdm
from torchvision import transforms
from models.model_constants import *
from torch.utils.data import DataLoader
from torchvision.models.video import r3d_18
from processing.data_constants import NUM_LABELS
from torchvision.transforms import Compose, Lambda
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, \
    ShortSideScale, Normalize


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def torch_accuracy(pred, target):
    correct = (pred == target).sum().item()
    return 100 * correct / len(target)


def collate_fn_r3d_18(batch):
    # imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [i['video'] for i in batch]
    label_batch = [i['label'] for i in batch]
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    # imgs_tensor = torch.transpose(imgs_tensor, 2, 1)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor


def get_data_loaders():
    h, w = 112, 112
    mean = [0.43216, 0.394666, 0.37645]
    std = [0.22803, 0.22145, 0.216989]

    tfs = [
        Lambda(lambda frames: frames.permute(0, 3, 1, 2)),  # Convert to channels-first format
        # UniformTemporalSubsample(8),  # Subsample frames
        UniformTemporalSubsample(NUM_FRAMES), 
        transforms.Resize((h, w)),
        Normalize(mean=mean, std=std),  # Normalize
        Lambda(lambda frames: frames.float()),  # Convert to float
    ]

    transformations = ApplyTransformToKey(
        key = "video", 
        transform = Compose(tfs),
    )

    # data prepare
    train_data = labeled_video_dataset('{}/train'.format(DATA_PATH), make_clip_sampler('random', CLIP_DURATION),
                                    transform=transformations, decode_audio=False)

    val_data = labeled_video_dataset('{}/val'.format(DATA_PATH),
                                    make_clip_sampler('constant_clips_per_video', CLIP_DURATION, 1),
                                    transform=transformations, decode_audio=False)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn= collate_fn_r3d_18)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn= collate_fn_r3d_18)

    return train_loader, val_loader, train_data, val_data



def train_one_epoch(model, train_loader, train_data, optimizer, loss_criterion, epoch):
    train_losses = []

    model.train()
    total_loss, total_acc, total_num = 0.0, 0, 0
    train_bar = tqdm(train_loader, total=math.ceil(train_data.num_videos / BATCH_SIZE), dynamic_ncols=True)
    
    for video, label in train_bar:
        # video, label = [i.cuda() for i in batch['video']], batch['label'].cuda()

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


def validate_one_epoch(model, val_loader, val_data, loss_criterion, epoch):
    val_losses = []

    model.eval()
    
    with torch.no_grad():
        total_top_1, total_top_5, total_num = 0, 0, 0
        test_bar = tqdm(val_loader, total=math.ceil(val_data.num_videos / BATCH_SIZE), dynamic_ncols=True)
        
        for batch in test_bar:
            video, labels = [i.cuda() for i in batch['video']], batch['label'].cuda()

            preds = model(video)
            loss = loss_criterion(preds, labels)
            val_losses += [loss.item()]

            total_top_1 += (torch.eq(preds.argmax(dim=-1), labels)).sum().item()
            total_top_5 += torch.any(torch.eq(preds.topk(k=2, dim=-1).indices, labels.unsqueeze(dim=-1)),
                                     dim=-1).sum().item()
            total_num += video[0].size(0)
            
            test_bar.set_description('Test Epoch: [{}/{}] | Top-1:{:.2f}% | Top-5:{:.2f}%'
                                     .format(epoch, EPOCHS, total_top_1 * 100 / total_num,
                                             total_top_5 * 100 / total_num))
    
    top_1, top_5 = total_top_1 / total_num, total_top_5 / total_num

    return top_1, top_5, np.array(val_losses).mean()


def train_model(train_loader, val_loader, train_data, val_data, model, loss_criterion, optimizer):


    best_acc = 0.0
    
    for epoch in range(1, EPOCHS + 1):
        
        loss, acc, train_losses = train_one_epoch(model, train_loader, train_data, optimizer, loss_criterion, epoch)
        
        top_1, top_5, val_losses = validate_one_epoch(model, val_loader, val_data, loss_criterion, epoch)
        
    

        if top_1 > best_acc:
            best_acc = top_1
            best_model = model
    

    print("Best accuracy: {:.2f}".format(top_1 * 100))



def main():
    model = r3d_18(pretrained=True) 
    model.fc = nn.Linear(model.fc.in_features, NUM_LABELS)
    
    train_loader, val_loader, train_dataset, val_dataset = get_data_loaders()
    
    loss_criterion = nn.CrossEntropyLoss(reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr=3e-5)

    train_model(train_loader, val_loader, train_dataset, val_dataset, model, loss_criterion, optimizer)



if __name__ ==  "__main__":
    main()