import torch
from torch import nn
from torch import optim
from torchvision import transforms
from models.model_constants import *
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from torchvision.models.video import r3d_18, R3D_18_Weights
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, Normalize


def collate_r3d_18(batch):
    imgs_batch = [i['video'] for i in batch]
    label_batch = [i['label'] for i in batch]
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs) > 0]
    label_batch = [torch.tensor(l) for l, imgs in zip(
        label_batch, imgs_batch) if len(imgs) > 0]
    imgs_tensor = torch.stack(imgs_batch)
    labels_tensor = torch.stack(label_batch)
    return {'video': imgs_tensor, 'label': labels_tensor}


def get_transformations():

    transformations = [
        Lambda(lambda frames: frames.permute(0, 3, 1, 2)),
        UniformTemporalSubsample(NUM_FRAMES),
        transforms.Resize((SIDE_SIZE, SIDE_SIZE)),
        Normalize(mean=MEAN, std=STD),
        Lambda(lambda frames: frames.float()),  # Convert to float
    ]

    return ApplyTransformToKey(
        key="video",
        transform=Compose(transformations),
    )


def get_3dcnn_data_loaders(is_eval=False):

    transformations = get_transformations()

    if is_eval:
        test_data = labeled_video_dataset('{}/test'.format(PROCESSED_VIDEO_FOLDER),
                                          make_clip_sampler(
            'constant_clips_per_video', CLIP_DURATION, 1),
            transform=transformations, decode_audio=False)
        test_loader = DataLoader(
            test_data, batch_size=BATCH_SIZE, collate_fn=collate_r3d_18)

        return test_loader

    else:
        train_data = labeled_video_dataset('{}/train'.format(PROCESSED_VIDEO_FOLDER), make_clip_sampler('random', CLIP_DURATION),
                                           transform=transformations, decode_audio=False)

        val_data = labeled_video_dataset('{}/val'.format(PROCESSED_VIDEO_FOLDER),
                                         make_clip_sampler(
            'constant_clips_per_video', CLIP_DURATION, 1),
            transform=transformations, decode_audio=False)

        train_loader = DataLoader(
            train_data, batch_size=BATCH_SIZE, collate_fn=collate_r3d_18)
        val_loader = DataLoader(
            val_data, batch_size=BATCH_SIZE, collate_fn=collate_r3d_18)

        return train_loader, val_loader


def get_3dcnn_model(num_labels):

    # model define, loss setup and optimizer config
    if CUDA_ACTIVATED:
        model = r3d_18(pretrained=True).cuda()
        model.fc = nn.Linear(model.fc.in_features, num_labels).cuda()

    else:
        model = r3d_18(weights=R3D_18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, num_labels)

    loss_criterion = nn.CrossEntropyLoss()  # reduction="sum"
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    return model, loss_criterion, optimizer
