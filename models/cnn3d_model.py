import torch
from torch import nn
from torch import optim
from models.model_constants import *
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, \
    ShortSideScale, Normalize


class PackPathway(nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def forward(self, frames):
        frame_list = frames
        # perform temporal sampling from the fast pathway.

        return frame_list


def get_transformations():
    return ApplyTransformToKey(
        key="video",
        transform=Compose(
            [UniformTemporalSubsample(NUM_FRAMES),
             Lambda(lambda x: x / 255.0),
             Normalize(MEAN, STD),
             ShortSideScale(size=SIDE_SIZE),
             PackPathway()]
        )
    )


def get_3dcnn_data_loaders(is_eval=False):

    if is_eval:
        test_data = labeled_video_dataset('{}/test'.format(PROCESSED_VIDEO_FOLDER),
                                          make_clip_sampler(
            'constant_clips_per_video', CLIP_DURATION, 1),
            transform=get_transformations(), decode_audio=False)

        test_loader = DataLoader(
            test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        return test_loader

    else:
        train_data = labeled_video_dataset('{}/train'.format(PROCESSED_VIDEO_FOLDER),
                                           make_clip_sampler(
                                               'random', CLIP_DURATION),
                                           transform=get_transformations(), decode_audio=False)

        val_data = labeled_video_dataset('{}/val'.format(PROCESSED_VIDEO_FOLDER),
                                         make_clip_sampler(
            'constant_clips_per_video', CLIP_DURATION, 1),
            transform=get_transformations(), decode_audio=False)

        train_loader = DataLoader(
            train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        val_loader = DataLoader(
            val_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        return train_loader, val_loader


def get_3dcnn_model(num_labels, adam_optimizer=False, cross_entropy=True, 
                    lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY):
    # model define, loss setup and optimizer config
    model_name = 'i3d_r50'

    if CUDA_ACTIVATED:

        model = torch.hub.load(
            'facebookresearch/pytorchvideo:main', model_name, pretrained=True).cuda()
        model.blocks[6].proj = torch.nn.Linear(
            in_features=2048, out_features=num_labels, bias=True).cuda()

    else:
        model = torch.hub.load(
            'facebookresearch/pytorchvideo:main', model_name, pretrained=True)
        model.blocks[6].proj = torch.nn.Linear(
            in_features=2048, out_features=num_labels, bias=True)

    if cross_entropy:
        loss_criterion = nn.CrossEntropyLoss()
    else:
        loss_criterion = nn.NLLLoss()

    if adam_optimizer:
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)

    return model, loss_criterion, optimizer
