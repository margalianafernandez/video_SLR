import torch
from torch import nn
from torch.optim import SGD
from models.model_constants import *
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, \
    ShortSideScale, Normalize


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


def get_transformations():

    transformations = [
        UniformTemporalSubsample(NUM_FRAMES),
        Lambda(lambda x: x / 255.0),
        Normalize(MEAN, STD),
        ShortSideScale(size=SIDE_SIZE),
        PackPathway()
    ]

    return ApplyTransformToKey(
        key="video",
        transform=Compose(transformations),
    )


def get_slowfast_data_loaders(is_eval=False):

    transformations = get_transformations()

    if is_eval:
        test_data = labeled_video_dataset('{}/test'.format(PROCESSED_VIDEO_FOLDER),
                                          make_clip_sampler(
            'constant_clips_per_video', CLIP_DURATION, 1),
            transform=transformations, decode_audio=False)

        test_loader = DataLoader(
            test_data, batch_size=BATCH_SIZE, num_workers=8)

        return test_loader

    else:
        train_data = labeled_video_dataset('{}/train'.format(PROCESSED_VIDEO_FOLDER),
                                           make_clip_sampler(
                                               'random', CLIP_DURATION),
                                           transform=transformations, decode_audio=False)

        val_data = labeled_video_dataset('{}/val'.format(PROCESSED_VIDEO_FOLDER),
                                         make_clip_sampler(
            'constant_clips_per_video', CLIP_DURATION, 1),
            transform=transformations, decode_audio=False)

        train_loader = DataLoader(
            train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=True)

        return train_loader, val_loader


def get_slowfast_model(num_labels):

    # model define, loss setup and optimizer config
    if CUDA_ACTIVATED:
        model = torch.hub.load('facebookresearch/pytorchvideo:main',
                               model='slowfast_r50', pretrained=True).cuda()
        model.blocks[6].proj = torch.nn.Linear(
            in_features=2304, out_features=num_labels, bias=True).cuda()
    else:
        model = torch.hub.load('facebookresearch/pytorchvideo:main',
                               model='slowfast_r50', pretrained=True)
        model.blocks[6].proj = torch.nn.Linear(
            in_features=2304, out_features=num_labels, bias=True)

    loss_criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE,
                    momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    return model, loss_criterion, optimizer
