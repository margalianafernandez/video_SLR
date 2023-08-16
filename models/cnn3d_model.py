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
        test_data = labeled_video_dataset('{}/val'.format(PROCESSED_VIDEO_FOLDER),
                                          make_clip_sampler(
                                              'constant_clips_per_video', CLIP_DURATION, 1),
                                          transform=get_transformations(),
                                          decode_audio=False)

        test_loader = DataLoader(
            test_data, batch_size=BATCH_SIZE, num_workers=8)

        return test_loader

    train_data = labeled_video_dataset('{}/train'.format(PROCESSED_VIDEO_FOLDER),
                                       make_clip_sampler(
                                           'random', CLIP_DURATION),
                                       transform=get_transformations(),
                                       decode_audio=False)
    test_data = labeled_video_dataset('{}/val'.format(PROCESSED_VIDEO_FOLDER),
                                      make_clip_sampler(
                                          'constant_clips_per_video', CLIP_DURATION, 1),
                                      transform=get_transformations(),
                                      decode_audio=False)
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=8)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=8)

    return train_loader, test_loader


def get_3dcnn_model(num_labels):
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

    loss_criterion = nn.CrossEntropyLoss()
    # optimizer = Adam(slow_fast.parameters(), lr=1e-1)
    optimizer = optim.SGD(model.parameters(), lr=0.02,
                          momentum=0.9, weight_decay=0.001)

    return model, loss_criterion, optimizer
