# Import necessary libraries and modules
import torch
from torch import nn
from torch import optim
from models.model_constants import *  # Import constants from a custom module
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Lambda
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, \
    ShortSideScale, Normalize

# Custom module for model constants (e.g., NUM_FRAMES, MEAN, STD)
# and other relevant constants used in the code

# Define a custom module named "PackPathwaySlowFast" which inherits from nn.Module
class PackPathwaySlowFast(nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, alpha=4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames):
        fast_pathway = frames
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.alpha
            ).long())
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

# Define a function to get the transformation pipeline for SlowFast models
def get_slowfast_transformations():

    transformations = [
        UniformTemporalSubsample(NUM_FRAMES),
        Lambda(lambda x: x / 255.0),
        Normalize(MEAN, STD),
        ShortSideScale(size=SIDE_SIZE_SLOWFAST),
        PackPathwaySlowFast()
    ]

    return ApplyTransformToKey(
        key="video",
        transform=Compose(transformations),
    )

# Define a function to get data loaders for SlowFast models
def get_slowfast_data_loaders(is_eval=False, data_folder=PROCESSED_VIDEO_FOLDER, use_test_data=True):

    if is_eval:
        set_name = ("val", "test")[use_test_data]
        test_data = labeled_video_dataset('{}/{}'.format(data_folder, set_name),
                                          make_clip_sampler(
            'constant_clips_per_video', CLIP_DURATION, 1),
            transform=get_slowfast_transformations(), decode_audio=False)

        test_loader = DataLoader(
            test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        return test_loader

    else:
        train_data = labeled_video_dataset('{}/train'.format(data_folder),
                                           make_clip_sampler(
                                               'random', CLIP_DURATION),
                                           transform=get_slowfast_transformations(), decode_audio=False)

        val_data = labeled_video_dataset('{}/val'.format(data_folder),
                                         make_clip_sampler(
            'constant_clips_per_video', CLIP_DURATION, 1),
            transform=get_slowfast_transformations(), decode_audio=False)

        train_loader = DataLoader(
            train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
        val_loader = DataLoader(
            val_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

        return train_loader, val_loader

# Define a function to get the SlowFast model along with loss criterion and optimizer
def get_slowfast_model(num_labels, lr=LEARNING_RATE_SLOWFAST, momentum=MOMENTUM_SLOWFAST,
                       weight_decay=WEIGHT_DECAY_SLOWFAST, adam_optimizer=False, cross_entropy=True):

    # Model definition and setup
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

    # Choose the appropriate loss criterion (CrossEntropyLoss or NLLLoss)
    if cross_entropy:
        loss_criterion = nn.CrossEntropyLoss()
    else:
        loss_criterion = nn.NLLLoss()

    # Choose the optimizer (Adam or SGD)
    if adam_optimizer:
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr,
                              momentum=momentum, weight_decay=weight_decay)

    return model, loss_criterion, optimizer
