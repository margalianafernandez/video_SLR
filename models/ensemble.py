import torch
import torch.nn as nn
from models.cnn3d_model import *
from models.slowfast_model import *
from processing.data_constants import ProcessingType, TRAIN, TEST, VALIDATION
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class EnsembleMLP(nn.Module):
    def __init__(self, num_labels):
        super(EnsembleMLP, self).__init__()
        self.hidden = nn.Linear(3*num_labels, 25)
        self.act = nn.ReLU()
        self.output = nn.Linear(25, num_labels)
    def forward(self, x):
        # Combine the outputs from the three 3DCNN or SlowFast models
        input = x.to(self.hidden.weight.dtype)
        a = self.act(self.hidden(input))
        y = self.output(a)
        return y


def get_ensemble_data_loaders(model=Models.SLOWFAST, set=TRAIN, data_folder=PROCESSED_VIDEO_FOLDER):
    folder_path = '{}/{}'.format(data_folder, set)

    if set == TRAIN:
        clip_sample = make_clip_sampler('random', CLIP_DURATION)
    else:
        clip_sample = make_clip_sampler('constant_clips_per_video', CLIP_DURATION, 1)
        
    transform = (get_3dcnn_transformations(), get_slowfast_transformations())[model==Models.SLOWFAST]
    dataset = labeled_video_dataset(folder_path, clip_sample, transform=transform, video_sampler=torch.utils.data.SequentialSampler, decode_audio=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    return loader


def get_train_val_data_loaders(model=Models.SLOWFAST):
    train_loaders = {
        ProcessingType.ALL: get_ensemble_data_loaders(model, TRAIN, PROCESSED_VIDEO_FOLDER_ALL),
        ProcessingType.BODY_HANDS: get_ensemble_data_loaders(model, TRAIN, PROCESSED_VIDEO_FOLDER_BODY_AND_HANDS),
        ProcessingType.FACE_HANDS: get_ensemble_data_loaders(model, TRAIN, PROCESSED_VIDEO_FOLDER_FACE_AND_HANDS)
    }

    val_loaders = {
        ProcessingType.ALL: get_ensemble_data_loaders(model, VALIDATION, PROCESSED_VIDEO_FOLDER_ALL),
        ProcessingType.BODY_HANDS: get_ensemble_data_loaders(model, VALIDATION, PROCESSED_VIDEO_FOLDER_BODY_AND_HANDS),
        ProcessingType.FACE_HANDS: get_ensemble_data_loaders(model, VALIDATION, PROCESSED_VIDEO_FOLDER_FACE_AND_HANDS)
    }

    return train_loaders, val_loaders


def get_test_loaders(model=Models.SLOWFAST):
    
    return {
        ProcessingType.ALL: get_ensemble_data_loaders(model, TEST, PROCESSED_VIDEO_FOLDER_ALL),
        ProcessingType.BODY_HANDS: get_ensemble_data_loaders(model, TEST, PROCESSED_VIDEO_FOLDER_BODY_AND_HANDS),
        ProcessingType.FACE_HANDS: get_ensemble_data_loaders(model, TEST, PROCESSED_VIDEO_FOLDER_FACE_AND_HANDS)
    }


def get_ensemble_model(num_labels, train_loader):

    model = EnsembleMLP(num_labels)

    

    labels = [int(label) for batch in train_loader for  label in batch['label']]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)  # Convert to a PyTorch tensor

    if CUDA_ACTIVATED:
        model = model.cuda()
        class_weights = class_weights.cuda()

    loss_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer =  optim.Adam(model.parameters(), lr=0.001)

    return model, loss_criterion, optimizer
