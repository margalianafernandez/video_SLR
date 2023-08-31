import torch
import torch.nn as nn
from models.cnn3d_model import *
from processing.data_constants import ProcessingType, TRAIN, TEST, VALIDATION


class EnsembleMLP(nn.Module):
    def __init__(self, num_labels):
        super(EnsembleMLP, self).__init__()

        # Create the MLP for combining outputs
        self.mlp = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
            nn.Linear(512, num_labels)
        )

    def forward(self, outputs_3dcnn):
        # Combine the outputs from the three 3DCNN models
        input = outputs_3dcnn.to(self.mlp[0].weight.dtype)
        output = self.mlp(input)
        return output


def get_ensamble_data_loaders(set=TRAIN, data_folder=PROCESSED_VIDEO_FOLDER):
    folder_path = '{}/{}'.format(data_folder, set)

    if set == TRAIN:
        clip_sample = make_clip_sampler('random', CLIP_DURATION)
    else:
        clip_sample = make_clip_sampler('constant_clips_per_video', CLIP_DURATION, 1)
        
    transform = get_transformations()
    dataset = labeled_video_dataset(folder_path, clip_sample, transform=transform, video_sampler=torch.utils.data.SequentialSampler, decode_audio=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    return loader


def get_train_val_data_loaders():
    train_loaders = {
        ProcessingType.ALL: get_ensamble_data_loaders(TRAIN, PROCESSED_VIDEO_FOLDER_ALL),
        ProcessingType.BODY_HANDS: get_ensamble_data_loaders(TRAIN, PROCESSED_VIDEO_FOLDER_BODY_AND_HANDS),
        ProcessingType.FACE_HANDS: get_ensamble_data_loaders(TRAIN, PROCESSED_VIDEO_FOLDER_FACE_AND_HANDS)
    }

    val_loaders = {
        ProcessingType.ALL: get_ensamble_data_loaders(VALIDATION, PROCESSED_VIDEO_FOLDER_ALL),
        ProcessingType.BODY_HANDS: get_ensamble_data_loaders(VALIDATION, PROCESSED_VIDEO_FOLDER_BODY_AND_HANDS),
        ProcessingType.FACE_HANDS: get_ensamble_data_loaders(VALIDATION, PROCESSED_VIDEO_FOLDER_FACE_AND_HANDS)
    }

    return train_loaders, val_loaders


def get_test_loaders():
    
    return {
        ProcessingType.ALL: get_ensamble_data_loaders(TEST, PROCESSED_VIDEO_FOLDER_ALL),
        ProcessingType.BODY_HANDS: get_ensamble_data_loaders(TEST, PROCESSED_VIDEO_FOLDER_BODY_AND_HANDS),
        ProcessingType.FACE_HANDS: get_ensamble_data_loaders(TEST, PROCESSED_VIDEO_FOLDER_FACE_AND_HANDS)
    }


def get_ensemble_model(num_labels):

    model = EnsembleMLP(num_labels)

    if CUDA_ACTIVATED:
        model = model.cuda()

    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE_3DCNN,
                          momentum=MOMENTUM_3DCNN, weight_decay=WEIGHT_DECAY_3DCNN)

    return model, loss_criterion, optimizer
