# Import necessary libraries and modules
import torch
import torch.nn as nn
from models.cnn3d_model import *  # Import custom 3D CNN model
from models.slowfast_model import *  # Import custom SlowFast model
from processing.data_constants import ProcessingType, TRAIN, TEST, VALIDATION
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Custom module for model constants (e.g., Models, CLIP_DURATION, BATCH_SIZE)
# and other relevant constants used in the code

# Define an ensemble MLP (Multi-Layer Perceptron) model
class EnsembleMLP(nn.Module):
    def __init__(self, num_labels):
        super(EnsembleMLP, self).__init__()
        self.hidden = nn.Linear(3*num_labels, 25)  # 3 times the number of labels as input
        self.act = nn.ReLU()  # ReLU activation function
        self.output = nn.Linear(25, num_labels)  # Output layer with num_labels neurons

    def forward(self, x):
        # Combine the outputs from the three 3DCNN or SlowFast models
        input = x.to(self.hidden.weight.dtype)
        a = self.act(self.hidden(input))  # Apply ReLU activation to hidden layer
        y = self.output(a)  # Output layer
        return y

# Define a function to get data loaders for the ensemble model
def get_ensemble_data_loaders(model=Models.SLOWFAST, set=TRAIN, data_folder=PROCESSED_VIDEO_FOLDER):
    folder_path = '{}/{}'.format(data_folder, set)

    if set == TRAIN:
        clip_sample = make_clip_sampler('random', CLIP_DURATION)  # Random clip sampling for training
    else:
        clip_sample = make_clip_sampler('constant_clips_per_video', CLIP_DURATION, 1)  # Constant clips per video for validation/test

    # Choose the appropriate transformation pipeline based on the selected model
    transform = (get_3dcnn_transformations(), get_slowfast_transformations())[model == Models.SLOWFAST]
    
    # Create a labeled video dataset and DataLoader
    dataset = labeled_video_dataset(folder_path, clip_sample, transform=transform, video_sampler=torch.utils.data.SequentialSampler, decode_audio=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    return loader

# Define a function to get train and validation data loaders for the ensemble model
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

# Define a function to get test data loaders for the ensemble model
def get_test_loaders(model=Models.SLOWFAST):
    return {
        ProcessingType.ALL: get_ensemble_data_loaders(model, TEST, PROCESSED_VIDEO_FOLDER_ALL),
        ProcessingType.BODY_HANDS: get_ensemble_data_loaders(model, TEST, PROCESSED_VIDEO_FOLDER_BODY_AND_HANDS),
        ProcessingType.FACE_HANDS: get_ensemble_data_loaders(model, TEST, PROCESSED_VIDEO_FOLDER_FACE_AND_HANDS)
    }

# Define a function to get the ensemble model along with loss criterion and optimizer
def get_ensemble_model(num_labels, train_loader):
    model = EnsembleMLP(num_labels)  # Create an instance of the ensemble MLP model

    # Compute class weights for balanced loss
    labels = [int(label) for batch in train_loader for label in batch['label']]
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)  # Convert class weights to a PyTorch tensor

    if CUDA_ACTIVATED:
        model = model.cuda()  # Move the model to the GPU if CUDA is activated
        class_weights = class_weights.cuda()  # Move class weights to the GPU

    # Define the loss criterion (CrossEntropyLoss with class weights) and optimizer (Adam)
    loss_criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use Adam optimizer

    return model, loss_criterion, optimizer
