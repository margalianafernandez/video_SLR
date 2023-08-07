import os
import torch
import json
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from slowfast_constants import *
from slowfast import get_test_transform
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from data_constants import DATASET_FILE, TRAIN
from pytorchvideo.data import make_clip_sampler, labeled_video_dataset

current_datetime = datetime.datetime.now()
current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

model_filename = '{}/slow_fast_2023-08-04 11:17:50.pth'.format(CHECKPOINTS_PATH)
conf_matrix_filename = "{}/confusion_matrix_{}.png".format(CHECKPOINTS_PATH, current_datetime_str)


def get_labels():
    content = json.load(open(DATASET_FILE))
    labels = content[TRAIN].keys()
    return list(labels)


def get_mapping_labels(test_loader):
    mapping = {}

    for video_path, num_label in test_loader.dataset._labeled_videos._paths_and_labels:
        str_label = os.path.basename(os.path.dirname(video_path))

        if str_label in mapping: continue

        mapping[num_label] =  str_label

    return mapping


def get_data_loaders():
    
    test_transform = get_test_transform()

    # data prepare

    test_data = labeled_video_dataset('{}/test'.format(DATA_PATH),
                                    make_clip_sampler('constant_clips_per_video', CLIP_DURATION, 1),
                                    transform=test_transform, decode_audio=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, num_workers=8)

    return test_loader


def load_model():

    # Load the saved model
    model = torch.load(model_filename, map_location=torch.device('cpu'))
    
    return model


def store_confussion_matrix(labels, preds):
    labels_name = get_labels()
    
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, preds)

    # Plot confusion matrix as an image
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels_name, yticklabels=labels_name)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the plot as an image
    plt.savefig(conf_matrix_filename)  


def show_accuracy(labels, preds):
    correct_predictions = sum([int(label == pred) for label, pred in zip(labels, preds)])
    accuracy = 100  * correct_predictions / len(labels) 
    print(f'Accuracy: {accuracy:.2f}%')


def evaluate_model(model, test_loader):

    labels, predictions = [], []

    # Set the model in evaluation mode
    model.eval()
    
    with torch.no_grad():
        for batch in test_loader:
            video, label = batch['video'], batch['label']
            pred = model(video)

            labels += label.tolist()  # Convert tensor to a list of integers
            predictions += pred.argmax(dim=-1).tolist()  # Convert tensor to a list of integers

    # Mapping numeric labels to their origin name
    mapping = get_mapping_labels(test_loader)
    preds_name = [mapping[num_label] for num_label in predictions]
    labels_name = [mapping[num_label] for num_label in labels]
    
    store_confussion_matrix(labels_name, preds_name)
    show_accuracy(labels, predictions)


if __name__ == "__main__":
    
    test_loader = get_data_loaders()
    model = load_model()
    evaluate_model(model, test_loader)