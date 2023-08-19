import os
import json
import torch
import datetime
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from models.model_constants import *
from sklearn.metrics import confusion_matrix
from models.cnn3d_model import get_3dcnn_data_loaders
from processing.data_constants import DATASET_FILE, TRAIN
from models.slowfast_model import get_slowfast_data_loaders


def define_file_names(model_name):
    global CONF_MATRIX_FILENAME

    current_datetime = datetime.datetime.now()
    current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

    CONF_MATRIX_FILENAME = f"{CHECKPOINTS_PATH}/{model_name}_conf_matrix_{current_datetime_str}.png"


def get_labels():
    content = json.load(open(DATASET_FILE))
    labels = content[TRAIN].keys()
    return list(labels)


def get_mapping_labels(test_loader):
    mapping = {}

    for video_path, num_label in test_loader.dataset._labeled_videos._paths_and_labels:
        str_label = os.path.basename(os.path.dirname(video_path))

        if str_label in mapping:
            continue

        mapping[num_label] = str_label

    return mapping


def get_data_loaders(model_type):

    if model_type == Models.SLOWFAST:
        test_loader = get_slowfast_data_loaders(is_eval=True)
    else:
        test_loader = get_3dcnn_data_loaders(is_eval=True)

    return test_loader


def load_model(model_file_name):

    # Load the saved model
    model = torch.load(model_file_name, map_location=torch.device('cpu'))

    return model


def store_confussion_matrix(labels, preds):
    labels_name = get_labels()

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, preds)

    # Plot confusion matrix as an image
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels_name, yticklabels=labels_name)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Save the plot as an image
    plt.savefig(CONF_MATRIX_FILENAME)


def show_accuracy(labels, preds):
    correct_predictions = sum([int(label == pred)
                              for label, pred in zip(labels, preds)])
    accuracy = 100 * correct_predictions / len(labels)
    print(f'Accuracy during testing: {accuracy:.2f}%')


def evaluate_model(model_type, model_file_name):

    labels, predictions = [], []

    define_file_names(model_type.value)

    test_loader = get_data_loaders(model_type)
    model = load_model(model_file_name)

    # Set the model in evaluation mode
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            video, label = batch['video'], batch['label']
            pred = model(video)

            labels += label.tolist()  # Convert tensor to a list of integers
            # Convert tensor to a list of integers
            predictions += pred.argmax(dim=-1).tolist()

    # Mapping numeric labels to their origin name
    mapping = get_mapping_labels(test_loader)
    preds_name = [mapping[num_label] for num_label in predictions]
    labels_name = [mapping[num_label] for num_label in labels]

    store_confussion_matrix(labels_name, preds_name)
    show_accuracy(labels, predictions)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate video classification model.")
    parser.add_argument("--model", type=Models, default=Models.CNN_3D,
                        help="Name of the model to train: SLOWFAST or CNN_3D")
    parser.add_argument("--file", type=str, default="model.pth",
                        help="Name of the file containing the trained model")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    print("EVALUATING MODEL", args.model.value.upper())
    evaluate_model(args.model, args.file)
