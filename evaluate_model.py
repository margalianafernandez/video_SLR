import os
import json
import torch
import datetime
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from models.model_constants import *
from processing.data_constants import LABELS
from sklearn.metrics import confusion_matrix
from models.cnn3d_model import get_3dcnn_data_loaders
from models.slowfast_model import get_slowfast_data_loaders


def define_file_names(model_name, checkpoint=CHECKPOINTS_PATH):
    global CONF_MATRIX_FILENAME

    current_datetime = datetime.datetime.now()
    current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

    CONF_MATRIX_FILENAME = f"{checkpoint}/{model_name}_conf_matrix_{current_datetime_str}.png"


def get_mapping_labels(test_loader):
    mapping = {}

    for video_path, num_label in test_loader.dataset._labeled_videos._paths_and_labels:
        str_label = os.path.basename(os.path.dirname(video_path))

        if str_label in mapping:
            continue

        mapping[num_label] = str_label

    return mapping


def get_data_loaders(model_type, data_folder=PROCESSED_VIDEO_FOLDER,  use_test_data=True):

    if model_type == Models.SLOWFAST:
        test_loader = get_slowfast_data_loaders(is_eval=True, data_folder=data_folder, use_test_data=use_test_data)
    else:
        test_loader = get_3dcnn_data_loaders(is_eval=True, data_folder=data_folder, use_test_data=use_test_data)

    return test_loader


def load_model(model_file_name):

    # Load the saved model
    model = torch.load(model_file_name, map_location=torch.device('cpu'))

    return model


def store_confussion_matrix(labels, preds):

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, preds)

    # Plot confusion matrix as an image
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=LABELS, yticklabels=LABELS)
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

    return  accuracy

def evaluate_model(model_type, model_file_name, checkpoint_path=CHECKPOINTS_PATH, data_folder=PROCESSED_VIDEO_FOLDER, use_test_data=True):

    labels, predictions = [], []

    define_file_names(model_type.value, checkpoint=checkpoint_path)

    test_loader = get_data_loaders(model_type, data_folder=data_folder, use_test_data=use_test_data)
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
    return show_accuracy(labels, predictions)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate video classification model.")
    parser.add_argument("--model", type=Models, default=Models.CNN_3D,
                        help="Name of the model to train: SLOWFAST or CNN_3D")
    parser.add_argument("--file", type=str, default="model.pth",
                        help="Name of the file containing the trained model")
    parser.add_argument("--data", type=str, default=PROCESSED_VIDEO_FOLDER,
                        help="Path to the dataset folder tousee for testing, training and validation.")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINTS_PATH,
                        help="Path to the checkpoint to store the output files.")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    print(args.checkpoint)

    print("EVALUATING MODEL", args.model.value.upper())
    evaluate_model(args.model, args.file, checkpoint_path=args.checkpoint, data_folder=args.data)
