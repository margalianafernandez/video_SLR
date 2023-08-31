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
from models.ensamble import get_test_loaders
from processing.data_constants import ProcessingType

current_datetime = datetime.datetime.now()
current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

CONF_MATRIX_FILENAME = f"{CHECKPOINTS_PATH}/ensamble_conf_matrix_{current_datetime_str}.png"


def get_mapping_labels(test_loader):
    mapping = {}

    for video_path, num_label in test_loader.dataset._labeled_videos._paths_and_labels:
        str_label = os.path.basename(os.path.dirname(video_path))

        if str_label in mapping:
            continue

        mapping[num_label] = str_label

    return mapping


def load_model(model_file_name):

    # Load the pre-trained 3DCNN models
    model_1 = torch.load(join(ROOT_PATH, "models_3dcnn/3dcnn_model_all.pth")) # , map_location=torch.device('cpu')
    model_2 = torch.load(join(ROOT_PATH, "models_3dcnn/3dcnn_model_body_and_hands.pth"))
    model_3 = torch.load(join(ROOT_PATH, "models_3dcnn/3dcnn_model_face_and_hands.pth"))

    # Load the saved model
    model_mlp = torch.load(model_file_name, map_location=torch.device('cpu'))

    return [model_1, model_2, model_3], model_mlp


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


def get_sample_batch_data(batch):

    if CUDA_ACTIVATED:
        video, label = batch['video'].cuda(), batch['label'].cuda()
    else:
        video, label = batch['video'], batch['label']

    return video, label


def show_accuracy(labels, preds):
    correct_predictions = sum([int(label == pred)
                              for label, pred in zip(labels, preds)])
    accuracy = 100 * correct_predictions / len(labels)
    print(f'Accuracy during testing: {accuracy:.2f}%')

    return  accuracy

def evaluate_model(model_file_name):

    labels, predictions = [], []

    val_loader = get_test_loaders()

    models_3dcnn, model_mlp = load_model(model_file_name)

    # Set the model in evaluation mode
    model_mlp.eval()

    with torch.no_grad():
        
        iterator = zip(val_loader[ProcessingType.ALL], val_loader[ProcessingType.FACE_HANDS], val_loader[ProcessingType.BODY_HANDS])

        for batches_all, batches_fah, batches_bah in iterator:
            
            batches = [batches_all, batches_bah, batches_fah]
            labels = batches_all['label']
            outputs_3dcnn = []

            for loader_batch, model_3dcnn in zip(batches, models_3dcnn):
                video, __ = get_sample_batch_data(loader_batch)
                out = model_3dcnn(video)
                outputs_3dcnn += [out.argmax(dim=-1)]

            outputs_3dcnn = torch.stack(outputs_3dcnn, dim=1)
            pred = model_mlp(outputs_3dcnn)
            labels += labels.tolist()  # Convert tensor to a list of integers
           
            predictions += pred.argmax(dim=-1).tolist()
    

    # Mapping numeric labels to their origin name
    mapping = get_mapping_labels(val_loader[ProcessingType.ALL])
    preds_name = [mapping[num_label] for num_label in predictions]
    labels_name = [mapping[num_label] for num_label in labels]

    store_confussion_matrix(labels_name, preds_name)
    return show_accuracy(labels, predictions)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate video classification model.")
    parser.add_argument("--file", type=str, default="model.pth",
                        help="Name of the file containing the trained model")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    print(args.checkpoint)

    print("EVALUATING MODEL ENSAMBLE")
    evaluate_model(args.file)
