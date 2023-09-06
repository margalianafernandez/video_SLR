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
from models.ensemble import get_test_loaders
from processing.data_constants import ProcessingType

current_datetime = datetime.datetime.now()
current_datetime_str = current_datetime.strftime('%Y-%m-%d %H:%M:%S')

CONF_MATRIX_FILENAME = f"{CHECKPOINTS_PATH}/ensemble_conf_matrix_{current_datetime_str}.png"


def get_mapping_labels(test_loader):
    mapping = {}

    for video_path, num_label in test_loader.dataset._labeled_videos._paths_and_labels:
        str_label = os.path.basename(os.path.dirname(video_path))

        if str_label in mapping:
            continue

        mapping[num_label] = str_label

    return mapping


def load_model(model_file_name, prev_model_type):

    if prev_model_type == Models.SLOWFAST:
        model_1_file = "check_points_all__all_opt/slowfast_model_2023-08-30_12:14:49.pth"
        model_2_file = "check_points_bah_opt/slowfast_model_2023-08-30_21:41:28.pth"
        model_3_file = "check_points_fah_opt/slowfast_model_2023-08-31_00:03:33.pth"

    else:
        model_1_file = "models_3dcnn/3dcnn_model_all.pth"
        model_2_file = "models_3dcnn/3dcnn_model_body_and_hands.pth"
        model_3_file = "models_3dcnn/3dcnn_model_face_and_hands.pth"

    model_1 = torch.load(join(ROOT_PATH, model_1_file),
                         map_location=torch.device('cpu'))
    model_2 = torch.load(join(ROOT_PATH, model_2_file),
                         map_location=torch.device('cpu'))
    model_3 = torch.load(join(ROOT_PATH, model_3_file),
                         map_location=torch.device('cpu'))

    # Load the saved model
    if CUDA_ACTIVATED:
        model_mlp = torch.load(model_file_name).cuda()
    else:
        model_mlp = torch.load(
            model_file_name, map_location=torch.device('cpu'))

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


def show_accuracy(labels, preds):
    correct_predictions = sum([int(label == pred)
                              for label, pred in zip(labels, preds)])
    accuracy = 100 * correct_predictions / len(labels)
    print(f'Accuracy during testing: {accuracy:.2f}%')

    return accuracy


def evaluate_model(model_file_name, prev_model_type):

    all_labels, predictions = [], []

    val_loader = get_test_loaders(model=prev_model_type)

    models_3dcnn, model_mlp = load_model(model_file_name, prev_model_type)

    # Set the model in evaluation mode
    model_mlp.eval()
    
    predictions = torch.Tensor()

    with torch.no_grad():

        iterator = zip(val_loader[ProcessingType.ALL],
                       val_loader[ProcessingType.BODY_HANDS], val_loader[ProcessingType.FACE_HANDS])

        for batches_all, batches_fah, batches_bah in iterator:

            batches = [batches_all, batches_bah, batches_fah]
            labels = batches_all['label']
            outputs_3dcnn = []

            for loader_batch, model_3dcnn in zip(batches, models_3dcnn):
                out = model_3dcnn(loader_batch['video'])
                outputs_3dcnn += [out]
                # outputs_3dcnn += [out.argmax(dim=-1)]

            outputs_3dcnn = torch.stack(outputs_3dcnn, dim=1)
            outputs_3dcnn = outputs_3dcnn.reshape(
                outputs_3dcnn.shape[0], outputs_3dcnn.shape[1]*outputs_3dcnn.shape[2])

            predictions = torch.cat((predictions, outputs_3dcnn), dim=0)
        
            if CUDA_ACTIVATED:
                outputs_3dcnn = outputs_3dcnn.cuda()
                labels = labels.cuda()
                model = model.cuda()

            pred = model_mlp(outputs_3dcnn)
            all_labels += labels.tolist()  # Convert tensor to a list of integers
            predictions += pred.argmax(dim=-1).tolist()

    # Mapping numeric labels to their origin name
    mapping = get_mapping_labels(val_loader[ProcessingType.ALL])
    preds_name = [mapping[num_label] for num_label in predictions]
    labels_name = [mapping[num_label] for num_label in all_labels]

    store_confussion_matrix(labels_name, preds_name)
    return show_accuracy(all_labels, predictions)


def parse_arguments():
    default_file = join(
        CHECKPOINTS_PATH, "ensemble_model_2023-08-31_17:28:51.pth")

    parser = argparse.ArgumentParser(
        description="Evaluate video classification model.")
    parser.add_argument("--file", type=str, default=default_file,
                        help="Name of the file containing the trained model")
    parser.add_argument("--model", type=Models, default=Models.CNN_3D,
                        help="Name of the model to train: " + Models.SLOWFAST.value + " or " +
                        Models.CNN_3D.value)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()

    print("EVALUATING MODEL ENSEMBLE")
    evaluate_model(args.file, args.model)
