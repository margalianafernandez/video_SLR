# Import necessary libraries
from train_model import *
from itertools import product
import matplotlib.pyplot as plt
from evaluate_model import evaluate_model
from processing.data_constants import *

LABELS = face_motion_labels + hands_motion_labels + body_motion_labels
NUM_LABELS = len(LABELS)

def plot_hyperparameters_summary():
    
    # Extract individual hyperparameter values for the graph
    learning_rates = [params[0] for params in hyperparameter_values]
    momentums = [params[1] for params in hyperparameter_values]
    weight_decays = [params[2] for params in hyperparameter_values]

    
    # List of hyperparameter names and their corresponding values
    hyperparameter_names = ["Learning Rate", "Momentum", "Weight Decay"]
    hyperparameter_ranges = [learning_rates, momentums, weight_decays]

    # Loop through hyperparameters
    for i, (hyperparam_name, hyperparam_range) in enumerate(zip(hyperparameter_names, hyperparameter_ranges)):
        plt.figure(figsize=(8, 6))
        plt.scatter(hyperparam_range, accuracies)
        plt.xlabel(hyperparam_name)
        plt.ylabel('Accuracy')
        plt.title(f'{hyperparam_name} vs Accuracy')
        plt.tight_layout()
        plt.savefig(f"{hyperparam_name.lower().replace(' ', '_')}_accuracy.png")
        plt.close()


CURRENT_MODEL = Models.CNN_3D

print("Tunning", CURRENT_MODEL.value)

# Define ranges for hyperparameters

learning_rate_range = [0.001, 0.01, 0.1]
momentum_range = [0.4, 0.6, 0.9]
weight_decay_range = [0.0001, 0.001, 0.01]
epochs_range = [50]
cross_entropy_lf = [False, True]
adam_optimizer = [False, True]

MODEL_FILENAME = os.path.join(CHECKPOINTS_PATH, CURRENT_MODEL.value + "_tmp_model.pth")

define_file_names(CURRENT_MODEL.value, model_filename=MODEL_FILENAME)

best_accuracy = 0.0
best_hyperparameters = None

if CURRENT_MODEL == Models.SLOWFAST:
    get_data_loaders = get_slowfast_data_loaders
    get_model = get_slowfast_model
else:
    get_data_loaders = get_3dcnn_data_loaders
    get_model = get_3dcnn_model

# Generate all combinations of hyperparameters
hyperparam_comb = product(
    learning_rate_range,
    momentum_range,
    weight_decay_range,
    epochs_range,
    adam_optimizer,
    cross_entropy_lf,
)

hyperparameter_values = []
accuracies = []
DATA_FOLDER = "/dcs/pg22/u2288875/Documents/TFM/processed_data_all__all"


for lr, momentum, weight_decay, epochs, use_adam, use_ce in hyperparam_comb:

    print("NEW TEST:")
    print("\tLR:", lr)
    print("\tMOMENTUM:", momentum)
    print("\tWEIGHT DECAY:", weight_decay)
    print("\tWEIGHT DECAY:", weight_decay)
    print("\tADAM OPTIMIZER:", use_adam)
    print("\CROSS ENTROPY LOSS FUNCTION:", use_ce)


    # Train the model using the current hyperparameters
    train_loader, val_loader = get_data_loaders(data_folder=DATA_FOLDER)
    model, loss_criterion, optimizer = get_model(
        NUM_LABELS, lr=lr, momentum=momentum, adam_optimizer=use_adam, cross_entropy=use_ce)

    model=train_model(train_loader, val_loader, model, CURRENT_MODEL,
                loss_criterion, optimizer, epochs=epochs, store_files=False)
    
    top_1 = evaluate_model(CURRENT_MODEL, MODEL_FILENAME, data_folder=DATA_FOLDER, use_test_data=False)
    top_1 = float(top_1)

    os.remove(MODEL_FILENAME)

    hyperparameter_values += [(lr, momentum, weight_decay)]
    accuracies += [top_1]

    # Check if current model is better than the previous best
    if top_1 > best_accuracy:
        best_accuracy = top_1
        best_hyperparameters = (
            lr, momentum, weight_decay, epochs)
        print("At the moment best hyperparameters:", best_hyperparameters)


print(hyperparameter_values)
print(accuracies)

# Print the best hyperparameters and accuracy
print("Best Hyperparameters:", best_hyperparameters)
print("Best Accuracy:", best_accuracy)


plot_hyperparameters_summary()