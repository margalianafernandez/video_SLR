# Import necessary libraries
from train_model import *
from itertools import product
import matplotlib.pyplot as plt
from evaluate_model import evaluate_model
from processing.data_constants import *

LABELS = face_motion_labels + hands_motion_labels + body_motion_labels
NUM_LABELS = len(LABELS)


def plot_hyperparameters_summary(logs):
    
    # Separate data into two lists: slowfast_logs and cnn_logs
    logs_acc = [entry['acc'] for entry in logs]

    # Create a dictionary to map parameter names to their values in the logs
    param_names = {
        0: 'Learning Rate',
        1: 'Momentum',
        2: 'Weight Decay',
        3: 'Optimizer',
        4: 'Loss Function',
    }

    # Iterate through each parameter and create a plot
    for param_idx, param_name in param_names.items():
        
        param_values = [entry['param'][param_idx] for entry in logs]

        if param_idx == 3:
            param_values = ["Adam" if value else "SDG" for value in param_values]
        
        if param_idx == 4:
            param_values = ["Cross Entropy" if value else "Log-Likelihood" for value in param_values]        

        # Get the corresponding parameter value for the highest accuracy
        idx_max = logs_acc.index(max(logs_acc))
        highest_param_value = param_values[idx_max]

        # Create a new figure and axis for each parameter
        plt.figure()
        plt.title(f'Effect of {param_name} on Accuracy')
        plt.xlabel(param_name)
        plt.ylabel('Accuracy')

        # Plot slowfast and cnn data points separately
        plt.scatter(param_values, logs_acc, label=CURRENT_MODEL.label, marker='o')

        # Mark the highest accuracy values with red points
        plt.scatter(highest_param_value, logs_acc[idx_max], c='m', marker='x', s=100, label='Highest')

        # Add legend
        plt.legend()

        # Show or save the plot (you can modify this part based on your preference)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plots/plot_{}.png".format(param_name))
        plt.show()



CURRENT_MODEL = Models.CNN_3D

print("Tunning", CURRENT_MODEL.value)

# Define ranges for hyperparameters

learning_rate_range = [0.001, 0.01, 0.1]
momentum_range = [0.4, 0.6, 0.9]
weight_decay_range = [0.0001, 0.001, 0.01]
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
    adam_optimizer,
    cross_entropy_lf,
)

logs = []

for lr, momentum, weight_decay, use_adam, use_ce in hyperparam_comb:

    print("NEW TEST:")
    print("\tLR:", lr)
    print("\tMOMENTUM:", momentum)
    print("\tWEIGHT DECAY:", weight_decay)
    print("\tWEIGHT DECAY:", weight_decay)
    print("\tADAM OPTIMIZER:", use_adam)
    print("\CROSS ENTROPY LOSS FUNCTION:", use_ce)


    # Train the model using the current hyperparameters
    train_loader, val_loader = get_data_loaders(data_folder=PROCESSED_VIDEO_FOLDER)
    model, loss_criterion, optimizer = get_model(
        NUM_LABELS, lr=lr, momentum=momentum, adam_optimizer=use_adam, cross_entropy=use_ce)

    model=train_model(train_loader, val_loader, model, CURRENT_MODEL,
                loss_criterion, optimizer, store_files=False)
    
    top_1 = evaluate_model(CURRENT_MODEL, MODEL_FILENAME, data_folder=PROCESSED_VIDEO_FOLDER, use_test_data=False)
    top_1 = float(top_1)

    os.remove(MODEL_FILENAME)

    logs += [{
        'param': [lr, momentum, weight_decay],
        'acc': top_1
    }]

    # Check if current model is better than the previous best
    if top_1 > best_accuracy:
        best_accuracy = top_1
        best_hyperparameters = (lr, momentum, weight_decay, use_adam, use_ce)
        print("At the moment best hyperparameters:", best_hyperparameters)



# Print the best hyperparameters and accuracy
print("Best Hyperparameters:", best_hyperparameters)
print("Best Accuracy:", best_accuracy)


plot_hyperparameters_summary(logs)