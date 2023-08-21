# Import necessary libraries
from train_model import *
from itertools import product

print("Tunning", CURRENT_MODEL.value)

# Define ranges for hyperparameters
learning_rate_range = [0.001, 0.01, 0.02, 0.1, 0.2, 0.3]
momentum_range = [0.4, 0.6, 0.9]
weight_decay_range = [1e-2, 0.001, 1e-4, 1e-5, 0]
epochs_range = [10, 20, 30, 50]
cross_entropy_lf = [False, True]
adam_optimizer = [False, True]

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

for lr, momentum, weight_decay, epochs, using_adam, using_ce in hyperparam_comb:

    print("NEW TEST:")
    print("\tLR:", lr)
    print("\tMOMENTUM:", momentum)
    print("\tWEIGHT DECAY:", weight_decay)
    print("\tEPOCHS RANGE:", epochs)
    print("\tUSING ADAM OPTIMIZER:", using_adam)
    print("\tUSING CE LOSS FUNCTION:", using_ce)

    # Modify your model creation and training code here
    # Update hyperparameters

    try:
        # Train the model using the current hyperparameters
        train_loader, val_loader = get_data_loaders()
        model, loss_criterion, optimizer = get_model(
            NUM_LABELS, lr=lr, momentum=momentum, adam_optimizer=using_adam, cross_entropy=using_ce)

        top_1 = train_model(train_loader, val_loader, model, CURRENT_MODEL,
                    loss_criterion, optimizer, epochs=epochs, store_files=False)

        top_1 = float(top_1)

        # Check if current model is better than the previous best
        if top_1 > best_accuracy:
            best_accuracy = top_1
            best_hyperparameters = (
                lr, momentum, weight_decay, epochs, using_adam, using_ce)
            print("At the moment best hyperparameters:", best_hyperparameters)

    except Exception as e:
        print("An error occurred with this combination. Error:", e)


# Print the best hyperparameters and accuracy
print("Best Hyperparameters:", best_hyperparameters)
print("Best Accuracy:", best_accuracy)
