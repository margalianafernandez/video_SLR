import os
import json
from os.path import join, splitext
from processing.data_constants import *
from models.model_constants import PROCESSED_VIDEO_FOLDER

# Specify the datasets to be converted
ACTUAL_VAL_TEST_DATASET = DatasetSelected.MSASL
ACTUAL_TRAIN_DATASET = DatasetSelected.WLASL

# Define the path to the split dataset file
SPLIT_DATASET_FILE_PATH = join(CONFIG_PATH, "{}_test_val_dataset_split.json")


def convert_from_val_and_test_to_train():
    """
    Convert video assignment from validation and test sets to the train set.
    """
    split_dataset = {}

    # Iterate through TEST and VALIDATION sets
    for set_name in [TEST, VALIDATION]:
        split_dataset[set_name] = {}
        set_path = join(PROCESSED_VIDEO_FOLDER, set_name)

        # Iterate through labels in the set
        for label in os.listdir(set_path):
            label_path = join(set_path, label)
            # Store video names without extension for each label
            split_dataset[set_name][label] = [
                splitext(file_name)[0] for file_name in os.listdir(label_path)]

    # Write the split dataset to a JSON file
    file_path = SPLIT_DATASET_FILE_PATH.format(ACTUAL_VAL_TEST_DATASET.value)
    with open(file_path, "w") as json_file:
        json.dump(split_dataset, json_file, indent=4)


def convert_from_test_to_val_and_test():
    """
    Convert video assignment from test set to validation and test sets.
    """
    # Initialize a dictionary to keep track of validation video counts
    val_videos = {}
    dataset_file = join(CONFIG_PATH, "dataset_{}.json".format(
        ACTUAL_TRAIN_DATASET.value))
    dataset_content = json.load(open(dataset_file))

    # Iterate through labels to set initial validation video counts
    for label in LABELS:
        for folder in [TEST, VALIDATION]:
            num_val_videos = sum([len(dataset_content[set_name][label])
                                  for set_name in SETS if label in dataset_content[set_name]])

            val_videos[label] = {
                "max": num_val_videos // 2,
                "counter": 0
            }

    # Initialize dictionaries to store the split dataset
    split_dataset = {TEST: {}, VALIDATION: {}}

    # Iterate through dataset content and assign videos to validation or test set
    for folder in dataset_content:
        for label in dataset_content[folder]:
            for video in dataset_content[folder][label]:

                # Determine whether video should go to validation or test set
                video_set = (VALIDATION, TEST)[
                    val_videos[label]["counter"] > val_videos[label]["max"]]

                if label not in split_dataset[video_set]:
                    split_dataset[video_set][label] = []
                split_dataset[video_set][label] += [video['video_id']]
                val_videos[label]["counter"] += 1

    # Write the split dataset to a JSON file
    file_path = SPLIT_DATASET_FILE_PATH.format(ACTUAL_TRAIN_DATASET.value)
    with open(file_path, "w") as json_file:
        json.dump(split_dataset, json_file, indent=4)


def flip_data_set_assignation():
    """
    Flip the assignment of videos between train and validation/test sets.
    """
    # Load split dataset information
    train_content = json.load(
        open(SPLIT_DATASET_FILE_PATH.format(ACTUAL_TRAIN_DATASET.value)))
    test_val_content = json.load(
        open(SPLIT_DATASET_FILE_PATH.format(ACTUAL_VAL_TEST_DATASET.value)))

    # Rename the old processed video folder and create a new one
    old_process_folder = PROCESSED_VIDEO_FOLDER+"_tmp"
    os.rename(PROCESSED_VIDEO_FOLDER, old_process_folder)
    os.mkdir(PROCESSED_VIDEO_FOLDER)

    # Create folders for each set and label
    for set_name in SETS:
        os.mkdir(join(PROCESSED_VIDEO_FOLDER, set_name))
        for label in LABELS:
            os.mkdir(join(PROCESSED_VIDEO_FOLDER, set_name, label))

    # Move videos from test/val to train folders based on updated assignments
    for set_name, labels in test_val_content.items():
        for label, file_names in labels.items():
            for file_name in file_names:
                source_file = join(old_process_folder, set_name,
                                   label, file_name+FILES_EXTENSION)
                destination_file = join(
                    PROCESSED_VIDEO_FOLDER, TRAIN, label, file_name+FILES_EXTENSION)
                os.rename(source_file, destination_file)

    # Move videos from train to test/val folders based on updated assignments
    for set_name, labels in train_content.items():
        for label, file_names in labels.items():
            for file_name in file_names:
                source_file = join(old_process_folder, TRAIN,
                                   label, file_name+FILES_EXTENSION)
                destination_file = join(
                    PROCESSED_VIDEO_FOLDER, set_name, label, file_name+FILES_EXTENSION)
                
                if not os.path.exists(source_file):
                    print("File:", source_file, "does not  exist")
                    continue

                os.rename(source_file, destination_file)

    os.remove(old_process_folder)
    

if __name__ == "__main__":
    # Execute the assignment flip function
    flip_data_set_assignation()
