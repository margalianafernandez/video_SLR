import os
import sys
import json
import shutil
import random
from data_constants import *
from os.path import join, splitext


# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Calculate the parent directory and add it to sys.path
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)
from models.model_constants import PROCESSED_VIDEO_FOLDER


def divide_dataset_into_test_and_val():
    """
    The dataset that will be divided is assigned in the data_constants file in the 
    variable called DATASET_FILE. This function will read all the videos in the dataset 
    and write a json file called START_SPLIT_DATASET_FILE_PATH.
    """
    # Initialize a dictionary to keep track of validation video counts
    val_videos = {}
    dataset_content = json.load(open(DATASET_FILE))

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
    with open(SPLIT_DATASET_FILE_PATH, "w") as json_file:
        json.dump(split_dataset, json_file, indent=4)


def split_train_test_from_two_datasets():
    msasl_content = json.load(
        open(START_SPLIT_DATASET_FILE_PATH.format(DatasetSelected.MSASL.value)))
    wlasl_content = json.load(
        open(START_SPLIT_DATASET_FILE_PATH.format(DatasetSelected.WLASL.value)))

    msasl_split = {VALIDATION: {}, TEST: {}, TRAIN: {}}
    wlasl_split = {VALIDATION: {}, TEST: {}, TRAIN: {}}

    for label in LABELS:
        msasl_split[TEST][label], msasl_split[VALIDATION][label] = [], []
        wlasl_split[TEST][label], wlasl_split[VALIDATION][label] = [], []

        for content, dts_split in [(msasl_content, msasl_split), (wlasl_content, wlasl_split)]:
            all_files = content[TEST][label] + content[VALIDATION][label]
            random.shuffle(all_files)
            num_files = len(all_files)
            test_num = int(num_files*TEST_RATE)
            val_num = int(num_files*VALIDATION_RATE)
            train_num = num_files - test_num - val_num

            dts_split[VALIDATION][label] = all_files[:val_num]
            dts_split[TRAIN][label] = all_files[val_num:val_num+train_num]
            dts_split[TEST][label] = all_files[val_num+train_num:]

    content = {
        DatasetSelected.WLASL.value: wlasl_split,
        DatasetSelected.MSASL.value: msasl_split
    }

    with open(JOIN_DATASET_FILE_PATH, "w") as json_file:
        json.dump(content, json_file, indent=4)


def validate_dataset_swap(test_val_content, train_content):

    # Move videos from test/val to train folders based on updated assignments
    for label in LABELS:

        # Check new TRAIN content
        correct_files = train_content[TEST][label] + train_content[VALIDATION][label] 

        directory_path = join(PROCESSED_VIDEO_FOLDER, TRAIN, label)
        directory_files = os.listdir(directory_path)
        actual_files = [os.path.splitext(filename)[0]
                        for filename in directory_files]

        if sorted(actual_files) != sorted(correct_files):
            print("From the new TRAIN content, label:", label, "is incorrect.")

        # Check new TEST/VAL content
        for set_name in [TEST, VALIDATION]:
            correct_files = test_val_content[set_name][label]
            directory_path = join(PROCESSED_VIDEO_FOLDER, set_name, label)
            directory_files = os.listdir(directory_path)
            actual_files = [os.path.splitext(filename)[0]
                            for filename in directory_files]

            if sorted(actual_files) != sorted(correct_files):
                print("From the new TEST/VAL content, label:",
                      label, " and set", set_name, "is incorrect.")


def set_each_dataset_to_train_or_test():
    """
    Flip the assignment of videos between train and validation/test sets.
    """
    # Load split dataset information
    train_content = json.load(
        open(START_SPLIT_DATASET_FILE_PATH.format(TRAIN_DATASET.value)))
    test_val_content = json.load(
        open(START_SPLIT_DATASET_FILE_PATH.format(VAL_TEST_DATASET.value)))

    # Rename the old processed video folder and create a new one
    if os.path.exists(PROCESSED_VIDEO_FOLDER):
        shutil.rmtree(PROCESSED_VIDEO_FOLDER)

    # Create folders for each set and label
    os.mkdir(PROCESSED_VIDEO_FOLDER)
    for set_name in SETS:
        os.mkdir(join(PROCESSED_VIDEO_FOLDER, set_name))
        for label in LABELS:
            os.mkdir(join(PROCESSED_VIDEO_FOLDER, set_name, label))

    iterate = [(train_content, TRAIN_DATASET, True),
               (test_val_content, VAL_TEST_DATASET, False)]

    for content, dataset, is_train in iterate:
        for set_name, labels in content.items():
            set_name = (set_name, TRAIN)[is_train]
            for label, files in labels.items():
                for file_name in files:
                    source_path = join(START_PROCESSED_VIDEO_FOLDER.format(
                        dataset.value), file_name+FILES_EXTENSION)
                    dest_path = join(
                        PROCESSED_VIDEO_FOLDER, set_name, label)
                    shutil.copy(source_path, dest_path)

    validate_dataset_swap(test_val_content, train_content)


def join_both_datasets_into_train_or_test():
    """
    Flip the assignment of videos between train and validation/test sets.
    """
    # Load split dataset information
    join_dataset_content = json.load(open(JOIN_DATASET_FILE_PATH))

    # Rename the old processed video folder and create a new one
    if os.path.exists(PROCESSED_VIDEO_FOLDER):
        shutil.rmtree(PROCESSED_VIDEO_FOLDER)

    # Create folders for each set and label
    os.mkdir(PROCESSED_VIDEO_FOLDER)
    for set_name in SETS:
        os.mkdir(join(PROCESSED_VIDEO_FOLDER, set_name))
        for label in LABELS:
            os.mkdir(join(PROCESSED_VIDEO_FOLDER, set_name, label))

    for dataset_name, dataset in join_dataset_content.items():
        for set_name, labels in dataset.items():
            for label, files in labels.items():
                for file_name in files:
                    source_path = join(START_PROCESSED_VIDEO_FOLDER.format(
                        dataset_name), file_name+FILES_EXTENSION)
                    dest_path = join(
                        PROCESSED_VIDEO_FOLDER, set_name, label)
                    shutil.copy(source_path, dest_path)


if __name__ == "__main__":
    # divide_dataset_into_test_and_val()
    set_each_dataset_to_train_or_test()
