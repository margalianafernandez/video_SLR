# IMPORTS
import os
import json
import shutil
from numpy import Inf
from os import listdir
from os.path import join

VIDEO_EXTENTION = ".mp4"


# Files and dir paths
CURRENT_DIR = os.getcwd()
CONFIG_DIR = join(CURRENT_DIR, "config")
VIDEO_DIR = join(CURRENT_DIR, "videos")
OUTPUT_FILE = join(CONFIG_DIR, "dataset.json")
DIR_INFO_VIDEOS = join(CONFIG_DIR, "WLASL_v0.3.json")


# Types of datasets
TEST = "test"
TRAIN = "train"
VALIDATION = "val"

# Print messages
SHOW_DATASET_NAME = lambda name: "* Dataset: " + name
SHOW_OCCURRENCE = lambda min, max: "\tMin occurrence: " + str(min) + ". Max occurrence: " + str(max)


# SPLIT THE DATA IN TRAINING, TESTING AND VALIDATION

class ASL_Dataset:

    def __init__(self, dataset_type):
        self.ds_type = dataset_type
        self.data = []
        self.labels = []
        self.singers_id = []
    
    def add_sample(self, x, y, singer_id):
        self.data += [x]
        self.labels += [y]
        self.singers_id += [singer_id]


def process_data(data, video_files):
    """ 
    Read the data and store the name of each mp4 file and the label in the corresponding dataset.
    """
    train_ds = ASL_Dataset(TRAIN)
    test_ds = ASL_Dataset(TEST)
    val_ds = ASL_Dataset(VALIDATION)

    for element in data:
        gloss = element["gloss"]
        instances = element["instances"]

        for instance in instances:
            video_id = instance["video_id"]
            if video_id not in video_files: continue
            
            if instance["split"] == TRAIN:
                train_ds.add_sample(video_id, gloss, instance["signer_id"])

            elif instance["split"] == TEST:
                test_ds.add_sample(video_id, gloss, instance["signer_id"])
            
            elif instance["split"] == VALIDATION:
                val_ds.add_sample(video_id, gloss, instance["signer_id"])
    
    
    return train_ds, test_ds, val_ds


def get_occurrence_same_values(arr):
    """ Get the max and min number of times the same value is stored in the input array 
    """
    max_count, min_count = 0, Inf

    for val in set(arr):
        count = arr.count(val)
        max_count = max(max_count, count)
        min_count = min(min_count, count)

    return max_count, min_count


def compare_singers(dataset):
    """ Show the number of times that the same singer is representing the same label
    """
    max_occurrence, min_occurrence = 0, Inf

    for label in set(dataset.labels):
        singers = []

        for idx, actual_label in enumerate(dataset.labels):
            if actual_label != label: continue
            singers += [dataset.singers_id[idx]]

        occurrence = get_occurrence_same_values(singers)
        max_occurrence = max(max_occurrence, occurrence[0])
        min_occurrence = min(min_occurrence, occurrence[1])
    
    print(SHOW_DATASET_NAME(dataset.ds_type))
    print(SHOW_OCCURRENCE(min_occurrence, max_occurrence))

    
def compare_singers_between_sets(dataset1, dataset2):
    """ Compare if the same singer is representing the same label in different datasets 
    """
    max_occurrence, min_occurrence = 0, Inf
    times_occurence = 0

    for label in set(dataset1.labels):
        singers_label_set1, singers_label_set2 = set(), set()
        
        for idx, actual_label in enumerate(dataset1.labels):
            if actual_label != label: continue
            singers_label_set1.add(dataset1.singers_id[idx])
        
        for idx, actual_label in enumerate(dataset2.labels):
            if actual_label != label: continue
            singers_label_set2.add(dataset2.singers_id[idx])
            
        occurrence = len(singers_label_set1.intersection(singers_label_set2))
        if occurrence != 0: times_occurence += 1
        
        max_occurrence = max(max_occurrence, occurrence)
        min_occurrence = min(min_occurrence, occurrence)

    percentage = times_occurence*100/len(set(dataset1.labels))
    print(SHOW_DATASET_NAME(dataset1.ds_type + "-" + dataset2.ds_type))
    print(SHOW_OCCURRENCE(min_occurrence, max_occurrence))
    print("\tTimes occurrence:", times_occurence, ". Percentage:", f'{percentage:.2f}', "%")


def show_data_info():
    """ Show the ASL dataset relevant information
    """
    total_videos = len(video_files)
    datasets = [train_ds, test_ds, val_ds]

    print("Number of videos and occurrence of the same label in each set:")
    
    for dataset in datasets:
        percentage = len(dataset.data)*100/total_videos
        print(SHOW_DATASET_NAME(dataset.ds_type))
        print("\tAmount of data:", len(dataset.data), f'. Percentage: {percentage:.2f}', "%")
        
        max_count, min_count = get_occurrence_same_values(dataset.labels)
        print("\tNumber of different types of labels:", len(set(dataset.labels)))
        print("\tDifferent types of labels occurrence: [Min:", min_count,", Max:", max_count, "]")


    print("\nNumber of singers in each set:")
    for dataset in datasets:
        max_count, min_count = get_occurrence_same_values(dataset.singers_id)
        print(SHOW_DATASET_NAME(dataset.ds_type))
        print(SHOW_OCCURRENCE(min_count, max_count))
    
    print("\nOccurrence of singers for the same label in each set:")
    for dataset in datasets:
        compare_singers(dataset)

    print("\nOccurrence of singers for the same label in different sets:")
    ds_commparation = [(train_ds, test_ds), (train_ds, val_ds), (test_ds, val_ds), (val_ds, test_ds)]
    
    for dataset1, dataset2 in ds_commparation:
        compare_singers_between_sets(dataset1, dataset2)


### STORE THE PROCESSED INFORMATION

def get_dataset_json(dataset):
    """ Return the inforation that is going to be stored from the dataset entered
    """
    return {
        "data": dataset.data,
        "labels": dataset.labels
    }

def store_information():
    """ Store the infromation of each dataset
    """
    obj = {
        "train": get_dataset_json(train_ds),
        "val": get_dataset_json(val_ds),
        "test": get_dataset_json(test_ds)
    }

    with open(OUTPUT_FILE, "w") as out_file:
        json.dump(obj, out_file)


def move_videos_dataset(dataset):
    for video_name in dataset.data:
        old_path = join(DIR_DATA, video_name+VIDEO_EXTENTION)
        new_path = join(SPLIT_DATA_DIR, dataset.ds_type, video_name+VIDEO_EXTENTION)
        shutil.copy(old_path, new_path)


if __name__ == "__main__":

    video_files = [os.path.splitext(f)[0] for f in listdir(DIR_DATA)]

    with open(DIR_INFO_VIDEOS, 'r') as myfile:
        data = myfile.read()
        data = json.loads(data)
        train_ds, test_ds, val_ds = process_data(data, video_files)


    show_data_info()
    store_information()

    move_videos_dataset(test_ds)
    move_videos_dataset(train_ds)
    move_videos_dataset(val_ds)