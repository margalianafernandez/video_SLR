import json
import itertools
from collections import Counter
from processing.data_constants import DATASET_FILE, TRAIN, TEST, VALIDATION, SETS


def read_dataset():
    return json.load(open(DATASET_FILE))    


def evaluate_dataset_split(dataset_content):
    train_rate, test_rate, val_rate = 0.0, 0.0, 0.0

    labels = dataset_content[TRAIN]

    for label in labels:
        total_samples = sum(len(dataset_content[data_set][label]) for data_set in SETS)
        
        train_rate += len(dataset_content[TRAIN][label]) / total_samples
        test_rate += len(dataset_content[TEST][label]) / total_samples
        val_rate += len(dataset_content[VALIDATION][label]) / total_samples

    num_labels = len(labels)
    train_rate = 100 * train_rate/num_labels
    test_rate = 100 * test_rate/num_labels
    val_rate = 100 * val_rate/num_labels

    print("\n\nDATA SETS RATE")
    print(f"* Train set: {train_rate:.2f}%")
    print(f"* Test set: {test_rate:.2f}%")
    print(f"* Validation set: {val_rate:.2f}%")


def evaluate_singers(content):

    print("SINGERS EVALUATION FOR EACH LABEL IN THE DIFERENT SETS")
    
    labels = dataset_content[TRAIN]

    for label in labels:
        
        signers = dict()
        total_singers = set()

        for data_set in SETS:
            signers[data_set] = [sign["signer_id"] for sign in content[data_set][label]]
            total_singers = total_singers.union(set(signers[data_set]))
        
        print("- Label:", label, ". Total num singers:", len(total_singers))

        for set1, set2 in list(itertools.combinations(SETS, 2)):
            print("\t* Compare", set1, "(" + str(len(set1)) + ")", "-", set2, "(" + str(len(set2)) + ")", end=": ")

            # Count occurrences of each element in both lists
            counter_set1 = Counter(signers[set1])
            counter_set2 = Counter(signers[set2])

            # Find the intersection of elements between the two lists
            common_elements = counter_set1.keys() & counter_set2.keys()

            print(len(common_elements))


if __name__ == "__main__":
    
    dataset_content = read_dataset()
    evaluate_singers(dataset_content)
    evaluate_dataset_split(dataset_content)
