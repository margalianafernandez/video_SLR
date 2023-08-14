"""
This script converts MSASL dataset information into the format of the WLASL dataset.
It prepares the data for processing in the video_downloader file.
"""

import json
from data_constants import LABELS

# Output file path
OUTPUT_FILE = "config/MSASL.json"

# Input files for different sets
MSASL_TEST_JSON_FILE = "config/MSASL_test.json"
MSASL_TRAIN_JSON_FILE = "config/MSASL_train.json"
MSASL_VAL_JSON_FILE = "config/MSASL_val.json"

# List of input files and corresponding set names
FILES = [
    (MSASL_TEST_JSON_FILE, "test"),
    (MSASL_TRAIN_JSON_FILE, "train"),
    (MSASL_VAL_JSON_FILE, "val")
]

# MSASL dataset field names
MSASL_LABEL = "clean_text"
MSASL_FRAME_START = "start"
MSASL_FRAME_END = "end"
START_TIME = "start_time"
END_TIME = "end_time"
FPS = "fps"
SIGNER_ID = "signer_id"
URL = "url"

# WLASL dataset field names
WLASL_LABEL = "gloss"
WLASL_VIDEO_ID = "video_id"
WLASL_FRAME_START = "frame_start"
WLASL_FRAME_END = "frame_end"
WLASL_DATA_SET = "split"
WLASL_INSTANCES = "instances"


# Define a custom sorting function
def sort_by_end_time(entry):
    return entry[END_TIME]


def get_content(output, input, set_name, start_id = 0):
    """
    Process the MSASL dataset content and convert it to the WLASL format.

    Args:
        output (dict): Dictionary to store the converted data.
        input (list): List of sign data from the MSASL dataset.
        set_name (str): Name of the dataset set (e.g., "train", "val").

    Returns:
        dict: Updated dictionary with converted data.
    """
    video_id = start_id + 1
    
    for sign in input:
        label = sign[MSASL_LABEL]

        if label not in LABELS:
            continue

        entry = {
            WLASL_VIDEO_ID: str(video_id),
            WLASL_FRAME_START: sign[MSASL_FRAME_START],
            WLASL_FRAME_END: sign[MSASL_FRAME_END],
            FPS: sign[FPS],
            URL: sign[URL],
            SIGNER_ID: sign[SIGNER_ID],
            WLASL_DATA_SET: set_name,
            END_TIME: sign[END_TIME],
            START_TIME: sign[START_TIME]
        }

        if label not in output:
            output[label] = []
        
        output[label] += [entry]
        video_id += 1

    return output, video_id


if __name__ == "__main__":
    content = {}
    last_video_id = 0
    
    # Loop through input files and their corresponding set names
    for file_name, set_name in FILES:
        with open(file_name, "r") as json_file:
            msasl_content = json.load(json_file)

        content, last_video_id = get_content(content, msasl_content, set_name, last_video_id)

    # Sort the dictionaries in the list based on END_TIME
    for label in content:
        content[label] = sorted(content[label], key=sort_by_end_time)

    final_content = []

    # Print the number of unique labels and missing labels
    print("Number of labels:", len(content.keys()))
    print("Labels missing:", [
          element for element in LABELS if element not in content.keys()])

    # Organize data into final format and write to output file
    for label, instances in content.items():
        final_content += [{
            WLASL_LABEL: label,
            WLASL_INSTANCES: instances
        }]

    with open(OUTPUT_FILE, "w") as json_file:
        json.dump(final_content, json_file, indent=4)
