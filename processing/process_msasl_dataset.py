# Import necessary libraries
import json
from data_constants import LABELS  # Import labels from a data_constants module

# Output file path for the converted dataset
OUTPUT_FILE = "config/MSASL.json"

# Input files for different sets from MSASL dataset
MSASL_TEST_JSON_FILE = "config/MSASL_test.json"
MSASL_TRAIN_JSON_FILE = "config/MSASL_train.json"
MSASL_VAL_JSON_FILE = "config/MSASL_val.json"

# List of input files and their corresponding set names
FILES = [
    (MSASL_TEST_JSON_FILE, "test"),
    (MSASL_TRAIN_JSON_FILE, "train"),
    (MSASL_VAL_JSON_FILE, "val")
]

# MSASL dataset field names
MSASL_LABEL = "clean_text"  # Label for sign language gestures in MSASL dataset
MSASL_FRAME_START = "start"  # Start frame of the sign gesture
MSASL_FRAME_END = "end"  # End frame of the sign gesture
START_TIME = "start_time"  # Start time of the sign gesture
END_TIME = "end_time"  # End time of the sign gesture
FPS = "fps"  # Frames per second
SIGNER_ID = "signer_id"  # ID of the signer
URL = "url"  # URL of the video

# WLASL dataset field names (desired format after conversion)
WLASL_LABEL = "gloss"  # Gloss for sign language gestures in WLASL dataset
WLASL_VIDEO_ID = "video_id"  # ID of the video
WLASL_FRAME_START = "frame_start"  # Start frame of the gesture in the video
WLASL_FRAME_END = "frame_end"  # End frame of the gesture in the video
WLASL_DATA_SET = "split"  # Dataset set name (e.g., "train", "val")
WLASL_INSTANCES = "instances"  # List of instances for each label in WLASL dataset

# Define a custom sorting function based on END_TIME


def sort_by_end_time(entry):
    return entry[END_TIME]

# Function to process MSASL dataset content and convert it to WLASL format


def get_content(output, input, set_name, start_id=0):
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
    # Initialize an empty dictionary to store the converted dataset
    content = {}
    last_video_id = 0

    # Loop through input files and their corresponding set names
    for file_name, set_name in FILES:
        with open(file_name, "r") as json_file:
            msasl_content = json.load(json_file)

        content, last_video_id = get_content(
            content, msasl_content, set_name, last_video_id)

    # Sort the dictionaries in the list based on END_TIME
    for label in content:
        content[label] = sorted(content[label], key=sort_by_end_time)

    final_content = []

    # Print the number of unique labels and missing labels
    print("Number of labels:", len(content.keys()))
    print("Labels missing:", [
          element for element in LABELS if element not in content.keys()])

    # Organize data into final format and write to the output file
    for label, instances in content.items():
        final_content += [{
            WLASL_LABEL: label,
            WLASL_INSTANCES: instances
        }]

    with open(OUTPUT_FILE, "w") as json_file:
        json.dump(final_content, json_file, indent=4)
