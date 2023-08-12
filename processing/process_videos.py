import json
import argparse
import cv2 as cv
import numpy as np
import mediapipe as mp
import face_recognition
from data_constants import *


class VideoBodyParts():
    """
    This class will detect the face and the hands of the input frame and store it
    """

    def __init__(self, type):
        """
        Initialize the VideoBodyParts class.

        Args:
            type (ProcessingType): Type of processing to perform (ALL, HANDS, FACE_HANDS, BODY_HANDS).
        """
        empty_image = np.zeros(
            (TARGET_SIZE, TARGET_SIZE, TARGET_CHANNELS), dtype=np.uint8)
        self.type = type
        self.face_im = empty_image
        self.hole_im = empty_image.copy()
        
        if type == ProcessingType.HANDS:
            self.left_hand_im = np.zeros(
                (FRAME_SIZE, TARGET_SIZE, TARGET_CHANNELS), dtype=np.uint8)
            self.right_hand_im = self.left_hand_im.copy()
        else: 
            self.left_hand_im = empty_image.copy()
            self.right_hand_im = empty_image.copy()
    

    @staticmethod
    def calculate_bounding_box(hand_landmarks, frame_width, frame_height):
        """
        Given the hands landmarks, get the box size dimensions of the hand in the original image
        """
        x_min, y_min,  x_max, y_max = frame_width, frame_height,  0, 0

        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)

            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y

        increment = max(x_max-x_min, y_max-y_min)
        return x_min, y_min, x_min+increment, y_min+increment

    def detect_hands_in_image(self, mp_hands, frame, width, height):
        """
        Detect the left and right hands in the image and store it in the self.left_hand_im and
        self.right_hand_im variables.
        """
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results_hands = mp_hands.process(frame)
        image_heigh = (TARGET_SIZE, FRAME_SIZE)[self.type== ProcessingType.HANDS]
        
        if results_hands.multi_hand_landmarks:

            for aux, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):

                # Calculate the bounding box coordinates around the hand
                x_min, y_min, x_max, y_max = VideoBodyParts.calculate_bounding_box(
                    hand_landmarks, width, height)
                cropped_im = frame[y_min:y_max, x_min:x_max]
                resized_image = cv.resize(
                    cropped_im, (TARGET_SIZE, image_heigh))
                resized_image = cv.cvtColor(resized_image, cv.COLOR_RGB2BGR)
                if results_hands.multi_handedness[aux].classification[0].label == LEFT_HAND_LABEL:
                    self.left_hand_im = resized_image

                else:
                    self.right_hand_im = resized_image

    def detect_face_in_image(self, frame):
        """
        Detect the face the image and store it in the self.face_im variable.
        """
        width = (FRAME_SIZE, TARGET_SIZE)[self.type == ProcessingType.ALL]

        # Detect faces in the image
        face_locations = face_recognition.face_locations(frame)

        # If a face is detected, crop the image to show only the face
        if len(face_locations) > 0:
            # Assume the first face found is the most prominent
            top, right, bottom, left = face_locations[0]
            cropped_image = frame[top:bottom, left:right]
            self.face_im = cv.resize(cropped_image, (width, TARGET_SIZE))

    def store_hole_im(self, frame):
        """
        Store the hole image, resized and blured, in the hole_im variable.
        """
        width = (FRAME_SIZE, TARGET_SIZE)[self.type == ProcessingType.ALL]

        blur_image = cv.GaussianBlur(frame, (15, 15), 0)
        self.hole_im = cv.resize(blur_image, (width, TARGET_SIZE))


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_image_regions(video_parts, type):
    """
    Create a combined image of four regions. The output image will be as follows:
    if type == ALL: the same video divided in four regions:
    | Left hand | Right hand |
    |   Face    | Hole image |

    elif type == HANDS: the same video divided in two regions:
    | Left hand | Right hand |

    elif type == FACE_HANDS: the same video divided in three regions:
    | Left hand | Right hand |
    |           Face         |

    elif type == BODY_HANDS: the same video divided in three regions:
    | Left hand | Right hand |
    |        Hole image      |
    """
    combined_hands_image = np.hstack(
        (video_parts.left_hand_im, video_parts.right_hand_im))

    if type == ProcessingType.ALL:
        combined_body_image = np.hstack(
            (video_parts.face_im, video_parts.hole_im))
        combined_image = np.vstack((combined_hands_image, combined_body_image))

    elif type == ProcessingType.BODY_HANDS:
        combined_image = np.vstack(
            (combined_hands_image,  video_parts.hole_im))

    elif type == ProcessingType.FACE_HANDS:
        combined_image = np.vstack((combined_hands_image, video_parts.face_im))

    else:
        combined_image = combined_hands_image

    return combined_image


def process_video(type, video_path, output_path):
    """
    Detect the face and hands of the input video path and return:
    if type == ALL: the same video divided in four regions:
    | Left hand | Right hand |
    |   Face    | Hole image |

    elif type == HANDS: the same video divided in two regions:
    | Left hand | Right hand |

    elif type == FACE_HANDS: the same video divided in three regions:
    | Left hand | Right hand |
    |           Face         |

    elif type == BODY_HANDS: the same video divided in three regions:
    | Left hand | Right hand |
    |        Hole image      |
    """
    video_parts = VideoBodyParts(type)

    cap = cv.VideoCapture(video_path)

    width, height = (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)))

    fps = int(cap.get(cv.CAP_PROP_FPS))
    fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')

    out_video = cv.VideoWriter()
    out_video.open(output_path, fourcc, fps, (FRAME_SIZE, FRAME_SIZE), True)

    mp_hands = mp.solutions.hands.Hands(
        max_num_hands=MAX_NUM_HANDS,
        min_detection_confidence=MIN_DETECTION_COFIDENCE,
        min_tracking_confidence=MIN_TRACKING_COFIDENCE)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        try:
            video_parts.detect_hands_in_image(mp_hands, frame, width, height)

            if type == ProcessingType.ALL or type == ProcessingType.FACE_HANDS:
                video_parts.detect_face_in_image(frame)

            if type == ProcessingType.ALL or type == ProcessingType.BODY_HANDS:
                video_parts.store_hole_im(frame)

            combined_image = get_image_regions(video_parts, type)
            out_video.write(combined_image)

        except Exception as error:
            print("Error in the file " + video_path + ". Error message", error)

    # Release resources
    cap.release()
    out_video.release()


def process_video_and_store(processing_type):

    num_videos = 0
    dataset_content = json.load(open(DATASET_FILE))

    for folder in dataset_content:

        for label in dataset_content[folder]:

            # label_folder = os.path.join(PROCESSED_VIDEO_FOLDER, folder, label)
            label_folder = os.path.join(PROCESSED_VIDEO_FOLDER, "train", label)
            create_folder_if_not_exists(label_folder)

            for video in dataset_content[folder][label]:

                video_id = video["video_id"]

                input_path = os.path.join(
                    VIDEOS_FOLDER, video_id + FILES_EXTENSION)
                output_path = os.path.join(
                    label_folder, video_id + FILES_EXTENSION)
                process_video(processing_type, input_path, output_path)

                num_videos += 1

    print("Num videos", num_videos)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train video classification model.")

    parser.add_argument("--type", type=ProcessingType, default=ProcessingType.HANDS,
                        help="Name of the model to train: " + ", ".join([x.value for x in ProcessingType]))

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    print("Type selected:", args.type.value)
    process_video_and_store(args.type)
