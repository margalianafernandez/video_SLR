# Sign Language Recognition with 3DCNN and SlowFast Neural Network Models

## Abstract
This research project delves into the realm of Sign Language Recognition (SLR) employing advanced models, such as, the 3DCNN and the SlowFast Neural Network, with the overarching goal of improving accuracy. The study encompasses rigorous experimentation, focusing on data processing techniques and model optimisation. Under computational constraints, 3DCNN achieved an accuracy of 68. 25\%, and SlowFast neural network reached 66. 23\%. The project emphasises the significance of data-processing methods, showcasing the importance of features such as facial expressions and hand movements in sign interpretation. Additionally, a manually curated set of 50 labels, representing various aspects of sign language, was used for model training. This report not only highlights achievements within limitations but also lays the groundwork for future endeavours.


## Repository Structure
The repository structure is organized as follows:


### Models Implementation
- The model implementation is stored in the `model/` directory and comprises four essential files:
  - `cnn3d_model`: This Python file downloads the Facebook 3DCNN pre-trained model and prepares the dataset according to the model's input expectations.
  - `slowfast_model`: This Python file downloads the Facebook SlowFast Neural Network pre-trained model and prepares the dataset accordingly.
  - `ensemble`: This Python file defines the ensemble model structure, taking the output of three 3DCNN models as input for the MLP model.
  - `model_constants`: This file defines the constants used throughout the three model definitions.

### Data Processing Implementation
- The data processing implementation is located in the `Processing/` directory, consisting of the following files:
  - `labels`: This file contains the 50 labels used for training the model.
  - `data_constants`: A file containing constant variables used throughout the data processing.
  - `video_downloader`: This file reads from either the `config/WLASL_v0.3.json` or `config/MSASL.json` files, depending on the value of the constant `DATASET_SELECTED` in the `data_constants` file. It downloads sign videos containing the labels defined in `labels.py` and stores them in the `VIDEOS_FOLDER` as specified in `data_constants`.
  - `process_msasl_datasets`: For handling the MSASL dataset, which has a different structure than the WLASL dataset, this file reads from `config/MSASL_test.json`, `config/MSASL_train.json`, and `config/MSASL_val.json` files and converts the information into the WLASL format, storing it in `config/MSASL.json`.
  - `split_datasets`: This file analyzes the downloaded samples and generates two files in the `config/` folder to categorize them into different train, test, and validation sets.
  - `process_videos`: This file accepts the chosen processing method as input. It reads from the JSON file containing videos to process, defined in the `JOIN_DATASET_FILE_PATH` constant variable, and stores the processed videos in the `PROCESSED_VIDEO_FOLDER`.

### Model Training and Evaluation Files
- The following files are used to train and evaluate the models:
  - `train_model`: This file trains the model and stores its loss function and the final model in the `checkpoint` folder. It accepts several arguments:
    - `model`: Specifies the model to train (3DCNN or SlowFast).
    - `checkpoint`: Specifies the path to the folder to store the output files.
    - `data`: Specifies the folder containing the dataset, which should include two folders: `val` and `train`, with samples categorized in folders named after their labels.
    - `eval`: A boolean indicating whether to execute `evaluate_model.py` after training.
  - `train_ensemble_model`: This file trains the ensemble model and stores its loss function and final model in the `checkpoint` folder. It expects the trained models to be stored in the `models_3dcnn/` folder.
  - `evaluate_model`: This file tests the input model using the test data, returns its accuracy, and stores the confusion matrix in the `checkpoint` file. It accepts several arguments:
    - `model`: Specifies the model to evaluate (3DCNN or SlowFast).
    - `checkpoint`: Specifies the path to the folder to store the output files.
    - `data`: Specifies the folder containing the test dataset, which should contain a `test` folder with samples categorized in folders named after their labels.
    - `file`: Specifies the name and path of the model produced by `train_model.py`.
  - `evaluate_ensemble_model`: This file tests the ensemble model using the test data, returns its accuracy, and stores the confusion matrix in the `checkpoint` folder. It accepts several arguments:
    - `model`: Specifies the model to evaluate (3DCNN or SlowFast).
    - `checkpoint`: Specifies the path to the folder to store the output files.
    - `data`: Specifies the folder containing the test dataset, which should contain a `test` folder with samples categorized in folders named after their labels.
    - `file`: Specifies the name and path of the model produced by `train_ensemble_model.py`.



## Example of Execution

Example of executing commands to train and evaluate the 3DCNN model:

### Download and Process Datasets

Example of executing commands to download and process video samples from the WLASL dataset:

```shell
python3.9 processing/video_downloader.py
python3.9 processing/process_videos.py --type face_and_hands
python3.9 processing/split_datasets.py
```

### Train and Evaluate a 3DCNN

Example of an execution to Train and evaluate the 3DCNN model:

To train and evaluate with evaluation
````
python3.9 train_model.py --model 3dcnn --eval True --data "/Document/processed_data" --checkpoint "/Documents/check_points"
````

To train without evaluation
````
python3.9 train_model.py --model 3dcnn --eval False --data "/Document/processed_data" --checkpoint "/Documents/check_points"
````

To evaluate the trained model
````
python3.9 evaluate_model.py --model 3dcnn --data "/Document/processed_data" --checkpoint "/Documents/check_points" --file "/Documents/check_points/3dcnn_model.pth"
````



#### Train and Evaluate a SlowFast Neural Network Model

Train and evaluate the SlowFast model:

````
python3.9 train_model.py --model slowfast --eval True --data "/Document/processed_data" --checkpoint "/Documents/check_points"
````


#### Train and Evaluate the Ensemble Model

Example of an execution training and evaluating an ensemble model
Train and evaluate the ensemble model:

````
python3.9 train_model_ensemble.py --eval True
````