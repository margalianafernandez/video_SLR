# Sign Language Recognition with 3DCNN and SlowFast Neural Network Models

## Abstract
This research project delves into the realm of Sign Language Recognition (SLR) employing advanced models, such as, the 3DCNN and the SlowFast Neural Network, with the overarching goal of improving accuracy. The study encompasses rigorous experimentation, focusing on data processing techniques and model optimisation. Under computational constraints, 3DCNN achieved an accuracy of 68. 25\%, and SlowFast neural network reached 66. 23\%. The project emphasises the significance of data-processing methods, showcasing the importance of features such as facial expressions and hand movements in sign interpretation. Additionally, a manually curated set of 50 labels, representing various aspects of sign language, was used for model training. This report not only highlights achievements within limitations but also lays the groundwork for future endeavours.


## Repository Structure
The implementation of this code is divided in four blocks:

- The models implementation: it is stored in the model/ directory which contains four files:
    * cnn3d_model: this python file downloads the Facebook 3DCNN pre-trained model and loads the dataset as the model is expecing it to be input.
    * slowfast_model: this python file downloads the Facebook SlowFast Neural Network pre-trained model and loads the dataset as the model is expecing it to be input.
    * ensemble: this python file defines the ensemble model structure. Having the output of the three 3DCNN models as the input to the MLP model.
    * model_constants: this file defines the constants used throwout the three models definitions.


- The data processing implementation is locades in the Processing/ directory files, which cotains the following files:
    * labels: this file contains the 50 labels with which the model will be trained.
    * data_constants: file containing the constant variables used throwout the processing data process.
    * video_downloader: this file reads from the config/WlASL_v0.3.json or the config/MSASL.json files, depending on the value of the cosnatnt DATASET_SELECTED of the data_constants file. It will download the sign videos cntaing the labels of the labels.py file. They will be stored in the VIDEOS_FOLDER, defined on the data_constants file.
    * process_msasl_datasets: The MSASL dataset json files are structure in a different way than the WLASL dataset. This file will read from the config/MSASL_test.json, config/MSASL_train.json and config/MSASL_val.json files and write the config/MSASL.json file containing the same information but in the WLASL format.
    * split_datasets:This file will analyse the samples downloaded and create two files in the config/ folder in order to place them in the diferent train, test and validation sets. 
    * process_videos: This file will have as an input the processing method choosen. It will read from the json file the videos to process, defined in the JOIN_DATASET_FILE_PATH  constant variable. The processed videos will be stored in the PROCESSED_VIDEO_FOLDER folder.


- The following files are used to train and evaluate the models:
    * train_model: This files will train the model and store its loss function and final model in the checkpoint folder. It has several argumments:
        * model: defines the model to train: slowfast or 3DCNN
        * checkpoint: path to the folder to store the output files
        * data: the folder containing the dataset. This needs to contain two folders: val and train. In each one of them the samples need to be stored categorized in folders named as their labels.
        * eval: a boolean indicating if after the training the evaluate_model.py should be executed.
    * train_ensemble_model: This files will train the ensemble model and store its loss function and final model in the checkpoint folder. It is expecting the trained models to be stored in the models_3dcnn/ folder.
    * evaluate_model: This file will test the input model using the test data, return its accuracy and store the confusion matrix in the checkpoint file. It has several argumments:
        * model: defines the model to train: slowfast or 3DCNN
        * checkpoint: path to the folder to store the output files
        * data: the folder containing the dataset. This needs to contain the folder "test". In needs to contain the samples categorized in folders named as their labels.
        * file: the name and path of the model produced by the train_model.py file.
    * evaluate_ensemble_model: This file will test the ensemble model using the test data, return its accuracy and store the confusion matrix in the checkpoint file. It has several argumments:
        * model: defines the model to train: slowfast or 3DCNN
        * checkpoint: path to the folder to store the output files
        * data: the folder containing the dataset. This needs to contain the folder "test". In needs to contain the samples categorized in folders named as their labels.
        * file: the name and path of the model produced by the train_ensemble_model.py file.


## Exemple of execution

An example of the execution of this project would be:


#### Download and process datasets

Example of an execute to download and process the video samples of the WLASL dataset:

````
python3.9 processing/video_downloader.py
python3.9 processing/process_videos.py --type face_and_hands
python3.9 processing/split_datasets.py
````


#### Train and evaluate a 3DCNN

Example of an execute to Ttain and evaluate the 3DCNN model:

````
python3.9 train_model.py --model 3dcnn --eval True --data "/Document/processed_data" --checkpoint "/Documents/check_points"
````
or 

````
python3.9 train_model.py --model 3dcnn --eval False --data "/Document/processed_data" --checkpoint "/Documents/check_points"

python3.9 evaluate_model.py --model 3dcnn --data "/Document/processed_data" --checkpoint "/Documents/check_points" --file "/Documents/check_points/3dcnn_model.pth"
````


#### Train and evaluate a SlowFast neural network model:

Train and evaluate the SlowFast model:
````
python3.9 train_model.py --model slowfast --eval True --data "/Document/processed_data" --checkpoint "/Documents/check_points"
````


#### Train and evaluate the ensemble model:

Train and evaluate the ensemble model:

````
python3.9 train_model_ensemble.py.py --eval True
````