# INF2009 Edge Computing & Analytics - Team T05

Welcome to the repository for Team T05's project on Edge Computing & Analytics, part of the INF2009 course. This project focuses on leveraging edge computing techniques for real-time object detection and analytics, with a specific emphasis on public transportation scenarios.

## Project Overview

Our project aims to enhance public transportation systems through advanced object detection and analytics. We have developed a comprehensive solution that includes human and bus detection, extraction of bus signage and numbers, sound analysis, and Bluetooth connectivity. This solution is designed to run on edge devices, providing real-time insights and improving the overall efficiency and safety of public transportation.

## Repository Structure

The repository is organized into several key folders and files, each serving a specific purpose in the project:

### Object Detection

- **Folder:** `objectDetection`
- **Description:** Contains the code for detecting humans and buses, as well as extracting bus signage and numbers.
- **Key File:** `yolov5s.py` - Demonstrates what a live video feed with Region of Interest (ROI) bounding boxes for object detection might look like.

### Video Frames

- **Folder:** `frames`
- **Description:** Contains images extracted per second from the video file `finalVideo.mp4`. These frames are used for testing and refining our object detection algorithms.

### BLE

- **Folder:** `BLE`
- **Description:** 
  - `bluetooth_model.py` contains code for detecting the bluetooth device as well as coe to train on the SVM Model. Output: `device_count(latest).csv`
; `device_name(latest).csv`
- `device_count(latest).csv` contain the number of device detected.
- `device_name(latest).csv` contain the name of device detected.

### Sound Analytics

- **Folder:** `soundAnalytics`
- **Description:** 
  - `Data Preprocessing.py` contains code for preprocessing on `data_for_training.wav` (around 15 minutes audio). Output: `finalVideo_svm_features_final_data.csv`.
  - `finalVideo_svm_features_final_data_updated.csv` includes manual label of the presence of bus.
  - `SVM_Model_Training.py` contains code to train on the SVM model. Output:  `bus_detection_model.pkl` and  `feature_scalar_file.pkl`.
  - `Prediction.py` contains code to make prediction base on the models. Input:`finalVideo.wav` or `silence_testdata.wav` ; Output:  `finalVideo_predictions.csv`.

### Tasks Documentation

- **File:** `tasks.pdf`
- **Description:** A document outlining the tasks we have attempted and completed thus far in the project. This includes both our successes and challenges, providing a comprehensive view of our project's progress.

### Final Code

- **Folder:** `finalCode`
- **Description:** Contains the final version of our code, which integrates sound analysis, object detection, and Bluetooth connectivity using a scheduling system. This is the culmination of our project's development efforts.

## Running the Final Code

To execute the final integrated solution, follow these steps:

1. Ensure that all files within the `finalCode` folder are downloaded to your local machine.
2. Open a terminal or command prompt.
3. Navigate to the directory containing the `final_scheduler.py` file.
4. Run the following command:

```bash
python final_scheduler.py
