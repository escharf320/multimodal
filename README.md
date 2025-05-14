# TypeFormAR

A repository for our multimodal final project.

## Installation

This project requires Python 3.10 and requires [Poetry](https://python-poetry.org/) to manage dependencies. Once these requirements have been installed, run the following in terminal to install all of the project dependencies.

```bash
poetry install
```

Finally, install the following dependency which is required by the Pyzbar dependency:

[Follow the installation instructions from the Pyzbar project instructions.](https://github.com/NaturalHistoryMuseum/pyzbar/?tab=readme-ov-file#installation)

## Download Project Binaries

The dataset and some of the final versions of the models are too big to be stored in Git. Instead, we've been hosting them in Google Drive.

The link to these files are available here:

[https://drive.google.com/drive/folders/1HFTJmDSOOkZPfSQp7BRMnma6N9rn3Zza?usp=sharing](https://drive.google.com/drive/folders/1HFTJmDSOOkZPfSQp7BRMnma6N9rn3Zza?usp=sharing)

Before running any of the commands below, ensure that the files stored in this folder are copied to the project directory. Copy the files directly such that the `models` and `data` folders are in the project root (multimodal directory).

Note that these are a subset of the total files we used within the project and are meant to be purely illustrative for the examples listed below in `Usage`.

## Usage

The bulk of the project work involved data preparation, model training, and model evaluation. As such, instead of having one main script, there are several examples to demonstrate components of our overall project. We list out several such demos below.

| **Disclaimer**: These scripts require that you have the project binaries stored locally. Read the previous section for instructions

### 1. Test Indiviudal Data Collection Components

#### Convert log into strings of words

This script parses the log file into human readable words.

```bash
poetry run poe segment_words
```

#### Segment video frames by word

Timestamp inference returns the word, start time, and respective joints.

```bash
poetry run poe timestamp_inference
```

### 2. Evaluate the transformer

This script will read the adapter model in the `models` directory and evaluate it against the pre-trained adapter transormer model.

```bash
poetry run poe eval_transformer
```

### 3. Train the transformer (This takes a long time)

Try training the adapted transformer! Delete the `models/adapter_model_checkpoint.pth` file and tth

```bash
poetry run poe train_transformer
```

### 4. Test out the key logger Electron application

The code for the key logger exists on the [jacob/type-tool](https://github.com/escharf320/multimodal/tree/jacob/type-tool) branch. A different set of instructions exist on that branch's `README.md`. To switch to that branch, use the following command.

```bash
git checkout jacob/type-tool
```

And to switch back to this code:

```bash
git checkout main
```

## Project Structure

```txt
TypeformAR
├── dataset_generation
│   ├── batch_video_preperation.py
│   ├── create_labeled_joint_data.py
│   ├── frame_timestamp_inference.py
│   ├── log_parser.py
│   ├── predict_timestamps.py
│   ├── test_timestamp_inference.py
│   ├── loss_array
│   │   ├── distance_funcs.py
│   │   ├── key_pressed.py
│   │   └── test_key_pressed.py
│   ├── segement_words
│   │   ├── find_words.py
│   │   ├── into_to_key.py
│   │   ├── segment_words_by_space.py
│   │   └── test_segment_words_by_space.py
│   ├── spacebar_detection
│   │   ├── data_joint_visualization.py
│   │   ├── dataset_preperation.py
│   │   └── space_detector_lstm.py
│   └── t5_finetuning
│       ├── dataset.py
│       ├── eval.py
│       ├── models.py
│       └── train.py
└── hand_detector.py
```

### Batch Video Preperation

- Process Video Files: The script scans data directory for .mov video files, extracts UUID and timestamps using QR code inference, and skips already processed videos by checking for existing pickle files.

- Extract Features & Logs: For each unprocessed video, the script extracts joint positions from the video frames and combines them with inferred timestamps. It also parses the associated log file for additional data.

- Save Data: The extracted data (timestamped joint positions and logs) is saved into pickle files named after the video’s UUID for future use or analysis.

### Create Labeled Joint Data

- Process Video & Log: The script processes a video (.mov) to extract timestamped joint positions and reads a corresponding log file (.log) to get timestamped words.

- Segment Joints by Word: It segments the joint positions based on the words in the log file, aligning the joint data with the words and their corresponding timestamps.

- Save Segmented Data: The segmented joint data (aligned with words) is saved to a pickle file.

### Frame Timestamp Inferernce

- Process QR and Hand Data: The script processes video frames to extract QR codes (UUID and timestamps) and hand landmarks (using HandDetector) from each frame.

- Parallel Processing: QR code extraction is parallelized using ThreadPoolExecutor to speed up processing, while hand landmark extraction is done sequentially.

- Combine Data: The extracted timestamps and joint positions are combined, missing timestamps are predicted, and the result (UUID and timestamped joint data) is returned.

### Log Parser

- Preprocess Log Data: Ensures the "type" key in each log dictionary is either "down" or "up", converting type codes 4 and 5 to their respective string values.

- Process Log File: It reads a logger file line by line, parsing each JSON line into a dictionary.

### Predict Timestamps

- Uses linear regression to predict missing timestamps based on known frame-timestamp pairs.

### Test Timestamp Inference

- The script processes a video (.mov) to extract joint positions with timestamps and parses a corresponding log file (log) for key presses and words.

### Key Pressed

- Creates a list the same length as frames in video. Assigns label of true or false if key is pressed in that frame

### Test Key Pressed

- Test time of different implementations of key_pressed

### Find Words

- Builds string representation of words from log file

### Int to Key

- Dictionary to convert integer representation of key to string representation

### Segment Words by Space

- Helper functions to return the joint frames for each word
- Returns tuple of (word, corresponding joints)

### Test Segment Words

- Parses the log file into human readable words

### Data Joint Visulaizationver group of

frames

### Dataset Preperation

- Processes video data to extract hand joint features (velocity, normalized positions, etc.), generating contiguous sequences of these features and labels indicating whether the spacebar was pressed

- Sliding Window Dataset Creation: Generates sliding windows from the contiguous sequences, each of size WINDOW_SIZE, with the middle element as the output label, preparing the data for model training

### Space Detector LSTM

- Defines a bidirectional LSTM-based model to detect spacebar presses based on hand joint features

- Prepares the dataset, handles class imbalance using weighted loss, and trains the model

### Hand Detector

- Runs MediaPipe's HandDetector using bu-ilt in webcam

Visualzie joing data in 3D o

### Dataset

- Reads segmented joint data from the `data/joints` directory and partitions it into a training and testing dataset following a 90-10 split.

### Models

- Defines the T5 adapter model and the time series adapter MLP model
- Includes helper methods for generating the model prompt and determining the device based on the host machine.

### Train

- Trains the model for 1000 epochs with helpers to save the model on early exits. Uses an AdamW optimizer.

### Eval

- Script for loading the pre-trained model and evaluating it against the test partition of the dataset.
