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

## Usage

Set and run scripts directly from the `pyproject.toml` file.

### Start the hand detector

```bash
poetry run poe hand_detector
```
