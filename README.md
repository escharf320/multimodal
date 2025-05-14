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

#### Segemnt video frames by word

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

