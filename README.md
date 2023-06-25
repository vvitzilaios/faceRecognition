# PyTorch Facial Recognition Project

This is a PyTorch-based project that is aimed at training classifiers for facial recognition. 
The project provides an efficient and user-friendly platform to train models, validate them, 
and calculate various performance metrics.

## Installation

To install and use this project, follow these steps:

1. Clone this repository: `git clone https://github.com/vvitzilaios/faceRecognition.git`
2. Navigate to the directory: `cd faceRecognition`
3. Install required Python packages: `pip3 install -r requirements.txt`

## Usage

## Scripts

This project provides the following scripts:

1. `add_face.py`: This script adds what is recognized as a face to the dataset using a live feed from a camera.
    #### Arguments:
    - `--name`: The name of the person to be added.
    #### Example:
    ```bash
    python add_face.py --name "John Doe"
    ```
    ###
2. `train.py`: This script trains a model using the dataset.
    #### Arguments:
    - `--model`: The name of the model to be trained.
    - `--epochs`: The number of epochs to train the model for.
    #### Example:
    ```bash
    python train.py --model "model_one" --epochs 10
    ```
    ###
3. `plot.py`: This script plots the training and validation loss and accuracy of a model.
    #### Arguments:
    - `--json`: The name of the `.json` file containing the metrics of the model.
    - `--save`: Whether to save the produced figures or not.
    - `--plot-epochs`: Plot specific epochs. If empty then all epochs will be plotted.
    #### Example:
    ```bash
    python plot.py --model "model_one"
    ```
    ###
4. `run.py`: This script runs the model on live feed from a camera.
    #### Arguments:
    - `--model`: The name of the model to be run.
    #### Example:
    ```bash
    python run.py --model "model_one"
    ```
   
## Models

This project provides the following models:
1. `model_one.py`: A simple CNN model with 2 convolutional layers and 3 fully connected layers.
2. `model_two.py`: A simple CNN model with 3 convolutional layers and 3 fully connected layers.

Training those models will create a `.pth` file of the same name under the `models` directory.

## Dataset

For this project, parts of the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset were used.
This because it was not feasible to train a model by only using the `add_face.py` or by using the whole LFW dataset.

## Metrics

By training a model, the following metrics are calculated and saved in a `.json` file inside the `metrics/performance/` directory
1. Accuracy
2. Precision
3. Recall
4. F1 Score

One can also find the produced figures of the training and validation loss and accuracy inside the `metrics/figures/` 
directory, saved in `.png` format.


