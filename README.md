# PyTorch Facial Recognition Project

This is a PyTorch-based project that is aimed at training classifiers for facial recognition. 
The project provides an efficient and user-friendly platform to train models, validate them, 
and calculate various performance metrics.

## Installation

To install and use this project, follow these steps:

1. Clone this repository: `git clone https://github.com/vvitzilaios/faceRecognition.git`
2. Navigate to the directory: `cd faceRecognition`
3. Install required Python packages: `pip3 install -r requirements.txt`

## Information

### Usage

To successfully use this project, follow these steps:
1. Add a face to the dataset by running the `add_face.py` script.
2. A window will open with the live feed from the camera detecting the closest to camera face (by size). 
3. Press `s` in order to capture the face and add it to the dataset.
4. Repeat steps 1-3 for as many faces as you want.
5. Train a model by running the `train.py` script.
6. Plot the training and validation loss and accuracy of the model by running the `plot.py` script.
7. Run the model on live feed from a camera by running the `run.py` script.

**Note:** For more details and examples, see the [Scripts](#scripts) section.
 
### Models

This project provides the following models:
1. `model_one.py`: A simple CNN model with 2 convolutional layers and 3 fully connected layers.
2. `model_two.py`: A simple CNN model with 3 convolutional layers and 3 fully connected layers.

**Note:** Training those models will create a `.pth` file of the same name under the `models` directory.

### Dataset

For this project, parts of the [Labeled Faces in the Wild](http://vis-www.cs.umass.edu/lfw/) dataset were used.
This because it was not feasible to train a model by only using the `add_face.py` or by using the whole LFW dataset.

**Note:**  The dataset used for this project can be found in the following [Google Drive link](https://drive.google.com/file/d/1UDZv-oiOtbRvNZm85ahVUBoHby5Gcah5/view?usp=share_link).

### Metrics

By training a model, the following metrics are calculated and saved in a `.json` file inside the `metrics/performance/` directory
1. Accuracy
2. Precision
3. Recall
4. F1 Score

**Note:** One may also find the produced figures of the training and validation loss and accuracy inside the `metrics/figures/` 
directory, saved in `.png` format.

## Scripts

This project provides the following scripts:

1. `add_face.py`: This script adds what is recognized as a face to the dataset using a live feed from a camera.
    #### Arguments:
    - `--name`: The name of the person to be added.
    #### Side Effects:
    - A window will open with the live feed from the camera detecting the closest to camera face (by size).
    - A directory with the name of the person will be created under the `data/dataset` directory.
    #### Example:
    ```bash
    python add_face.py --name "John Doe"
    ```
    ###
2. `train.py`: This script trains a model using the dataset.
    #### Arguments:
    - `--model`: The name of the model to be trained.
    - `--epochs`: The number of epochs to train the model for.
    #### Side Effects:
    - A `.pth` file with the name of the model will be created under the `models` directory.
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
    #### Side Effects:
    - Directories `metrics`, `metrics/performance` and `metrics/figures` will be created, depending on the above arguments.
    - `.png` files with a custom name will be created under the `metrics/figures` directory. Same goes for confusion matrices.
    #### Example:
    ```bash
    python plot.py --model "model_one"
    ```
    ###
4. `run.py`: This script runs the model on live feed from a camera.
    #### Arguments:
    - `--model`: The name of the model to be run.
    #### Side Effects:
    - A window will open with the live feed from the camera. The recognized faces will be marked with a green rectangle.
    #### Example:
    ```bash
    python run.py --model "model_one"
    ```
  


