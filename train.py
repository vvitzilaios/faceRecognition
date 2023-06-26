import argparse

from utils.prepare_data import prepare_train_data
from utils.train_classifier import train_classifier


def main():
    # Pass the model name and number of epochs as arguments
    parser = argparse.ArgumentParser(description='Train a face recognition model')
    parser.add_argument('--model', type=int, help='Model to train', required=True)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')

    inpout = parser.parse_args()
    args = prepare_train_data(inpout.model)

    train_classifier(args, inpout.epochs)


if __name__ == '__main__':
    main()
