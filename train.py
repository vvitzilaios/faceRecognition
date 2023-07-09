import argparse

from util.prepare_data import prepare_train_data
from service.train_service import train_classifier


def main():
    # Pass the model name and number of epochs as arguments
    parser = argparse.ArgumentParser(description='Train a face recognition model')
    parser.add_argument('--dataset', type=str, default='data/dataset', help='Path to the dataset')
    parser.add_argument('--model', type=int, help='Model to train', required=True)
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs to train for')
    parser.add_argument('--patience', type=int, default=10, help='Number of epochs to wait for improvement')

    inpout = parser.parse_args()
    args = prepare_train_data(inpout.model, inpout.dataset)

    train_classifier(args, inpout.epochs, inpout.patience)


if __name__ == '__main__':
    main()
