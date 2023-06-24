import argparse

from utils.prepare_data import prepare_test_args
from utils.run_model import run_model


def main():
    # Pass the trained model to run as an argument
    parser = argparse.ArgumentParser(description='Run a face recognition model')
    parser.add_argument('--model', type=str, help='Model to run', required=True)

    inpout = parser.parse_args()

    args = prepare_test_args(inpout.model)

    run_model(args)


if __name__ == '__main__':
    main()
