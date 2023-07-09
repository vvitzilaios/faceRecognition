import argparse

from util.prepare_data import prepare_test_args
from service.run_service import run_model


def main():
    # Pass the trained model to run as an argument
    parser = argparse.ArgumentParser(description='Run a face recognition model')
    parser.add_argument('--model', type=int, help='Model to run', required=True)

    inpout = parser.parse_args()

    args = prepare_test_args(inpout.model)

    run_model(args)


if __name__ == '__main__':
    main()
