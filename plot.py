import argparse

from utils.metrics_plot import plot_metrics
from utils.prepare_data import prepare_test_args


def main():
    # Pass the json metrics location to run as an argument
    parser = argparse.ArgumentParser(description='Plot metrics loaded from the corresponding .json file')
    parser.add_argument('--json', type=str, help='.json location', required=True)

    inpout = parser.parse_args()

    plot_metrics(inpout.json)


if __name__ == '__main__':
    main()