import argparse

from utils.metrics_plot import plot_metrics


def main():
    # Pass the json metrics location to run as an argument
    parser = argparse.ArgumentParser(description='Plot metrics loaded from the corresponding .json file')
    parser.add_argument('--json', type=str, help='.json location', required=True)
    parser.add_argument('--save', type=bool, help='Save figures', required=False, default=False)
    parser.add_argument('--plot-epochs', type=str, help='Comma-separated list of epochs to plot', required=False)

    inpout = parser.parse_args()
    if inpout.plot_epochs:
        inpout.plot_epochs = set(map(int, inpout.plot_epochs.split(',')))

    plot_metrics(inpout.json, inpout.save, inpout.plot_epochs)


if __name__ == '__main__':
    main()