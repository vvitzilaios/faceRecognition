import json
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(metrics_json_path):
    # Check if path exists
    if not os.path.exists(metrics_json_path):
        raise ValueError(f'{metrics_json_path} does not exist. Please train the model first.')

    # Load metrics from the JSON file
    with open(metrics_json_path, 'r') as f:
        loaded_json = json.load(f)
        model = loaded_json['model']
        metrics_data = pd.json_normalize(loaded_json, 'results')

    # Plot training and validation loss
    plt.figure(1, figsize=(10, 5))
    plt.plot(metrics_data['epoch'], metrics_data['train_loss'], label='Training Loss')
    plt.plot(metrics_data['epoch'], metrics_data['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    __save_figures(model, f'{model}_train_val_loss.png')

    # Plot validation accuracy, precision, recall, and F1 score
    plt.figure(2, figsize=(10, 5))
    plt.plot(metrics_data['epoch'], metrics_data['val_accuracy'], label='Accuracy')
    plt.plot(metrics_data['epoch'], metrics_data['val_precision'], label='Precision')
    plt.plot(metrics_data['epoch'], metrics_data['val_recall'], label='Recall')
    plt.plot(metrics_data['epoch'], metrics_data['val_f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Score')
    plt.title('Validation Accuracy, Precision, Recall, and F1 Score Over Epochs')
    plt.legend()
    __save_figures(model, f'{model}_val_metrics.png')

    plt.show()


def __save_figures(dirname: str, filename: str):
    figure_dir = 'metrics/figures'
    if dirname and filename:
        save_location = f'{figure_dir}/{dirname}/{filename}'
        os.makedirs(os.path.dirname(f'{save_location}'), exist_ok=True)
        plt.savefig(save_location)
    else:
        raise ValueError('Please provide a valid directory and file name')

