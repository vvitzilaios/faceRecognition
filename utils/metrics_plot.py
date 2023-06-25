import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_metrics(metrics_json_path, save=False, plot_epochs=None):
    # Check if path exists
    if not os.path.exists(metrics_json_path):
        raise ValueError(f'{metrics_json_path} does not exist. Please train the model first.')

    # Load class mappings from JSON file
    with open('class_to_label.json', 'r') as f:
        class_map = json.load(f)

    # Load metrics from the JSON file
    with open(metrics_json_path, 'r') as f:
        loaded_json = json.load(f)
        model = loaded_json['model']
        metrics_data = pd.json_normalize(loaded_json, 'results')

    # Define class labels based on the class_map json
    labels = [class_map[str(i)] for i in range(len(class_map))]

    # Plot training and validation loss
    __plot_train_val_loss(metrics_data, model, save)

    # Plot validation accuracy, precision, recall, and F1 score
    __plot_val_results(metrics_data, model, save)

    # plot confusion matrix for specific epoch
    if plot_epochs:
        for epoch in plot_epochs:
            __plot_epoch_conf_mat(metrics_data, model, labels, epoch, save)
    else:
        # Plot the confusion matrices for each epoch
        __plot_all_conf_mats(labels, metrics_data, model, save)

    plt.show()


def __plot_epoch_conf_mat(metrics_data, model, labels, epoch, save=False):
    val_cm_series = metrics_data['val_cm']
    val_cm = val_cm_series[epoch - 1]
    plt.figure(figsize=(10, 5))
    sns.heatmap(np.array(val_cm), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix for Epoch {epoch}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    if save:
        __save_figures(f'{model}/conf_matrices', f'{model}_val_cm_epoch_{epoch}.png')


def __plot_all_conf_mats(labels, metrics_data, model, save=False):
    val_cm_series = metrics_data['val_cm']
    for idx, val_cm in enumerate(val_cm_series):
        plt.figure(figsize=(10, 5))
        sns.heatmap(np.array(val_cm), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix for Epoch {idx + 1}')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        if save:
            __save_figures(f'{model}/conf_matrices', f'{model}_val_cm_epoch_{idx + 1}.png')


def __plot_val_results(metrics_data, model, save=False):
    plt.figure(2, figsize=(10, 5))
    plt.plot(metrics_data['epoch'], metrics_data['val_accuracy'], label='Accuracy')
    plt.plot(metrics_data['epoch'], metrics_data['val_precision'], label='Precision')
    plt.plot(metrics_data['epoch'], metrics_data['val_recall'], label='Recall')
    plt.plot(metrics_data['epoch'], metrics_data['val_f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Score')
    plt.title('Validation Accuracy, Precision, Recall, and F1 Score Over Epochs')
    plt.legend()
    if save:
        __save_figures(model, f'{model}_val_metrics.png')


def __plot_train_val_loss(metrics_data, model, save=False):
    plt.figure(1, figsize=(10, 5))
    plt.plot(metrics_data['epoch'], metrics_data['train_loss'], label='Training Loss')
    plt.plot(metrics_data['epoch'], metrics_data['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    if save:
        __save_figures(model, f'{model}_train_val_loss.png')


def __save_figures(dirname: str, filename: str):
    figure_dir = 'metrics/figures'
    if dirname and filename:
        save_location = f'{figure_dir}/{dirname}/{filename}'
        os.makedirs(os.path.dirname(f'{save_location}'), exist_ok=True)
        plt.savefig(save_location)
    else:
        raise ValueError('Please provide a valid directory and file name')
