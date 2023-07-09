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
    with open('data/class_to_label.json', 'r') as f:
        class_map = json.load(f)

    # Load metrics from the JSON file
    with open(metrics_json_path, 'r') as f:
        loaded_json = json.load(f)
        model = loaded_json['model']
        metrics_data = pd.json_normalize(loaded_json, 'results')

    # Define class labels based on the class_map json
    labels = [class_map[str(i)] for i in range(len(class_map))]

    # Plot training and validation loss
    fig, ax = __plot_train_val_loss(metrics_data, model)
    if save:
        __save_figures(model, f'{model}_train_val_loss.png', fig)
    fig.show()

    # Plot validation accuracy, precision, recall, and F1 score
    fig, ax = __plot_val_results(metrics_data, model)
    if save:
        __save_figures(model, f'{model}_val_metrics.png', fig)
    fig.show()

    # plot confusion matrix for specific epoch
    if plot_epochs:
        for epoch in plot_epochs:
            fig, ax = __plot_epoch_conf_mat(metrics_data, model, labels, epoch)
            if save:
                __save_figures(f'{model}/conf_matrices', f'{model}_val_cm_epoch_{epoch}.png', fig)
            fig.show()
    else:
        # Plot the confusion matrices for each epoch
        __plot_all_conf_mats(labels, metrics_data, model, save)


def __plot_epoch_conf_mat(metrics_data, model, labels, epoch):
    val_cm_series = metrics_data['val_cm']
    val_cm = val_cm_series[epoch - 1]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(np.array(val_cm), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix for Epoch {epoch}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    return fig, ax


def __plot_all_conf_mats(labels, metrics_data, model, save=False):
    val_cm_series = metrics_data['val_cm']
    for idx, val_cm in enumerate(val_cm_series):
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(np.array(val_cm), annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix for Epoch {idx + 1}')
        plt.ylabel('Actual Class')
        plt.xlabel('Predicted Class')
        if save:
            __save_figures(f'{model}/conf_matrices', f'{model}_val_cm_epoch_{idx + 1}.png', fig)
        fig.show()


def __plot_val_results(metrics_data, model):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_data['epoch'], metrics_data['val_accuracy'], label='Accuracy')
    ax.plot(metrics_data['epoch'], metrics_data['val_precision'], label='Precision')
    ax.plot(metrics_data['epoch'], metrics_data['val_recall'], label='Recall')
    ax.plot(metrics_data['epoch'], metrics_data['val_f1'], label='F1 Score')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric Score')
    ax.set_title('Validation Accuracy, Precision, Recall, and F1 Score Over Epochs')
    ax.legend()
    return fig, ax


def __plot_train_val_loss(metrics_data, model):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(metrics_data['epoch'], metrics_data['train_loss'], label='Training Loss')
    ax.plot(metrics_data['epoch'], metrics_data['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss Over Epochs')
    ax.legend()
    return fig, ax


def __save_figures(dirname: str, filename: str, fig):
    figure_dir = 'metrics/figures'
    if dirname and filename:
        save_location = f'{figure_dir}/{dirname}/{filename}'
        os.makedirs(os.path.dirname(f'{save_location}'), exist_ok=True)
        fig.savefig(save_location)
    else:
        raise ValueError('Please provide a valid directory and file name')
