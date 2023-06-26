import json
import os
import time

import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

from utils.prepare_data import define_model


def train_classifier(args, num_epochs: int):
    # Initialize the data
    device, num_classes, selected_model, train_loader, val_loader = __init_data(args)

    # Define the model
    model = define_model(selected_model, num_classes).to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Initialize lists for targets and predictions
    results = []

    # Train the model and calculate the validation loss
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        start_time_train = time.time()                      # Start time for the epoch
        for _, train_data in enumerate(train_loader, 0):
            inputs, train_labels = train_data[0].to(device), train_data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate the validation loss and metrics
        model.eval()
        all_val_labels = []
        all_val_preds = []
        validation_loss = 0.0
        for _, val_data in enumerate(val_loader, 0):
            inputs, labels = val_data[0].to(device), val_data[1].to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(predicted.cpu().numpy())
            loss = criterion(outputs, labels)
            validation_loss += loss.item()

        # Calculate the metrics using sklearn methods
        val_accuracy = accuracy_score(all_val_labels, all_val_preds)
        val_precision = precision_score(all_val_labels, all_val_preds, average='macro', zero_division=1)
        val_recall = recall_score(all_val_labels, all_val_preds, average='macro')
        val_f1 = f1_score(all_val_labels, all_val_preds, average='macro')
        val_cm = confusion_matrix(all_val_labels, all_val_preds)

        end_time_train = time.time()                        # End time for the epoch

        epoch_time = end_time_train - start_time_train      # Time taken for the epoch in seconds
        train_loss = running_loss / len(train_loader)       # Average training loss for the epoch
        val_loss = validation_loss / len(val_loader)        # Average validation loss for the epoch
        epoch_num = epoch + 1                               # Epoch number

        print(f'Epoch {epoch_num}, training loss: {train_loss}, validation loss: {val_loss}, '
              f'time elapsed: {epoch_time:.3f} seconds')

        result = {
            'epoch': epoch_num,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'val_cm': val_cm.tolist(),
            'epoch_time': epoch_time
        }
        results.append(result)

    result_objects = {
        'model': selected_model,
        'num_epochs': num_epochs,
        'total_time': sum([result['epoch_time'] for result in results]),
        'results': results
    }

    # Save the metrics
    __save_metrics(result_objects, selected_model)

    # Save the trained model
    torch.save(model.state_dict(), f'models/pth/{selected_model}.pth')
    print('Finished Training')


def __save_metrics(result_objects, selected_model):
    # Create a new directory for the metrics if it doesn't exist
    folder_path = 'metrics/performance'
    os.makedirs(folder_path, exist_ok=True)
    # Save the result objects as a JSON file
    with open(f'{folder_path}/{selected_model}_metrics.json', 'w') as f:
        json.dump(result_objects, f, indent=4)


def __init_data(dataset_info):
    return dataset_info.device, \
        dataset_info.num_classes, \
        dataset_info.model, \
        dataset_info.train_loader, \
        dataset_info.val_loader
