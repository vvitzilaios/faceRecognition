import torch
import torch.optim as optim

from utils.prepare_data import define_model


def train_classifier(args, num_epochs: int):
    # Initialize the data
    device, num_classes, selected_model, train_loader, val_loader = __init_data(args)

    # Define the model
    model = define_model(selected_model, num_classes).to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        for _, train_data in enumerate(train_loader, 0):
            inputs, labels = train_data[0].to(device), train_data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            with torch.no_grad():  # No calculation of gradients to save memory
                validation_loss = 0.0
                for _, val_data in enumerate(val_loader, 0):
                    inputs, labels = val_data[0].to(device), val_data[1].to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    validation_loss += loss.item()

        print(f'Epoch {epoch + 1}, training loss: {running_loss / len(train_loader)}, '
              f'validation loss: {validation_loss / len(val_loader)}')

    # Save the trained model
    torch.save(model.state_dict(), f'{selected_model}.pth')
    print('Finished Training')


def __init_data(dataset_info):

    return dataset_info.device,\
        dataset_info.num_classes, \
        dataset_info.model, \
        dataset_info.train_loader, \
        dataset_info.val_loader
