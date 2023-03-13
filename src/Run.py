import torch
import torchvision.transforms as transforms
import torch.nn as nn

import datetime
import time
import os
from tqdm import tqdm
import importlib
from utils.Visualization import plot_metrics, plot_time_measures
from utils.Header import print_header
import utils.helpermethods as helpermethods


def main() -> None:
    NUMBER_CLASSES = 10
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Parse the arguments
    args = helpermethods.parse_arguments()

    base_path = 'E:\\ML\\'
    measures_folder_path = base_path + f"Measures\\{args.model}"
    checkpoint_path = base_path + 'Checkpoints\\' + args.checkpoint + '.pt'

    # check if the folder exists
    helpermethods.make_folder(measures_folder_path)

    measures_folder_path = measures_folder_path + f"\\{date}.csv"
    measures_file = open(measures_folder_path, 'a')
    measures_file.write("num_epoch,image (ms),criterion (ms),optimizer (ms),accuracy (ms),checkpoint (ms),epoch (s)\n")
    # Define the model
    model = importlib.import_module('Models.' + args.model).Model(NUMBER_CLASSES)
    # count the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = format(num_params, ',d').replace(',', '.')

    if args.transformed:
        CustomDataset_module = importlib.import_module('dataLoaders.CustomDatasetTransformed')
        file_training = base_path + 'Datasets\\imagenette2\\transformed\\train.txt'
        file_validation = base_path + 'Datasets\\imagenette2\\transformed\\val.txt'
    else:
        CustomDataset_module = importlib.import_module('dataLoaders.CustomDatasetRaw')
        file_training = base_path + 'Datasets\\imagenette2\\train.txt'
        file_validation = base_path + 'Datasets\\imagenette2\\val.txt'
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    header_params = {'date': date,
                     'device': device,
                     'checkpoint_path': checkpoint_path,
                     'NUMBER_CLASSES': NUMBER_CLASSES,
                     'num_params': num_params
                     }
    print_header(args, header_params)
    # Define the transformation to be applied on the images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load the datasets
    print('Loading the dataset...')
    start = time.time()
    train_loader = helpermethods.load_dataset(CustomDataset_module, file_training, transform, args)
    val_loader = helpermethods.load_dataset(CustomDataset_module, file_validation, transform, args)
    print(f"Time to load the Dataset : {(time.time()-start)*1000:.3f} ms")

    # Train the model
    print('Training the model...')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Load the checkpoint if it exists
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_loss = checkpoint["best_loss"]
    else:
        best_loss = float("inf")

    losses_training = []
    accuracies_training = []

    losses_validation = []
    accuracies_validation = []

    for epoch in range(args.epochs):
        # Training
        train_loss = 0.0
        train_correct = 0.0
        train_total = 0.0

        # Set model to training mode
        model.train()

        # Use the dataloader to access the data in batches
        start_epoch = time.time()
        for inputs, labels in tqdm(train_loader, desc='Training', leave=True):
            # Move the data to the GPU if available
            inputs, labels = inputs.to(device), labels.to(device)

            start_train = time.time()
            outputs = model(inputs)
            end_train = time.time()

            start_crit = time.time()
            loss = criterion(outputs, labels)
            end_crit = time.time()

            # Backward pass and optimization
            start_opti = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            end_opti = time.time()

            # compute accuracy
            start_acc = time.time()
            train_accuracy = helpermethods.compute_accuracy(outputs, labels, args)
            end_acc = time.time()

            # update metrics
            train_loss += loss.item() * inputs.size(0)
            train_correct += train_accuracy
            train_total += inputs.size(0)

            start_save = 0.0
            if loss.item() < best_loss:
                start_save = time.time()
                best_loss = loss.item()

                # Save the best weights to file
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_loss": best_loss
                }, checkpoint_path)
            end_save = 0.0 if start_save == 0 else time.time()
        end_epoch = time.time()

        # Compute average training loss and accuracy
        train_loss /= float(len(train_loader.dataset))  # type: ignore
        train_acc = train_correct / train_total

        # Validation
        val_loss = 0.0
        val_correct = 0.0
        val_total = 0.0

        # Set model to evaluation mode
        model.eval()

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation', leave=True):
                # Move the data to the GPU if available
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Calculate validation loss and accuracy
                val_accuracy = helpermethods.compute_accuracy(outputs, labels, args)

                # update metrics
                val_correct += val_accuracy
                val_loss += loss.item() * inputs.size(0)
                val_total += inputs.size(0)

        measures = f'{epoch},{end_train-start_train:.3f},{end_crit-start_crit:.3f},{end_opti-start_opti:.3f}, \
            {end_acc-start_acc:.3f},{end_save-start_save:.3f},{end_epoch-start_epoch:.3f}\n'
        measures_file.write(measures)

        # Compute average validation loss and accuracy
        val_loss /= float(len(val_loader.dataset))  # type: ignore
        val_acc = val_correct / val_total

        # log metrics
        losses_training.append(train_loss)
        accuracies_training.append(train_acc)

        losses_validation.append(val_loss)
        accuracies_validation.append(val_acc)
        print(f"Epoch [{epoch+1}/{args.epochs}]: Training loss = {train_loss:.3f}, Training accuracy = {train_acc:.3f}")
        print(f"Epoch [{epoch+1}/{args.epochs}]: Validation loss = {val_loss:.3f}, Validation accuracy = {val_acc:.3f}")

    measures_file.close()
    if args.figures and args.epochs > 1:
        plot_metrics(losses_training, accuracies_training, losses_validation, accuracies_validation, args)
        plot_time_measures(measures_folder_path)


if __name__ == '__main__':
    main()
