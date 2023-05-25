import os
import datetime
import time
import importlib

import torch
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from utils.Visualization import plot_metrics, plot_time_measures
from utils.Header import print_header
import utils.helpermethods as helpermethods


class Args:
    def __init__(self, epochs: int, model_name: str, batch_size: int, target: int, checkpoint: str):
        self.epochs = epochs
        self.model = model_name
        self.batch_size = batch_size
        self.metric = target
        self.transformed = False
        self.figures = False
        self.checkpoint = checkpoint


def main() -> None:
    NUMBER_CLASSES = 10
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Parse the arguments
    args = Args(EPOCHS, model_name, BATCH_SIZE, TARGET, CHECKPOINT)
    # args = helpermethods.parse_arguments()

    base_path = "E:/ML/"
    measures_folder_path = base_path + f"Measures/{args.model}"
    metric_file_path = base_path + "/metrics.pt"
    checkpoint_path = base_path + "Checkpoints/" + args.checkpoint + ".pt"

    # check if the folder exists
    helpermethods.make_folder(measures_folder_path)

    measures_folder_path = measures_folder_path + f"/{date}.csv"
    measures_file = open(measures_folder_path, "a")
    measures_file.write(
        "num_epoch,image (ms),criterion (ms),optimizer (ms),accuracy (ms),checkpoint (ms),epoch (s)\n"
    )
    criterion = nn.CrossEntropyLoss()
    # Define the model
    model = importlib.import_module("Models." + args.model).Model(NUMBER_CLASSES, criterion)
    # count the number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = format(num_params, ",d").replace(",", ".")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    if args.transformed:
        CustomDataset_module = importlib.import_module(
            "dataLoaders.CustomDatasetTransformed"
        )
        file_training = base_path + "Datasets/imagenette2/transformed/train.txt"
        file_validation = base_path + "Datasets/imagenette2/transformed/val.txt"
    else:
        CustomDataset_module = importlib.import_module("dataLoaders.CustomDatasetRaw")
        file_training = base_path + "Datasets/imagenette2/train.txt"
        file_validation = base_path + "Datasets/imagenette2/val.txt"
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    header_params = {
        "date": date,
        "device": device,
        "checkpoint_path": checkpoint_path,
        "NUMBER_CLASSES": NUMBER_CLASSES,
        "num_params": num_params,
    }
    print_header(args, header_params)
    # Define the transformation to be applied on the images
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Load the datasets
    print("Loading the dataset...")
    start = time.time()

    train_dataset = helpermethods.load_dataset(
        CustomDataset_module,
        file_training,
        transform,
    )

    val_dataset = helpermethods.load_dataset(
        CustomDataset_module,
        file_validation,
        transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6
    )

    print(f"Time to load the Dataset : {(time.time()-start)*1000:.3f} ms")

    # Train the model
    print("Training the model...")

    # check if checkpoint exists
    if os.path.exists(checkpoint_path):
        if reset:
            # delete checkpoint
            os.remove(checkpoint_path)
            print("Checkpoint removed.")
            best_loss = float(
                "inf"
            )  # As checkpoint is removed, initializing best_loss as infinity
        else:
            print("Checkpoint exists.")
            # Load the checkpoint
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            best_loss = checkpoint["best_loss"]
    else:
        print("No checkpoint found.")
        best_loss = float(
            "inf"
        )  # As no checkpoint is found, initializing best_loss as infinity

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(args.epochs):
        # Training
        train_loss_epoch = []
        train_acc_epoch = []

        # Set model to training mode
        model.train()

        # Use the dataloader to access the data in batches
        start_epoch = time.time()
        with tqdm(total=len(train_loader), unit="batch", desc="Training", leave=True) as pbar:
            pbar.set_description(f"Epoch {epoch+1}")
            for inputs, labels in train_loader:
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
                train_loss_epoch.append(loss.item())
                train_acc_epoch.append(train_accuracy)

                pbar.update(1)
                pbar.set_postfix({"Loss": loss.item()})

            pbar.set_description(f"Done training epoch [{epoch+1}/{args.epochs}]")
        # Compute average training loss and accuracy
        train_acc.append(np.mean(train_acc_epoch))
        train_loss.append(np.mean(train_loss_epoch))
        end_epoch = time.time()

        # Validation
        val_loss_epoch = []
        val_acc_epoch = []

        # Set model to evaluation mode
        model.eval()

        with torch.no_grad():
            with tqdm(total=len(val_loader), unit="batch", desc="Validating", leave=True) as pbar:
                pbar.set_description(f"Epoch {epoch+1}")
                for inputs, labels in val_loader:
                    # Move the data to the GPU if available
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # Calculate validation loss and accuracy
                    val_accuracy = helpermethods.compute_accuracy(outputs, labels, args)

                    # update metrics
                    val_loss_epoch.append(loss.item())
                    val_acc_epoch.append(val_accuracy)
                    start_save = 0.0
                    if loss.item() < best_loss:
                        best_loss = loss.item()

                        # Save the best weights to file
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "best_loss": best_loss,
                            },
                            checkpoint_path,
                        )
                    end_save = 0.0 if start_save == 0 else time.time()

                    pbar.update(1)
                    pbar.set_postfix({"Loss": loss.item()})
                pbar.set_description(f"Done validating epoch [{epoch+1}/{args.epochs}]")

        measures = f"{epoch},{end_train-start_train:.3f},{end_crit-start_crit:.3f},{end_opti-start_opti:.3f}, \
            {end_acc-start_acc:.3f},{end_save-start_save:.3f},{end_epoch-start_epoch:.3f}\n"
        measures_file.write(measures)

        # Compute average validation loss and accuracy
        val_acc.append(np.mean(val_acc_epoch))
        val_loss.append(np.mean(val_loss_epoch))

    # log metrics
    metrics = {
        "losses_training": train_loss,
        "accuracies_training": train_acc,
        "losses_validation": val_loss,
        "accuracies_validation": val_acc,
    }
    torch.save(metrics, metric_file_path)

    for k, v in metrics.items():
        print(f"{k} : {v}")

    measures_file.close()
    if args.figures and args.epochs > 1:
        plot_metrics(train_loss, train_acc, val_loss, val_acc, args)
        plot_time_measures(measures_folder_path)


if __name__ == "__main__":
    reset = False
    EPOCHS = 1
    model_name = "LeNet_5"
    BATCH_SIZE = 32
    TARGET = 1
    NUMBER_CLASSES = 10
    DEVICE_ITERATIONS = 1
    CHECKPOINT = "test"
    main()
