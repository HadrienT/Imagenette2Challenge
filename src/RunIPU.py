import os

if "Imagenette2Challenge" not in os.getcwd():
    os.chdir("./Imagenette2Challenge")

import datetime
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import poptorch
from tqdm import tqdm
import importlib

from src.utils.Header import print_header
import src.utils.helpermethods as helpermethods


class Args:
    def __init__(self, epochs: int, model_name: str, batch_size: int, target: int, checkpoint: str, training_mode: bool):
        self.epochs = epochs
        self.model = model_name
        self.batch_size = batch_size
        self.metric = target
        self.transformed = False
        self.figures = False
        self.checkpoint = checkpoint
        self.training_mode = training_mode


def main():
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    args = Args(EPOCHS, MODEL_NAME, BATCH_SIZE, TARGET, CHECKPOINT, TRAINING_MODE)

    base_path = "./"
    checkpoint_path = base_path + "Checkpoints/" + args.checkpoint + ".pt"

    opts = poptorch.Options()
    opts.deviceIterations(DEVICE_ITERATIONS)
    criterion = nn.CrossEntropyLoss()

    model = importlib.import_module("src.Models." + args.model).Model(NUMBER_CLASSES, criterion)
    CustomDataset_module = importlib.import_module("src.dataLoaders.CustomDatasetRaw")
    file_training = base_path + "Datasets/imagenette2/train.txt"
    file_validation = base_path + "Datasets/imagenette2/val.txt"
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    if args.training_mode:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

        train_dataset = helpermethods.load_dataset(
                CustomDataset_module, file_training, transform,
            )
        train_loader = poptorch.DataLoader(options=opts, dataset=train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

        # Wrap the model with poptorch training wrapper
        poptorch_model = poptorch.trainingModel(
            model, options=train_loader.options, optimizer=optimizer
        )

    val_dataset = helpermethods.load_dataset(
            CustomDataset_module, file_validation, transform,
        )
    val_loader = poptorch.DataLoader(options=opts, dataset=val_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # check if checkpoint exists
    if os.path.exists(checkpoint_path):
        if RESET:
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

            # Load the optimizer state and best loss only during training
            if args.training_mode:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                best_loss = checkpoint["best_loss"]
            else:
                best_loss = float(
                    "inf"
                )  # If not in training mode, initializing best_loss as infinity
    else:
        print("No checkpoint found.")
        best_loss = float(
            "inf"
        )  # As no checkpoint is found, initializing best_loss as infinity

    device = "IPU"
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = format(num_params, ",d").replace(",", ".")

    header_params = {
        "date": date,
        "device": device,
        "checkpoint_path": checkpoint_path,
        "NUMBER_CLASSES": NUMBER_CLASSES,
        "num_params": num_params,
    }
    # print_header(args, header_params)

    # Create model

    model.train()

    # Set optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)

    model.eval()  # Switch model to evaluation mode
    poptorch_inference_model = poptorch.inferenceModel(model, options=val_loader.options)
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for epoch in range(args.epochs):
        if args.training_mode:
            # Training
            train_loss_epoch = []
            train_acc_epoch = []
            # Use the dataloader to access the data in batches
            with tqdm(total=len(train_loader), unit="batch", desc="Training", leave=True) as pbar:
                pbar.set_description(f"Epoch {epoch+1}")
                for inputs, labels in train_loader:
                    outputs, loss = poptorch_model(inputs, labels)

                    # Calculate validation loss and accuracy
                    train_accuracy = helpermethods.compute_accuracy_IPU(
                        outputs, labels, args, DEVICE_ITERATIONS
                    )

                    # update metrics
                    train_loss_epoch.append(loss.item())
                    train_acc_epoch.append(train_accuracy)

                    pbar.update(1)
                    pbar.set_postfix({"Loss": loss.item()})
                pbar.set_description(f"Done training epoch [{epoch+1}/{args.epochs}]")

            train_acc.append(np.mean(train_acc_epoch))
            train_loss.append(np.mean(train_loss_epoch))

        # Validation
        val_loss_epoch = []
        val_acc_epoch = []

        with torch.no_grad():
            with tqdm(total=len(val_loader), unit="batch", desc="Validating", leave=True) as pbar:
                pbar.set_description(f"Epoch {epoch+1}")
                for inputs, labels in val_loader:
                    outputs = poptorch_inference_model(inputs)
                    loss = criterion(outputs, labels)
                    # Calculate validation loss and accuracy
                    val_accuracy = helpermethods.compute_accuracy_IPU(
                        outputs, labels, args, DEVICE_ITERATIONS
                    )

                    # update metrics
                    val_loss_epoch.append(loss.item())
                    val_acc_epoch.append(val_accuracy)

                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "best_loss": best_loss,
                            },
                            checkpoint_path,
                        )

                    pbar.update(1)
                    pbar.set_postfix({"Loss": loss.item()})

                pbar.set_description(f"Done validating epoch [{epoch+1}/{args.epochs}]")

        # Compute average validation loss and accuracy
        val_acc.append(np.mean(val_acc_epoch))
        val_loss.append(np.mean(val_loss_epoch))

    # Save metrics to a file
    metrics = {
        "losses_training": train_loss,
        "accuracies_training": train_acc,
        "losses_validation": val_loss,
        "accuracies_validation": val_acc,
    }
    torch.save(metrics, f"./Metrics/{args.model}.pt")

    for k, v in metrics.items():
        print(f"{k} : {v}")


if __name__ == "__main__":
    RESET = False
    EPOCHS = 1
    MODEL_NAME = "LeNet_5"
    BATCH_SIZE = 32
    TARGET = 1
    NUMBER_CLASSES = 10
    DEVICE_ITERATIONS = 1
    CHECKPOINT = "test"
    TRAINING_MODE = False

    main()
