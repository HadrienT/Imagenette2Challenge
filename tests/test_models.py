import importlib
import argparse

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import pytest

import utils.helpermethods as helpermethods


def train(model_module: str) -> tuple[bool, float]:
    threshold = 0.1
    module = importlib.import_module(model_module)
    NUMBER_CLASSES = 10
    model = module.Model(NUMBER_CLASSES)
    parser = argparse.ArgumentParser(description='Train or evaluate a model')
    args = parser.parse_args()
    args.epochs = 100
    args.batch_size = 32
    file_training = 'E:/ML/Datasets/imagenette2/train.txt'
    CustomDataset_module = importlib.import_module('dataLoaders.CustomDatasetRaw')
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the transformation to be applied on the images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Load the datasets
    train_loader = helpermethods.load_dataset(CustomDataset_module, file_training, transform, args)
    # Train the model

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    inputs, labels = next(iter(train_loader))
    loss = float('inf')
    for epoch in range(args.epochs):
        print(f'Epoch {epoch + 1}/{args.epochs} - Loss: {loss:.4f}')
        # Set model to training mode
        model.train()
        # Move the data to the GPU if available
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()  # type: ignore
        optimizer.step()
        if loss.item() < threshold:  # type: ignore
            break
    return loss.item() < threshold, loss.item()  # type: ignore


@pytest.mark.skip(reason='This test takes too long to run')
def test_LeNet() -> None:
    model_module = 'Models.LeNet_5'
    threshold_passed, loss = train(model_module)
    assert threshold_passed, f'Loss is {loss}'


@pytest.mark.skip(reason='This test takes too long to run')
def test_AlexNet() -> None:
    model_module = 'Models.AlexNet'
    threshold_passed, loss = train(model_module)
    assert threshold_passed, f'Loss is {loss}'


if __name__ == '__main__':
    test_LeNet()
    test_AlexNet()
