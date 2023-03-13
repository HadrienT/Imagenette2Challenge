import importlib
import argparse
import time

import torch
import torchvision.transforms as transforms
import torch.nn as nn
from tqdm import tqdm

import utils.helpermethods as helpermethods
from Models import LeNet_5 as nnModel


def test_LeNet5() -> None:
    NUMBER_CLASSES = 10
    model = nnModel.Model(NUMBER_CLASSES)
    parser = argparse.ArgumentParser(description='Train or evaluate a model')
    args = parser.parse_args()
    args.epochs = 100
    args.batch_size = 32
    file_training = 'E:\\ML\\Datasets\\imagenette2\\train.txt'
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
    print('Loading the dataset...')
    train_loader = helpermethods.load_dataset(CustomDataset_module, file_training, transform, args)
    # Train the model
    print('Training the model...')

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    inputs, labels = next(iter(train_loader))

    for epoch in tqdm(range(args.epochs), desc="Training", leave=True):
        # Set model to training mode
        model.train()
        # Move the data to the GPU if available
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch [{epoch+1}/{args.epochs}]: Training loss = {loss:.3f}")


if __name__ == '__main__':
    test_LeNet5()
