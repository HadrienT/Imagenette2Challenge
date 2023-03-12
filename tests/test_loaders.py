import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataLoaders import CustomDatasetRaw
from dataLoaders import CustomDatasetTransformed
from utils import helpermethods as helpermethods


@pytest.fixture(scope="module")
def sample_data_raw():
    # Generate sample data
    file = "E:\\ML\\Datasets\\imagenette2\\train.txt"
    image_paths, labels = helpermethods.get_paths_and_labels(file)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    return image_paths, labels, transform


def test_custom_dataset_raw(sample_data_raw):
    # Test the CustomDataset class
    image_paths, labels, transform = sample_data_raw
    dataset = CustomDatasetRaw.CustomDataset(image_paths, labels, transform=transform)

    assert len(dataset) == len(image_paths)

    # Test the first item in the dataset
    data, label = dataset[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (3, 256, 256)
    assert isinstance(label, torch.Tensor)
    assert label.item() == 0

    # Test the dataset with a DataLoader
    dataloader = DataLoader(dataset, batch_size=2)
    batch_data, batch_labels = next(iter(dataloader))
    assert isinstance(batch_data, torch.Tensor)
    assert batch_data.shape == (2, 3, 256, 256)
    assert isinstance(batch_labels, torch.Tensor)
    assert batch_labels.shape == (2,)


@pytest.fixture(scope="module")
def sample_data_transformed():
    # Generate sample data
    file = "E:\\ML\\Datasets\\imagenette2\\transformed\\train.txt"
    image_paths, labels = helpermethods.get_paths_and_labels(file)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    return image_paths, labels, transform


def test_custom_dataset_transformed(sample_data_transformed):
    # Test the CustomDataset class
    image_paths, labels, transform = sample_data_transformed
    dataset = CustomDatasetTransformed.CustomDataset(image_paths, labels, transform=transform)

    assert len(dataset) == len(image_paths)

    # Test the first item in the dataset
    data, label = dataset[0]
    assert isinstance(data, torch.Tensor)
    assert data.shape == (3, 256, 256)
    assert isinstance(label, torch.Tensor)
    assert label.item() == 0

    # Test the dataset with a DataLoader
    dataloader = DataLoader(dataset, batch_size=2)
    batch_data, batch_labels = next(iter(dataloader))
    assert isinstance(batch_data, torch.Tensor)
    assert batch_data.shape == (2, 3, 256, 256)
    assert isinstance(batch_labels, torch.Tensor)
    assert batch_labels.shape == (2,)
