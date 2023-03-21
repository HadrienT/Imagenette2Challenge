import os
import tempfile

import pytest
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import PIL.Image as Image

from dataLoaders import CustomDatasetRaw, CustomDatasetTransformed, InferLoader
from utils import helpermethods as helpermethods


@pytest.skip("Skip to pass GitHub Actions")
@pytest.fixture(scope="module")
def sample_data_raw() -> tuple[list[str], list[int], transforms.Compose]:
    # Generate sample data
    file = "E:\\ML\\Datasets\\imagenette2\\train.txt"
    image_paths, labels = helpermethods.get_paths_and_labels(file)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    return image_paths, labels, transform


@pytest.skip("Skip to pass GitHub Actions")
def test_custom_dataset_raw(sample_data_raw: tuple[list[str], list[int], transforms.Compose]) -> None:
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


@pytest.skip("Skip to pass GitHub Actions")
@pytest.fixture(scope="module")
def sample_data_transformed() -> tuple[list[str], list[int], transforms.Compose]:
    # Generate sample data
    file = "E:\\ML\\Datasets\\imagenette2\\transformed\\train.txt"
    image_paths, labels = helpermethods.get_paths_and_labels(file)
    transform = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    return image_paths, labels, transform


@pytest.skip("Skip to pass GitHub Actions")
def test_custom_dataset_transformed(sample_data_transformed: tuple[list[str], list[int], transforms.Compose]) -> None:
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


def test_infer_loader() -> None:
    # Generate sample data
    with tempfile.TemporaryDirectory() as tempdir:
        img1 = Image.new("RGB", (32, 32), color="red")
        img2 = Image.new("RGB", (32, 32), color="blue")
        img3 = Image.new("RGB", (32, 32), color="green")
        img1.save(os.path.join(tempdir, "1.jpg"))
        img2.save(os.path.join(tempdir, "2.jpg"))
        img3.save(os.path.join(tempdir, "3.jpg"))
        transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
        dataset = InferLoader.CustomDataset(tempdir, transform=transform)

        assert len(dataset) == 3
        # Test the first item in the dataset
        data = dataset[0]
        assert isinstance(data, torch.Tensor)
        assert data.shape == (3, 64, 64)

        # Test the dataset with a DataLoader
        dataloader = DataLoader(dataset, batch_size=2)
        batch_data = next(iter(dataloader))
        assert isinstance(batch_data, torch.Tensor)
        assert batch_data.shape == (2, 3, 64, 64)
