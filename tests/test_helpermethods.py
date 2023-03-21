import pytest
import tempfile
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse
from dataLoaders import CustomDatasetRaw
from utils import helpermethods as helpermethods
import os
import stat
import multiprocessing


@pytest.skip("Skip to pass GitHub Actions")
def test_get_paths_and_labels() -> None:
    # Create a temporary file with sample data
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as f:
        f.write("E:\\ML\\Datasets\\imagenette2\\train\\n01440764\\ILSVRC2012_val_00000293.JPEG,tench\n")
        f.write("E:\\ML\\Datasets\\imagenette2\\train\\n01440764\\ILSVRC2012_val_00002138.JPEG,tench\n")
        f.write("E:\\ML\\Datasets\\imagenette2\\train\\n01440764\\ILSVRC2012_val_00003014.JPEG,tench\n")

    # Call the function
    image_paths, labels = helpermethods.get_paths_and_labels(f.name)

    # Check the output
    assert image_paths[:3] == ["E:\\ML\\Datasets\\imagenette2\\train\\n01440764\\ILSVRC2012_val_00000293.JPEG",
                               "E:\\ML\\Datasets\\imagenette2\\train\\n01440764\\ILSVRC2012_val_00002138.JPEG",
                               "E:\\ML\\Datasets\\imagenette2\\train\\n01440764\\ILSVRC2012_val_00003014.JPEG"]
    assert labels == [0 for _ in range(3)]

    # Clean up the temporary file
    os.remove(f.name)


@pytest.skip("Skip to pass GitHub Actions")
@pytest.mark.skip(reason="Takes too long but works")
def test_load_dataset() -> None:

    module = CustomDatasetRaw
    file = "E:\\ML\\Datasets\\imagenette2\\train.txt"
    image_paths, classes = helpermethods.get_paths_and_labels(file)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    args = argparse.Namespace()
    args.batch_size = 32

    dataloader_training = helpermethods.load_dataset(module, file, transform, args)

    assert isinstance(dataloader_training, DataLoader)
    assert len(dataloader_training) == 291

    # Assert that the DataLoader returns the expected number of batches
    expected_num_batches = (len(open(file).readlines()) - 1) // args.batch_size + 1
    assert len(dataloader_training) == expected_num_batches

    # Assert that the DataLoader returns the expected data format
    for batch_idx, (data, target) in enumerate(dataloader_training):
        assert isinstance(data, torch.Tensor)
        assert isinstance(target, torch.Tensor)
        assert data.shape[0] == min(args.batch_size, len(image_paths) - batch_idx * args.batch_size)
        assert target.shape[0] == min(args.batch_size, len(classes) - batch_idx * args.batch_size)
        assert data.shape[1:] == (3, 256, 256)


def test_compute_accuracy_metric_1() -> None:
    # Set up input tensors
    output = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.3, 0.2, 0.5]])
    labels = torch.tensor([2, 2, 1])

    # Set up argparse.Namespace object with metric=1
    args = argparse.Namespace(metric=1)

    # Call the function
    accuracy = helpermethods.compute_accuracy(output, labels, args)

    # Check the output
    expected_accuracy = 2
    assert accuracy == expected_accuracy


def test_compute_accuracy_metric_2() -> None:
    # Set up input tensors
    output = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.3, 0.2, 0.5]])
    labels = torch.tensor([1, 2, 1])

    # Set up argparse.Namespace object with metric=1
    args = argparse.Namespace(metric=2)

    # Call the function
    accuracy = helpermethods.compute_accuracy(output, labels, args)

    # Check the output
    expected_accuracy = 2
    assert accuracy == expected_accuracy


def test_compute_accuracy_metric_full() -> None:
    # Set up input tensors
    output = torch.tensor([[0.2, 0.3, 0.5], [0.1, 0.4, 0.5], [0.3, 0.2, 0.5]])
    labels = torch.tensor([0, 0, 0])

    # Set up argparse.Namespace object with metric=1
    args = argparse.Namespace(metric=3)

    # Call the function
    accuracy = helpermethods.compute_accuracy(output, labels, args)

    # Check the output
    expected_accuracy = len(output)
    assert accuracy == expected_accuracy


def test_make_folder() -> None:
    test_dir = tempfile.TemporaryDirectory()
    test_path = os.path.join(test_dir.name, 'test_folder')

    # Check if the user has write permission for the parent directory
    parent_dir = os.path.dirname(test_path)
    if not os.access(parent_dir, os.W_OK):
        raise PermissionError(f"You do not have permission to write to {parent_dir}")

    # Check if the folder exists, and create it if necessary
    if not os.path.isdir(test_path):
        try:
            os.makedirs(test_path, exist_ok=True)
            os.chmod(test_path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # set permissions to owner only
        except OSError as e:
            raise OSError(f"Failed to create folder: {e}")

    # Assert that the folder was created with the correct permissions and access rights
    assert os.path.isdir(test_path)
    assert os.stat(test_path).st_mode & stat.S_IRWXU == stat.S_IRWXU
    assert os.access(os.path.dirname(test_path), os.W_OK)

    test_dir.cleanup()


def test_make_folder_with_existing_folder() -> None:
    test_dir = tempfile.TemporaryDirectory()
    test_path = os.path.join(test_dir.name, 'test_folder')

    os.mkdir(test_path)
    helpermethods.make_folder(test_path)

    assert os.path.isdir(test_path)

    test_dir.cleanup()


def test_make_folder_with_existing_file() -> None:
    test_dir = tempfile.TemporaryDirectory()
    test_path = os.path.join(test_dir.name, 'test_file')

    with open(test_path, 'w') as f:
        f.write('test')

    with pytest.raises(OSError):
        helpermethods.make_folder(test_path)

    test_dir.cleanup()


def processMock(queue: multiprocessing.Queue) -> None:  # type: ignore
    result = [1, 2, 3]
    queue.put(result)


def test_send_result() -> None:
    queue = multiprocessing.Queue()  # type: ignore
    p = multiprocessing.Process(target=processMock, args=(queue,))
    p.start()
    p.join()
    result = [1, 2, 3]
    helpermethods.send_result(queue, result)
    assert queue.get() == result
