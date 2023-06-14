import argparse
import multiprocessing
import types
import stat
import os
from typing import Any, Tuple, List

import torch
from torch.utils.data import DataLoader


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for model training or evaluation.

    :return: Namespace object with argument values as attributes.
    """
    # Define the argument parser
    parser = argparse.ArgumentParser(description='Train or evaluate a model')
    # Add arguments
    parser.add_argument('--model', type=str, default='LeNet_5_0', help='model to use')
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to use')
    parser.add_argument('--figures', type=bool, action=argparse.BooleanOptionalAction, help='Save figures (--no-figures to not display figures)')
    parser.add_argument('--transformed', type=bool, action=argparse.BooleanOptionalAction, help='Select which dataset to use (--no-transformed for raw images)')
    parser.add_argument('--metric', type=int, default=1, help='Number of most probable classes to correctly classify (default: 1)')
    return parser.parse_args()


def get_paths_and_labels(file: str) -> Tuple[List[str], List[int]]:
    """
    Parse a file for image paths and corresponding class labels.

    :param file: Path to the file containing image paths and labels.
    :return: Tuple of lists containing image paths and labels.
    """
    labels = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
    label_to_index = {label: index for index, label in enumerate(labels)}

    with open(file, 'r') as f:
        image_paths = []  # List of image file paths
        classes = []  # List of labels corresponding to each image
        for line in f:
            target = label_to_index.get(line.split(',')[1].replace('\n', ''))
            if target is not None:
                image_paths.append(line.split(',')[0])
                # Remove the newline character from the label and convert it to an integer
                classes.append(target)
    return image_paths, classes


def load_dataset(CustomDataset_module: types.ModuleType, file: str, transform: Any) -> DataLoader[Any]:
    """
    Load a dataset from a file and apply a transformation.

    :param CustomDataset_module: Module containing the CustomDataset class.
    :param file: Path to the file containing image paths and labels.
    :param transform: Transform to apply to the images.
    :return: DataLoader object for the dataset.
    """
    image_paths, classes = get_paths_and_labels(file)
    data_training = CustomDataset_module.CustomDataset(image_paths, classes, transform)
    return data_training


def compute_accuracy(output: torch.Tensor, labels: torch.Tensor, top_targets: int) -> float:
    """
    Compute the accuracy of the model's output.

    :param output: The model's output.
    :param labels: Ground truth labels.
    :param args: Command-line arguments.
    :return: Accuracy of the model's output.
    """
    _, predicted = torch.topk(output, k=top_targets, dim=1)
    correct = torch.eq(predicted, labels.view(-1, 1).expand_as(predicted)).sum().item()
    return float(correct)


def compute_accuracy_IPU(output: torch.Tensor, labels: torch.Tensor, top_targets: int, device_iterations: int) -> float:
    """
    Compute the accuracy of the model's output on an IPU device.

    :param output: The model's output.
    :param labels: Ground truth labels.
    :param args: Command-line arguments.
    :param device_iterations: Number of device iterations.
    :return: Accuracy of the model's output.
    """
    # Reshape labels to match output's batch size
    labels = labels.view(device_iterations, -1, 1)
    _, predicted = torch.topk(output, k=top_targets, dim=1)
    # Calculate the number of correct predictions for each mini-batch and sum them up
    correct = torch.eq(predicted, labels).sum().item()
    return float(correct)


def make_folder(path: str) -> None:
    """
    Create a new folder at the specified path.

    :param path: Path to the folder.
    """

    # Check if the measures folder exists, and create it if necessary
    if not os.path.isdir(path):
        try:
            os.makedirs(path, exist_ok=True)
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)  # set permissions to owner only
        except OSError as e:
            raise OSError(f"Failed to create measures folder: {e}")


def send_result(queue: multiprocessing.Queue, result: List[int]) -> None:  # type: ignore
    """
    Send a result to a multiprocessing queue.

    :param queue: Multiprocessing queue.
    :param result: Result to send to the queue.
    """
    queue.put(result)
