from typing import List
import argparse

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# plot metrics over time


def plot_metrics(losses_training: List[float], accuracies_training: List[float],
                 losses_validation: List[float], accuracies_validation: List[float],
                 args: argparse.Namespace) -> None:
    """
    Plot the training and validation losses and accuracies over time.

    Args:
        losses_training (List[float]): List of training losses.
        accuracies_training (List[float]): List of training accuracies.
        losses_validation (List[float]): List of validation losses.
        accuracies_validation (List[float]): List of validation accuracies.
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """
    abscisse = np.arange(1, args.epochs+1)
    _, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), tight_layout=True)
    ax = ax.flatten()
    ax[0].plot(abscisse, losses_training, label='Training loss', color='b', linestyle='-', marker='o')
    ax[0].plot(abscisse, losses_validation, label='Validation loss', color='r', linestyle='-', marker='o')
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].grid(linestyle='--', linewidth=0.5)
    ax[0].hlines(y=0, xmin=0, xmax=args.epochs, color='k', linestyles='-')
    ax[0].legend()
    ax[0].set_xlim([1, args.epochs])

    ax[1].plot(abscisse, accuracies_training, label='Training accuracy', color='b', linestyle='-', marker='o')
    ax[1].plot(abscisse, accuracies_validation, label='Validation accuracy', color='r', linestyle='-', marker='o')
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].grid(linestyle='--', linewidth=0.5)
    ax[1].set_ylim([-0.05, 1])
    ax[1].hlines(y=0, xmin=0, xmax=args.epochs, color='k', linestyles='-')
    ax[1].legend()
    ax[1].set_xlim([1, args.epochs])

    plt.show()


def plot_time_measures(file_path: str) -> None:
    """
    Plot the time measures from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        None
    """
    data_type = {'num_epoch': int,
                 'image (ms)': np.float64,
                 'criterion (ms)': np.float64,
                 'optimizer (ms)': np.float64,
                 'accuracy (ms)': np.float64,
                 'checkpoint (ms)': np.float64,
                 'epoch (s)': np.float64
                 }

    df = pd.read_csv(file_path, sep=',', dtype=data_type)
    cols = df.columns[1:]
    _, ax = plt.subplots(2, 3, figsize=(15, 10))
    ax = ax.flatten()
    for i, col in enumerate(cols):
        ax[i].plot(list(range(1, len(df[col])+1)), df[col], label=f'{col}')
        ax[i].grid(linestyle='--', linewidth=0.5)
        ax[i].hlines(y=df[col].mean(), xmin=0, xmax=len(df[col]), color='r', linestyles='-')
        ax[i].set_xlim([1, len(df[col])])
        ax[i].set_xlabel("Epoch")
        ax[i].set_ylabel("Time (ms)")
        ax[i].legend()
    plt.show()
