import torch
import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics(model: str, start_epoch: int):
    base_path = 'E:/ML/'
    measures_folder_path = base_path + f"Measures/{model}"
    # Load metrics from the file
    metrics = torch.load(f"{measures_folder_path}/{model}.pt")

    losses_training = metrics["losses_training"][start_epoch:]
    accuracies_training = metrics["accuracies_training"][start_epoch:]
    losses_validation = metrics["losses_validation"][start_epoch:]
    accuracies_validation = metrics["accuracies_validation"][start_epoch:]

    epochs_range = range(start_epoch + 1, len(losses_training) + start_epoch + 1)

    # Setting Seaborn style
    sns.set_style("whitegrid")

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training and validation loss
    ax[0].plot(epochs_range, losses_training, 'go-', label='Training Loss', linewidth=2, markersize=8)
    ax[0].plot(epochs_range, losses_validation, 'bo-', label='Validation Loss', linewidth=2, markersize=8)
    ax[0].set_title('Losses', fontweight='bold', fontsize=16, pad=20)
    ax[0].set_xlabel('Epochs', fontsize=14, labelpad=10)
    ax[0].set_ylabel('Loss', fontsize=14, labelpad=10)
    ax[0].legend()

    # Plot training and validation accuracy
    ax[1].plot(epochs_range, accuracies_training, 'go-', label='Training Accuracy', linewidth=2, markersize=8)
    ax[1].plot(epochs_range, accuracies_validation, 'bo-', label='Validation Accuracy', linewidth=2, markersize=8)
    ax[1].set_title('Accuracies', fontweight='bold', fontsize=16, pad=20)
    ax[1].set_xlabel('Epochs', fontsize=14, labelpad=10)
    ax[1].set_ylabel('Accuracy', fontsize=14, labelpad=10)
    ax[1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    model = "LeNet_5"
    start_epoch = 0  # for example, start from the second epoch
    plot_metrics(model, start_epoch)
