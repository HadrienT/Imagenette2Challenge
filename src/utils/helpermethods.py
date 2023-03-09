import torch
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import multiprocessing
import types
import stat
import os 
from typing import Any

def parse_arguments() -> argparse.Namespace:
   # Define the argument parser
    parser = argparse.ArgumentParser(description='Train or evaluate a model')
    # Add arguments
    parser.add_argument('--model', type=str, default='LeNet_5_0', help='model to use')
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to use')
    parser.add_argument('--figures', type=bool,action=argparse.BooleanOptionalAction, help='Save figures (--no-figures to not display figures)')
    parser.add_argument('--transformed',type=bool, action=argparse.BooleanOptionalAction,help='Select which dataset to use (--no-transformed for raw images)')
    parser.add_argument('--metric' , type=int, default=1, help='Number of most probable classes to correctly classify (default: 1)')
    return parser.parse_args()

def load_dataset(CustomDataset_module:types.ModuleType,file:str, transform:transforms, args:argparse.Namespace) -> DataLoader[Any]:
    labels = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
    label_to_index = {label: index for index, label in enumerate(labels)}
    
    with open(file, 'r') as f:
        image_paths = [] # List of image file paths
        classes = [] # List of labels corresponding to each image
        for line in f:
            image_paths.append(line.split(',')[0])
            classes.append(label_to_index.get(line.split(',')[1].replace('\n', ''))) # Remove the newline character from the label and convert it to an integer
    f.close()   
    
    # Create a dataloader to load the data in batches
    data_training = CustomDataset_module.CustomDataset(image_paths, classes, transform)
    dataloader_training = torch.utils.data.DataLoader(data_training, batch_size=args.batch_size, shuffle=True,num_workers=6)

    return dataloader_training
    
def compute_accuracy(output:torch.Tensor, labels:torch.Tensor,args:argparse.Namespace) -> float:
    _, predicted = torch.topk(output, k=args.metric, dim=1)
    correct = (predicted == labels.view(-1, 1)).sum().item()
    return float(correct)
       
def make_folder(path:str) -> None:

    # Check if the user has write permission for the parent directory
    parent_dir = os.path.dirname(path)
    if not os.access(parent_dir, os.W_OK):
        raise PermissionError(f"You do not have permission to write to {parent_dir}")

    # Check if the measures folder exists, and create it if necessary
    if not os.path.isdir(path):
        try:
            os.makedirs(path, exist_ok=True)
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR) # set permissions to owner only
        except OSError as e:
            raise OSError(f"Failed to create measures folder: {e}")
        
def send_result(queue:multiprocessing.Queue, result:list[int]) -> None:
    queue.put(result)