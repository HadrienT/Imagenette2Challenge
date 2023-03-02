import torch
import argparse
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import types

def parse_arguments() -> argparse.Namespace:
   # Define the argument parser
    parser = argparse.ArgumentParser(description='Train or evaluate a model')
    # Add arguments
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'infer'], help='mode to run the script in')
    parser.add_argument('--model', type=str, default='LeNet_5_0', help='model to use')
    parser.add_argument('--epochs', type=int, default=5, help='number of training epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint to use')
    parser.add_argument('--figures', type=bool,action=argparse.BooleanOptionalAction, help='Save figures (--no-figures to not display figures)')
    parser.add_argument('--transformed',type=bool, action=argparse.BooleanOptionalAction,help='Select which dataset to use (--no-transformed for raw images)')
    parser.add_argument('--metric' , type=int, default=1, help='Number of most probable classes to correctly classify (default: 1)')
    return parser.parse_args()





def load_dataset(CustomDataset_module:types.ModuleType,file:str, transform:transforms, args:argparse.Namespace) -> DataLoader:
    labels = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
    label_to_index = {label: index for index, label in enumerate(labels)}
    
    with open(file, 'r') as f:
        image_paths = [] # List of image file paths
        labels = [] # List of labels corresponding to each image
        for line in f:
            image_paths.append(line.split(',')[0])
            labels.append(label_to_index.get(line.split(',')[1].replace('\n', ''))) # Remove the newline character from the label and convert it to an integer
    f.close()  
    
    data_training = CustomDataset_module.CustomDataset(image_paths, labels, transform)
    # Create a dataloader to load the data in batches
    dataloader_training = torch.utils.data.DataLoader(data_training, batch_size=args.batch_size, shuffle=True,num_workers=6)
    return dataloader_training
    
def compute_accuracy(output:torch.Tensor, labels:torch.Tensor,args:argparse.Namespace):
    _, predicted = torch.topk(output, k=args.metric, dim=1)
    correct = (predicted == labels.view(-1, 1)).sum().item()
    return correct
    # _, predicted = torch.topk(output.data, k=args.metric, dim=1)
    # predicted = predicted.t()
    # correct = predicted.eq(labels.view(1, -1).expand_as(predicted))
    # correct_1 = correct[:1].view(-1).float().sum(0, keepdim=True)
    # correct_2 = correct[1:].view(-1).float().sum(0, keepdim=True)
    # correct_both = (correct_1 + correct_2).item()
    # accuracy = correct_both / labels.size(0)
    # return accuracy