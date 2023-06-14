import argparse
from typing import Any


def print_header(args: argparse.Namespace, params: dict[Any, Any]) -> None:
    """
    Prints the header information for the image classification.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.
        params (dict[Any, Any]): The dictionary of parameters independant of the CLI.

    Returns:
        None
    """
    print("##########################################################################")
    print("#                                                                        #")
    print("#                          Image Classification                          #")
    print("#                                                                        #")
    print(f"#                          Date: {params.get('date')}                     #")
    print("#                                                                        #")
    print("##########################################################################")
    print()
    print("Hyperparameters:")
    print(f"- Model: {args.model}")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Metric: {args.metric}")
    if hasattr(args, 'transformed'):
        print(f"- Transformed: {args.transformed}")
    if hasattr(args, 'figures'):
        print(f"- Figures: {args.figures}")
    print()
    print("Configuration:")
    print(f"- Device: {params.get('device')}")
    print(f"- Checkpoint: {params.get('checkpoint_path')}")
    print(f"- Number of classes: {params.get('NUMBER_CLASSES')}")
    print(f"- Number of trainable parameters: {params.get('num_params')}")
    print()
    print("##########################################################################")
    print()
