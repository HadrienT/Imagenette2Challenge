import argparse
from datetime import datetime

from utils import Header


def test_print_header() -> None:
    """
    Test function for the print_header() method in the Header class.
    Only checks if the method runs without raising any exceptions.
    Raises:
        AssertionError: If an exception is raised during the execution of the print_header() method.
    """
    args = argparse.Namespace(model='resnet', epochs=10, batch_size=32, metric='accuracy',
                              transformed=True, figures=False)
    params = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'device': 'cuda',
        'checkpoint_path': '/path/to/checkpoint',
        'NUMBER_CLASSES': 10,
        'num_params': 500000
    }

    try:
        Header.print_header(args, params)
    except Exception as e:
        assert False, f"print_header() raised {e.__class__.__name__} unexpectedly!"
