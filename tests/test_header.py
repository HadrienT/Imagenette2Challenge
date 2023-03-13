import argparse
from datetime import datetime

from utils import Header


def test_print_header() -> None:
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
