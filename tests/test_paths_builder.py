from PIL import Image
import tempfile
import os

from utils import pathsBuilder


def test_valid_image() -> None:
    """
    Test the valid_image function in the pathsBuilder module.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # test a valid RGB JPEG image
        with Image.new('RGB', (256, 256)) as img:
            img.save(os.path.join(tmpdir, 'test.JPEG'))
            assert pathsBuilder.valid_image(os.path.join(tmpdir, 'test.JPEG'))

        # test a valid grayscale JPEG image
        with Image.new('L', (256, 256)) as img:
            img.save(os.path.join(tmpdir, 'test.JPEG'))
            assert not pathsBuilder.valid_image(os.path.join(tmpdir, 'test.JPEG'))

        # test a non-existent file
        assert not pathsBuilder.valid_image('nonexistent.JPEG')

        # test a non-JPEG file
        with Image.new('RGB', (256, 256)) as img:
            img.save(os.path.join(tmpdir, 'test.png'))
            assert not pathsBuilder.valid_image(os.path.join(tmpdir, 'test.png'))


def test_create_path_transformed() -> None:
    """
    Test the create_path_transformed function in the pathsBuilder module.
    """
    # create temporary directories for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        root = tmpdir
        train_dir = os.path.join(root, 'transformed', 'train')
        val_dir = os.path.join(root, 'transformed', 'val')
        os.makedirs(os.path.join(train_dir, 'tench'))
        os.makedirs(os.path.join(val_dir, 'tench'))

        # create sample images
        with Image.new('RGB', (256, 256)) as img:
            img.save(os.path.join(train_dir, 'tench', '0.JPEG'))
        with Image.new('RGB', (256, 256)) as img:
            img.save(os.path.join(val_dir, 'tench', '0.JPEG'))

        # run the function
        pathsBuilder.create_path_transformed(root)

        # assert that the output files were created correctly
        with open(os.path.join(root, "transformed", "train.txt")) as f:
            train_txt = f.read()
            assert os.path.join(root, "transformed\\train\\tench\\0.JPEG,tench\n").replace('\\', '\\\\') in train_txt
        with open(os.path.join(root, "transformed", "val.txt")) as f:
            val_txt = f.read()
            assert os.path.join(root, "transformed\\val\\tench\\0.JPEG,tench\n").replace('\\', '\\\\') in val_txt


def test_create_path_raw() -> None:
    """
    Test the create_path_raw function in the pathsBuilder module.
    """
    # create temporary directories for testing
    with tempfile.TemporaryDirectory() as tmpdir:
        root = tmpdir
        train_dir = os.path.join(root, 'train')
        val_dir = os.path.join(root, 'val')
        os.makedirs(os.path.join(train_dir, 'tench'))
        os.makedirs(os.path.join(val_dir, 'tench'))

        # create sample images
        with Image.new('RGB', (256, 256)) as img:
            img.save(os.path.join(train_dir, 'tench', '0.JPEG'))
        with Image.new('RGB', (256, 256)) as img:
            img.save(os.path.join(val_dir, 'tench', '0.JPEG'))

        # run the function
        pathsBuilder.create_path_raw(root)

        # assert that the output files were created correctly
        with open(os.path.join(root, "train.txt")) as f:
            train_txt = f.read()
            assert os.path.join(root, "train\\tench\\0.JPEG,tench\n").replace('\\', '\\\\') in train_txt
        with open(os.path.join(root, "val.txt")) as f:
            val_txt = f.read()
            assert os.path.join(root, "val\\tench\\0.JPEG,tench\n").replace('\\', '\\\\') in val_txt
