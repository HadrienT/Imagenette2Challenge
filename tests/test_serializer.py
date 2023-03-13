import tempfile
import os
import shutil
import random

import pytest
import torch
from torchvision import transforms
import PIL.Image as Image

from utils import SerializeDataset


def test_get_class_folders() -> None:
    folders = SerializeDataset.get_class_folders('E:\\ML\\Datasets\\imagenette2\\train')
    target = ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']
    assert folders == target


def test_make_hierarchy(monkeypatch: pytest.MonkeyPatch) -> None:

    def mock_get_class_folders(root) -> list[str]:
        return ['n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079', 'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257']

    monkeypatch.setattr(SerializeDataset, 'get_class_folders', mock_get_class_folders)
    with tempfile.TemporaryDirectory() as tempdir:
        SerializeDataset.make_hierarchy(tempdir)
        train_dir = os.path.join(tempdir, 'transformed', 'train')
        val_dir = os.path.join(tempdir, 'transformed', 'val')
        folders = mock_get_class_folders(None)
        for folder in folders:
            train_folder = os.path.join(train_dir, folder)
            val_folder = os.path.join(val_dir, folder)
            assert os.path.exists(train_folder)
            assert os.path.exists(val_folder)


def test_image_to_tensor() -> None:
    root = ".\\tests\\test_images"
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    paths = os.listdir(root)
    assert len(paths) > 0 # test
    nb_color_images = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        for path in paths:
            if Image.open(os.path.join(root, path)).mode == "RGB":
                nb_color_images += 1
                img_path = os.path.join(root, path)
                target_name = os.path.join(tmpdir, path.replace(".JPEG", ".pt"))
                SerializeDataset.image_to_tensor(img_path, target_name, transform)
                assert os.path.exists(target_name)
                # Load the saved tensor and check its shape
                tensor = torch.load(target_name)
                assert tensor.shape == (3, 256, 256)
        # Test if grey images are not saved
        assert len(os.listdir(tmpdir)) == nb_color_images


def test_convert_dataset() -> None:
    root = "E:\\ML\\Datasets\\imagenette2"
    folders = ['train', 'val']

    # Create a skeleton of the dataset
    with tempfile.TemporaryDirectory() as temp_dir:
        for folder in folders:
            sub_folders = os.listdir(os.path.join(root, folder))
            for class_folder in sub_folders:
                os.makedirs(os.path.join(temp_dir, folder, class_folder))
                for _ in range(10):
                    random_file = random.choice(os.listdir(os.path.join(root, folder, class_folder)))
                    path_to_file = os.path.join(root, folder, class_folder, random_file)
                    target_path = os.path.join(temp_dir, folder, class_folder, random_file)
                    shutil.copy(path_to_file, target_path)

        SerializeDataset.convert_dataset(temp_dir)
        root = os.path.join(temp_dir, 'transformed')
        assert os.path.exists(os.path.join(temp_dir, 'transformed'))
        for folder in folders:
            sub_folders = os.listdir(os.path.join(root, folder))
            for class_folder in sub_folders:
                for file in os.listdir(os.path.join(root, folder, class_folder)):
                    file_path = os.path.join(root, folder, class_folder, file)
                    assert file.endswith('.pt')
                    assert torch.load(file_path).shape == (3, 256, 256)
