import os
import torch
import torchvision.transforms as transforms
from PIL import Image


def get_class_folders(root: str) -> list[str]:
    return os.listdir(root)


def make_hierarchy(root: str) -> None:
    folders = get_class_folders(os.path.join(root, 'train'))

    train_dir = os.path.join(root, 'transformed/train')
    val_dir = os.path.join(root, 'transformed/val')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    for folder in folders:
        train_folder = os.path.join(train_dir, folder)
        val_folder = os.path.join(val_dir, folder)

        if not os.path.exists(train_folder):
            os.makedirs(train_folder)

        if not os.path.exists(val_folder):
            os.makedirs(val_folder)


def image_to_tensor(image_path: str, target_name: str, transform: transforms.Compose) -> None:
    image = Image.open(image_path)
    if image.mode == 'RGB':
        tensor = transform(image)
        torch.save(tensor, target_name)


def convert_dataset(root_dir: str) -> None:
    # Define the transformation to be applied on the images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for folder in ["train", "val"]:
        src_folder = os.path.join(root_dir, folder)
        dst_folder = os.path.join(root_dir, "transformed", folder)
        for subfolder in os.listdir(src_folder):
            subfolder_path = os.path.join(src_folder, subfolder)
            dst_subfolder_path = os.path.join(dst_folder, subfolder)
            if not os.path.exists(dst_subfolder_path):
                os.makedirs(dst_subfolder_path)
            for image_file in os.listdir(subfolder_path):
                image_path = os.path.join(subfolder_path, image_file)
                dst_image_path = os.path.join(dst_subfolder_path, image_file)
                target_name = dst_image_path.replace(".JPEG", ".pt")
                image_to_tensor(image_path, target_name, transform)


def main() -> None:
    root = "E:/ML/Datasets/imagenette2"
    make_hierarchy(root)
    convert_dataset(root)


if __name__ == "__main__":
    main()
