import os
import torch
import torchvision.transforms as transforms
from PIL import Image

def main():
    folders = ['n01440764','n02102040','n02979186','n03000684','n03028079','n03394916','n03417042',
    'n03425413','n03445777','n03888257']

    train_dir = './imagenette2/transformed/train/'
    val_dir = './imagenette2/transformed/val/'

    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    if not os.path.exists(val_dir):
        os.mkdir(val_dir)

    for folder in folders:
        train_folder = train_dir + folder
        val_folder = val_dir + folder

        if not os.path.exists(train_folder):
            os.mkdir(train_folder)

        if not os.path.exists(val_folder):
            os.mkdir(val_folder)


    # Define the transformation to be applied on the images
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])


    root_dir = "./imagenette2"
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
                image = Image.open(image_path)
                target_name = dst_image_path.replace(".JPEG", ".pt")
                if image.mode == 'RGB':
                    tensor = transform(image)
                    torch.save(tensor, target_name)

if __name__ == "__main__":
    main()