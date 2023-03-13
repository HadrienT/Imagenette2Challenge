import os
import PIL.Image as Image
import argparse


def valid_image(path: str) -> bool:
    try:
        with Image.open(path) as img:
            return img.format == 'JPEG' and img.mode == 'RGB'  # type: ignore
    except (IOError, SyntaxError):
        return False


def create_path_transformed(root: str) -> None:
    labels = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']
    root = os.path.join(root, 'transformed')

    with open(root + '/train.txt', 'w') as f:
        filename = 'train'
        for idx, im_class in enumerate(os.listdir(os.path.join(root, filename))):
            for image in os.listdir(os.path.join(root, filename, im_class)):
                full_path = os.path.join(root, filename, im_class, image)
                full_path = full_path + ',' + labels[idx] + '\n'
                full_path = full_path.replace('\\', '\\\\')
                f.write(full_path)

    with open(root + '/val.txt', 'w') as f:
        filename = 'val'
        for idx, im_class in enumerate(os.listdir(os.path.join(root, filename))):
            for image in os.listdir(os.path.join(root, filename, im_class)):
                full_path = os.path.join(root, filename, im_class, image)
                full_path = full_path + ',' + labels[idx] + '\n'
                full_path = full_path.replace('\\', '\\\\')
                f.write(full_path)


def create_path_raw(root: str) -> None:
    labels = ['tench', 'English springer', 'cassette player', 'chain saw', 'church', 'French horn', 'garbage truck', 'gas pump', 'golf ball', 'parachute']

    with open(root + '/train.txt', 'w') as f:
        filename = 'train'
        for idx, im_class in enumerate(os.listdir(os.path.join(root, filename))):
            for image in os.listdir(os.path.join(root, filename, im_class)):
                full_path = os.path.join(root, filename, im_class, image)
                if valid_image(full_path):
                    full_path = full_path + ',' + labels[idx] + '\n'
                    full_path = full_path.replace('\\', '\\\\')
                    f.write(full_path)

    with open(root + '/val.txt', 'w') as f:
        filename = 'val'
        for idx, im_class in enumerate(os.listdir(os.path.join(root, filename))):
            for image in os.listdir(os.path.join(root, filename, im_class)):
                full_path = os.path.join(root, filename, im_class, image)
                if valid_image(full_path):
                    full_path = full_path + ',' + labels[idx] + '\n'
                    full_path = full_path.replace('\\', '\\\\')
                    f.write(full_path)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build paths files')
    parser.add_argument('--transformed', type=bool, action=argparse.BooleanOptionalAction, help='Make paths for transformed or raw images')
    root = 'E:\\ML\\Datasets\\imagenette2\\'
    args = parser.parse_args()
    if args.transformed:
        create_path_transformed(root)
    else:
        create_path_raw(root)


if __name__ == '__main__':
    main()
