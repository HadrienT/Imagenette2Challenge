from PIL import Image
from utils import pathsBuilder


def test_valid_image():
    # test a valid RGB JPEG image
    with Image.new('RGB', (256, 256)) as img:
        img.save('test.jpg')
        assert pathsBuilder.valid_image('test.jpg')

    # test a valid grayscale JPEG image
    with Image.new('L', (256, 256)) as img:
        img.save('test.jpg')
        assert not pathsBuilder.valid_image('test.jpg')

    # test a non-existent file
    assert not pathsBuilder.valid_image('nonexistent.jpg')

    # test a non-JPEG file
    with Image.new('RGB', (256, 256)) as img:
        img.save('test.png')
        assert not pathsBuilder.valid_image('test.png')
