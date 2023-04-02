import numpy as np
from PIL import Image

def cvtColor(image):
    if len(np.shape(image)) == 3 and image.shape[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image, size):
    w, h = size
    new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def preprocess_input(image):
    image /= 255.0
    return image

def show_config(**kwargs):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in kwargs.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def get_new_image_size(height, width, img_min_size=600):
    if height >= width:
        f = float(img_min_size) / width
        resize_width = int(img_min_size)
        resize_height = int(f * height)
    else:
        f = float(img_min_size) / height
        resize_height = int(img_min_size)
        resize_width = int(f * width)
    return resize_height, resize_width