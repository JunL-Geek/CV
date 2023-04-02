import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from utils import cvtColor, preprocess_input

class FRCNNDataset(Dataset):
    def __init__(self, annotation_lines, input_shape=[600, 600], train=True):
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.train = train

    def __len__(self):
        return self.length

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=.7, val=0.4, random=True):
        line = annotation_line.split()

        image = Image.open(line[0])
        image = cvtColor(Image)

        iw, ih = image.size
        h, w = self.input_shape

        box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])

        # 验证，测试阶段使用
        if not random:
            scale = min(w/iw, h/ih)
            nw = int(scale * iw)
            nh = int(scale * ih)
            dx = (w - nw) / 2
            dy = (h - nh) / 2

            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)



    def __getitem__(self, index):
        index = index % self.length
        image, y = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random=self.train)
        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box_data = np.zeros((len(y), 5))