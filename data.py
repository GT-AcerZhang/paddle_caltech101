import paddle
import os
import numpy as np
from PIL import Image, ImageEnhance 


class Dataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._train_ls = self._load_dict(os.path.join(data_dir, "Train.txt"))
        self._test_ls = self._load_dict(os.path.join(data_dir, "Test.txt"))
        self._eval_ls = self._load_dict(os.path.join(data_dir, "Eval.txt"))
        self._image_dir = os.path.join(data_dir, "Images")

    def _load_dict(self, file_path):
        ls = []
        with open(file_path, "r", encoding="utf8") as f:
            for line in f.readlines():
                line = line.strip()
                if len(line) == 0:
                    continue

                ss = list(line.split("\t"))
                if len(ss) > 1:
                    ss[1] = int(ss[1])
                ls.append(ss)
        return ls

    def train_data(self):
        np.random.shuffle(self._train_ls)
        for img, label in self._train_ls:
            img = Image.open(os.path.join(self._image_dir, img))
            img = img.resize([224, 224])
            yield np.array(img), label


if __name__ == '__main__':
    dataset = Dataset("data")
    for data in dataset.train_data():
        pass
