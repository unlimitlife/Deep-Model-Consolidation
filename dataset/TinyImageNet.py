import torch.utils.data as data
import os
from PIL import Image


class TinyImageNet(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        self.root = os.path.join(self.root, 'tiny-imagenet-200')

        self.images = []
        self.labels = []
        class_list = os.listdir(os.path.join(self.root, 'train'))
        class_list.sort()

        if train:
            self.root = os.path.join(self.root, 'train')
            for name_of_class in class_list:
                for image_name in os.listdir(os.path.join(self.root, name_of_class, 'images')):
                    self.images.append(
                        Image.open(os.path.join(self.root, name_of_class, 'images', image_name)).convert('RGB'))
                    self.labels.append(class_list.index(name_of_class))
        else:
            self.root = os.path.join(self.root, 'val')
            with open(os.path.join(self.root, 'val_annotations.txt')) as f:
                for line in f.readlines():
                    self.images.append(
                        Image.open(os.path.join(self.root, 'images', line.split('\t')[0])).convert('RGB'))
                    self.labels.append(class_list.index(line.split('\t')[1]))

        assert (len(self.images) == len(self.labels))

    def __getitem__(self, idx):
        image, label = self.images[idx], self.labels[idx]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.images)
