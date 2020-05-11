import torch.utils.data as data
import os
from PIL import Image


class SubImageNet(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.root = os.path.join(self.root, 'SubImageNet/Imagenet')
        self.root = os.path.join(self.root, 'train' if train else 'val')

        self.images = []
        self.labels = []

        for label in os.listdir(self.root):
            for image_name in os.listdir(os.path.join(self.root, label)):
                self.images.append(Image.open(os.path.join(self.root, label, image_name)).convert('RGB'))
                self.labels.append(int(label))

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
