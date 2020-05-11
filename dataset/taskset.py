import os
import pickle
import random
import utils
import copy

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
from .TinyImageNet import TinyImageNet
from .SubImageNet import SubImageNet

_datasets = {'cifar10': torchvision.datasets.CIFAR10,
             'cifar100': torchvision.datasets.CIFAR100,
             'mnist': torchvision.datasets.MNIST,
             'stl10': lambda data_path, train, download: torchvision.datasets.STL10(data_path,
                                                                                    split='train' if train else 'test',
                                                                                    download=download),
             'tiny_image': TinyImageNet,
             'sub_image': SubImageNet}


def preprocess(data_path, dataset):
    """ If the dataset does not exist, download it and create a dataset.
        Args:
            data_path (str): root directory of dataset.
            dataset (str): name of dataset.
    """
    il_data_path = os.path.join(data_path, 'il' + dataset)
    train_path = os.path.join(il_data_path, 'train')
    val_path = os.path.join(il_data_path, 'val')

    if os.path.isdir(il_data_path):
        return

    os.makedirs(train_path)
    os.makedirs(val_path)

    train_set = _datasets[dataset](data_path, train=True, download=True)
    val_set = _datasets[dataset](data_path, train=False, download=True)

    # dump pickles for each class
    for cur_set, cur_path in [[train_set, train_path], [val_set, val_path]]:
        for idx, item in enumerate(cur_set):
            label = item[1]
            if not os.path.exists(os.path.join(cur_path, str(label))):
                os.makedirs(os.path.join(cur_path, str(label)))
            with open(os.path.join(cur_path, str(label), str(idx) + '.p'), 'wb') as f:
                pickle.dump(item, f)


class Taskset(data.Dataset):
    def __init__(self, root, task, task_idx, DA=False, train=True, transform=None, target_transform=None):
        """
        Args:
            root (str): root directory of dataset prepared for incremental learning (by preper_for_IL)
            task (list): list of classes that are assigned for the task
            task_idx (int): index of the task, ex) 2nd task among total 10 tasks
            train (bool): whether it is for train or not
            transform (callable) : transforms for dataset
            target_transform (callable) : transforms for target
        """
        if train:
            self.root = os.path.expanduser(root) + '/train'
        else:
            self.root = self.root = os.path.expanduser(root) + '/val'
        self.task = task  # task should be a list with class number such as [0, 6, 7, 10, 99, ...]
        self.task_idx = task_idx

        # label converting from original setting to the incremental setting
        self.converted_label = {}
        for i in range(len(task)):
            self.converted_label[task[i]] = i 
            #task_idx * len(task) + i

        if not os.path.isdir(self.root):
            print('Exception: there is no such directory : {}'.format(self.root))

        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        self.data = []
        self.targets = []
        self.filenames = []
        self.scores = []

        # now load the picked PIL Image and label
        for cls in task:
            file_path = self.root + '/' + str(cls)
            for file in os.listdir(file_path):
                with open(file_path + '/' + file, 'rb') as f:
                    entry = pickle.load(f)
                    self.data.append(entry[0])
                    self.targets.append(entry[1])
                    self.filenames.append(file)
                    self.scores.append(0)

        self.scores = torch.Tensor(self.scores)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, soft_label) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        target = self.converted_label[target]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def update_score(self, indices, scores):
        with torch.no_grad():
            self.scores[indices] += scores.to(torch.float)

    def reset_score(self):
        self.scores = torch.zeros_like(self.scores)

    def __len__(self):
        return len(self.data)


class DAset(data.Dataset):
    def __init__(self, base_data, transform=None, target_transform=None):
        self.data = []
        self.targets = []
        self.transform = transform
        self.target_transform = target_transform

        for idx in range(len(base_data)):
            img = base_data.data[idx]
            label = base_data[idx][1]

            temp_img_set = []
            temp_img_set.append(utils.bright(copy.deepcopy(img)))
            temp_img_set.append(transforms.ColorJitter(contrast=0.8)(copy.deepcopy(img)))

            temp_len = len(temp_img_set)
            temp_img_set.append(transforms.RandomCrop(32, padding=4)(copy.deepcopy(img)))
            for i in range(temp_len):
                temp_img_set.append(transforms.RandomCrop(32, padding=4)(copy.deepcopy(temp_img_set[i])))

            temp_len = len(temp_img_set)
            temp_img_set.append(transforms.RandomHorizontalFlip(1.0)(copy.deepcopy(img)))
            for i in range(temp_len):
                temp_img_set.append(transforms.RandomHorizontalFlip(1.0)(copy.deepcopy(temp_img_set[i])))

            self.data += temp_img_set
            self.targets += [label for _ in range(len(temp_img_set))]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.data)


##############################################################################


class VoidDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.datas = []
        self.labels = []

    def __getitem__(self, index):
        return self.datas[index].squeeze(0), self.labels[index]

    def update(self, data, label):
        self.datas = data
        self.labels = label

    def __len__(self):
        return self.datas.size(0)


##############################################################################
class FlexibleMemory(data.Dataset):
    def __init__(self, root, num_classes, num_tasks, curriculum, capacity=2000, shuffle=True,
                 transform=None, target_transform=None):
        self.root = root + '/train'
        self.capacity = capacity
        self.curriculum = curriculum
        self.task_idx = 0
        self.shuffle = shuffle
        self.step = int(num_classes / num_tasks)
        self.converted_label = {}
        for idx, curr in enumerate(curriculum):
            for i in range(len(curr)):
                self.converted_label[curr[i]] = idx * self.step + i

        self.transform = transform
        self.target_transform = target_transform

        self.imgs_for_class = []
        self.labels_for_class = []
        self.data = []
        self.targets = []

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, soft_label) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        target = self.converted_label[target]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    def add_new(self, files):
        imgs_list = []
        labels_list = []
        for i in range(len(files)):
            imgs = []
            labels = []
            cls = self.curriculum[self.task_idx][i]
            file_path = self.root + '/' + str(cls)
            for file in files[i]:
                with open(file_path + '/' + file, 'rb') as f:
                    entry = pickle.load(f)
                    imgs.append(entry[0])
                    labels.append(entry[1])
            imgs_list.append(imgs)
            labels_list.append(labels)
        self.imgs_for_class.extend(imgs_list)
        self.labels_for_class.extend(labels_list)

        self.data = []
        self.targets = []
        # print(len(self.labels_for_class))
        for i in range(self.step * (self.task_idx + 1)):
            self.data.extend(self.imgs_for_class[i])
            self.targets.extend(self.labels_for_class[i])

    def update(self, BF=False):
        # number of samples for each class
        samples_per_class = int(self.capacity / (self.task_idx if BF else self.task_idx + 1) / self.step)

        if self.task_idx >= 0:
            for i in range(len(self.imgs_for_class)):
                self.imgs_for_class[i] = self.imgs_for_class[i][:samples_per_class]
                self.labels_for_class[i] = self.labels_for_class[i][:samples_per_class]

        self.data = []
        self.targets = []
        # print(len(self.labels_for_class))
        for i in range(len(self.imgs_for_class)):
            self.data.extend(self.imgs_for_class[i])
            self.targets.extend(self.labels_for_class[i])

        self.task_idx += 1


##############################################################################

class SampleMemory(data.Dataset):
    def __init__(self, root, num_classes, num_tasks, curriculum, DA=False, capacity=2000, shuffle=True, transform=None,
                 target_transform=None):
        """
        Args:
            root (str): root directory of dataset prepared for incremental learning (by preper_for_IL)
            num_classes (int): number of classes we have
            num_tasks (int) : number of tasks defined for incremental learning
            curriculum (list of list) : list of task where task is the list of class
            capacity (int): max number of sample can SampleMemory can have
            shuffle (bool): shuffle or not
        """
        self.DA = DA
        self.root = os.path.expanduser(root + '/train')
        self.num_classes = num_classes
        self.num_tasks = num_tasks
        self.capacity = capacity
        self.step = int(num_classes / num_tasks)
        self.task_idx = 0
        self.curriculum = curriculum
        self.shuffle = shuffle

        self.transform = transform
        self.target_transform = target_transform
        self.converted_label = {}
        self.task_converted_label = {}
        for task_idx, task in enumerate(curriculum):
            for i, cls in enumerate(task):
                self.converted_label[cls] = task_idx * self.step + i
                self.task_converted_label[cls] = task_idx

        self.imgs_for_class = []
        self.labels_for_class = []
        self.data = []
        self.targets = []
        self.task_data = []
        self.task_targets = []
        self.cur_task = -1

    def __getitem__(self, index, task=-1):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target, soft_label) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])
        target = self.converted_label[target]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.data)

    def update(self, BF=False):
        task = self.curriculum[self.task_idx]
        toTensor = transforms.ToTensor()
        toImage = transforms.ToPILImage()
        mirror = transforms.RandomHorizontalFlip(1.0)
        random_crop = transforms.Compose([
            transforms.RandomCrop((24, 24)),
            transforms.Resize((32, 32))]
        )

        # number of samples for each class
        samples_per_class = int(self.capacity / (self.task_idx if BF else self.task_idx + 1) / self.step)

        if self.task_idx > 0:
            for i in range(len(self.imgs_for_class)):
                self.imgs_for_class[i] = self.imgs_for_class[i][:samples_per_class]
                self.labels_for_class[i] = self.labels_for_class[i][:samples_per_class]

        for cls in task:
            file_path = self.root + '/' + str(cls)
            files = os.listdir(file_path)
            random.shuffle(files)
            if self.DA:
                files = files[:int(samples_per_class / 12)]
            else:
                files = files[:samples_per_class]
            # For temporal use
            imgs = []
            labels = []

            for file in files:
                with open(file_path + '/' + file, 'rb') as f:
                    entry = pickle.load(f)
                    if not self.DA:
                        imgs.append(entry[0])
                        labels.append(entry[1])
                    else:
                        label = entry[1]
                        org = entry[0]
                        org = toTensor(org)
                        rand_bright = (0.5 * random.random() - 0.25) * torch.ones_like(org)
                        bright = org + rand_bright
                        bright = torch.clamp(bright, 0., 1.)
                        rand_contrast = 2.0 * random.random() - 0.2
                        mean = torch.mean(org) * torch.ones_like(org)
                        contrast = (org - mean) * rand_contrast + mean
                        contrast = torch.clamp(contrast, 0., 1.)

                        tensor_list = [org, bright, contrast]
                        img_list = []

                        for tensor in tensor_list:
                            img_list.append(toImage(tensor))

                        new_img_list = []
                        for img in img_list:
                            new_img_list.append(random_crop(img))
                        new_img_list += img_list

                        final_list = []
                        for img in new_img_list:
                            final_list.append(mirror(img))

                        final_list += new_img_list

                        for i, img in enumerate(final_list):
                            imgs.append(img)
                            labels.append(label)

            self.imgs_for_class.append(imgs)
            self.labels_for_class.append(labels)

        self.data = []
        self.targets = []
        # print(len(self.labels_for_class))
        for i in range(len(self.labels_for_class)):
            self.data.extend(self.imgs_for_class[i])
            self.targets.extend(self.labels_for_class[i])

        self.task_idx += 1

    def herding_update(self, net, BF=False):
        def feature(x):
            x = net.module.conv1(x)
            x = net.module.bn1(x)
            x = net.module.relu(x)
            x = net.module.layer1(x)
            x = net.module.layer2(x)
            x = net.module.layer3(x)
            x = net.module.avgpool(x)
            return x.view(x.size(0), -1)

        net.eval()
        task = self.curriculum[self.task_idx]

        # number of samples for each class
        samples_per_class = int(self.capacity / (self.task_idx if BF else self.task_idx + 1) / self.step)

        if self.task_idx > 0:
            for i in range(len(self.imgs_for_class)):
                self.imgs_for_class[i] = self.imgs_for_class[i][:samples_per_class]
                self.labels_for_class[i] = self.labels_for_class[i][:samples_per_class]

        # herding selection
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5070757, 0.48654833, 0.4409185), (0.26733443, 0.25643864, 0.2761505)),
        ])

        for cls in task:
            file_path = self.root + '/' + str(cls)
            files_list = os.listdir(file_path)
            #################################
            # random.shuffle(files)
            #################################
            tmp_dataset = []

            for file in files_list:
                with open(file_path + '/' + file, 'rb') as f:
                    entry = pickle.load(f)
                    tmp_dataset.append([test_transform(entry[0]), entry[1], file])

            tmp_loader = torch.utils.data.DataLoader(tmp_dataset, batch_size=100, shuffle=False, num_workers=2)
            tmp_feature = None
            for data in tmp_loader:
                tmp_img, tmp_label, tmp_file = data
                tmp_img, tmp_label, tmp_file = tmp_img.cuda(), tmp_label.cuda(), tmp_file

                if tmp_feature is None:
                    tmp_feature = feature(tmp_img)
                    tmp_file_ = tmp_file
                else:
                    tmp_feature = torch.cat([tmp_feature, feature(tmp_img)])
                    tmp_file_ += tmp_file

            files = []
            for ixx in range(len(tmp_feature)):
                tmp_feature[ixx] = tmp_feature[ixx] / torch.norm(tmp_feature[ixx], dim=0)

            # check for normalization code. if all output is 1 and length is matched, it correct.
            # print(torch.norm(tmp_feature, dim=1), torch.norm(tmp_feature, dim=1).size())

            tmp_mu = torch.mean(tmp_feature, dim=0)
            tmp_w_t = tmp_mu.clone()
            iter_num = 0
            while len(files) < samples_per_class and iter_num < 1000:
                tmp_score = tmp_w_t * tmp_feature
                t = torch.argmax(tmp_score.sum(dim=1), dim=0)
                if tmp_file_[t] in files:
                    tmp_w_t = tmp_w_t + tmp_mu - tmp_feature[t]
                else:
                    files.append(tmp_file_[t])
                    tmp_w_t = tmp_w_t + tmp_mu - tmp_feature[t]
                iter_num += 1

            # print("herding!")
            # print(len(files))
            files += list(set(tmp_file_) - set(files))[:samples_per_class - len(files)]
            # print(len(files))
            if self.DA:
                files = files[:int(samples_per_class / 12)]
            else:
                files = files[:samples_per_class]
            # For temporal use
            imgs = []
            labels = []

            for file in files:
                with open(file_path + '/' + file, 'rb') as f:
                    entry = pickle.load(f)
                    imgs.append(entry[0])
                    labels.append(entry[1])
            self.imgs_for_class.append(imgs)
            self.labels_for_class.append(labels)

        self.data = []
        self.targets = []
        # print(len(self.labels_for_class))
        for i in range(len(self.labels_for_class)):
            self.data.extend(self.imgs_for_class[i])
            self.targets.extend(self.labels_for_class[i])

        self.task_idx += 1

        net.train()


if __name__ == "__main__":
    import sys

    sys.path.append(os.getcwd())
    from config import config

    for dataset in _datasets:
        preprocess(config['data_path'], dataset)
