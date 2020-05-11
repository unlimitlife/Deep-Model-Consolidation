import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from tqdm import tqdm

transforms = transforms.Compose([
                transforms.ToTensor(),
            ])
external_taskset = ImageFolder('train', transform=transforms)
train_loader = torch.utils.data.DataLoader(external_taskset, batch_size=1024,
                                            shuffle=True, num_workers=8)
mean = [0, 0, 0]
std = [0, 0, 0]
for i, data in tqdm(enumerate(train_loader)):
    img = data[0].to('cuda')
    cur_batch = img.shape[0]
    r = torch.index_select(img, 1, torch.tensor([0]).to('cuda'))
    g = torch.index_select(img, 1, torch.tensor([1]).to('cuda'))
    b = torch.index_select(img, 1, torch.tensor([2]).to('cuda'))
    mean[0] += r.mean()*cur_batch
    mean[1] += g.mean()*cur_batch
    mean[2] += b.mean()*cur_batch
    std[0] += (r**2).mean()*cur_batch
    std[1] += (g**2).mean()*cur_batch
    std[2] += (b**2).mean()*cur_batch

mean[0] /= len(external_taskset)
mean[1] /= len(external_taskset)
mean[2] /= len(external_taskset)
std[0] /= len(external_taskset)
std[1] /= len(external_taskset)
std[2] /= len(external_taskset)

std[0] = torch.sqrt(std[0] - mean[0]**2)
std[1] = torch.sqrt(std[1] - mean[1]**2)
std[2] = torch.sqrt(std[2] - mean[2]**2)

print('mean', mean)
print('std', std)