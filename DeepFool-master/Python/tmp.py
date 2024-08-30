import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision import datasets
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision
from utils.readData import read_dataset
 
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

'''下载训练集 CIFAR-10 10分类训练集'''
# train_dataset = datasets.CIFAR10('dataset', train=True, transform=transforms.ToTensor(), download=True)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
train_loader,valid_loader,test_loader = read_dataset(batch_size=100,pic_path='dataset')
 
classes=('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
 
def imshow(img):
    img = img/2+0.5
    npimg = img.numpy()
    '''
    img 格式： channels,imageSize,imageSize
    imshow需要格式：imageSize,imageSize,channels
    np.transpose 转换数组
    '''
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()

def de_normalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1)
    std = torch.tensor(std).reshape(1, 3, 1, 1)
    tensor = tensor * std + mean
    return tensor

# 反标准化并转换为图像
def tensor_to_pil(tensor):
    tensor = de_normalize(tensor, mean, std)
    tensor = torch.clamp(tensor, 0, 1).numpy()  # 限制值在 [0, 1] 之间
    return tensor # 去掉批量维度

dataIter = iter(test_loader)

images,labels=dataIter.__next__()
print(f'label:{labels}')
#拼接图像：make_grid
# imshow(torchvision.utils.make_grid(images))
# print(' '.join('%5s' % classes[labels[j]] for j in range(64)))
 
# 单独查看某一张图片
pic = tensor_to_pil(images[1,:,:,:])
pic=pic[0]

# 这时pic的shape是3,224,224，plt.imshow的时候应该是224,224,3的shape
# 下面转换一下通道
pic = np.transpose(pic,(1,2,0))
print(pic.shape)
print(type(pic))
#保存图片
plt.imsave('pic.jpg',pic)

plt.imshow(pic)
plt.show()