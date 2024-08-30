import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.utils.data as data_utils
from torch.autograd import Variable
import math
import torchvision.models as models
from PIL import Image
from deepfool import deepfool
import os
from utils.readData import read_dataset
from utils.ResNet import ResNet18

# net = models.resnet18()

# Switch to evaluation mode
# net.eval()
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

def de_normalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1)
    std = torch.tensor(std).reshape(1, 3, 1, 1)
    tensor = tensor * std + mean
    return tensor

# 反标准化并转换为图像
def tensor_to_pil(tensor):
    tensor = de_normalize(tensor, mean, std)
    tensor = torch.clamp(tensor, 0, 1).numpy()  # 限制值在 [0, 1] 之间
    return tensor.squeeze(0) # 去掉批量维度

# set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_class = 10
batch_size = 100
model = ResNet18() # 得到预训练模型
model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
model.fc = torch.nn.Linear(512, n_class) # 将最后的全连接层修改
# 载入权重
model.load_state_dict(torch.load('checkpoint/resnet18_cifar10.pt'))
model = model.to(device)
model.eval()

im_orig = Image.open('pic.jpg')

# Remove the mean
im = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    ])(im_orig)

r, loop_i, label_orig, label_pert, pert_image = deepfool(im, model)

labels = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

str_label_orig = labels[np.int(label_orig)].split(',')[0]
str_label_pert = labels[np.int(label_pert)].split(',')[0]

print("Original label = ", str_label_orig)
print("Perturbed label = ", str_label_pert)

def clip_tensor(A, minv, maxv):
    A = torch.max(A, minv*torch.ones(A.shape))
    A = torch.min(A, maxv*torch.ones(A.shape))
    return A

pic = tensor_to_pil(pert_image.cpu()[0])
pic = np.transpose(pic,(1,2,0))
plt.imsave('pic_perturbed.jpg',pic)

#同时展示pic和im_orig
plt.figure()
plt.subplot(1,2,1)
plt.imshow(im_orig)
plt.title(f'Original label: {str_label_orig}')
plt.axis('off')
plt.subplot(1,2,2)
plt.imshow(pic)
plt.title(f'Perturbed label: {str_label_pert}')
plt.axis('off')
plt.show()

# plt.show()
# plt.imshow(pert_image.cpu()[0])