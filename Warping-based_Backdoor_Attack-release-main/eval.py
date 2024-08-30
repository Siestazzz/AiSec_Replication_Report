import json
import os
import shutil
from time import time

import config
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from classifier_models import PreActResNet18, ResNet18
from networks.models import Denormalizer, NetC_MNIST, Normalizer
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import RandomErasing
from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
from torch.autograd import Variable
import matplotlib.pyplot as plt
import collections

mean=[0.4914, 0.4822, 0.4465]
std=[0.247, 0.243, 0.261]

def de_normalize(tensor, mean, std):
    mean = torch.tensor(mean).reshape(1, 3, 1, 1)
    std = torch.tensor(std).reshape(1, 3, 1, 1)
    tensor = tensor * std + mean
    return tensor

# 反标准化并转换为图像
def tensor_to_pil(tensor):
    tensor = tensor.cpu()
    tensor = de_normalize(tensor, mean, std)
    tensor = torch.clamp(tensor, 0, 1).numpy()  # 限制值在 [0, 1] 之间
    return np.transpose(tensor[0],(1,2,0)) # 去掉批量维度
    # return tensor.squeeze(0)[0]

def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

    if opt.dataset == "cifar10" or opt.dataset == "gtsrb":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    if opt.dataset == "celeba":
        netC = ResNet18().to(opt.device)
    if opt.dataset == "mnist":
        netC = NetC_MNIST().to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC

def dump_pic(inputs, origin_label, inputs_bd, perturbed_label):
    labels = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    origin_image = tensor_to_pil(inputs)
    perturbed_image = tensor_to_pil(inputs_bd)
    plt.imsave('WaNet_origin_image.jpg',origin_image)
    plt.imsave('WaNet_perturbed_image.jpg',perturbed_image)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(origin_image)
    plt.title(f'Original label: {labels[origin_label]}')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(perturbed_image)
    plt.title(f'Perturbed label: {labels[perturbed_label]}')
    plt.axis('off')
    plt.show()
    print(origin_label)
    print(perturbed_label)
    # exit(0)

def eval(
    netC,
    optimizerC,
    schedulerC,
    test_dl,
    noise_grid,
    identity_grid,
    opt,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    total_ae_loss = 0

    criterion_BCE = torch.nn.BCELoss()

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():

            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # if batch_idx <= 10:
            #     labels = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            #     origin_label = predict(netC,inputs[0][None, :, :, :])
            #     # perturbed_label = predict(netC,inputs_bd[0][None, :, :, :])
            #     origin_image = tensor_to_pil(inputs[1, :, :, :])
            #     # perturbed_image = tensor_to_pil(inputs_bd[1, :, :, :])
            #     plt.imsave('WaNet_origin_image.jpg',origin_image)
            #     # plt.imsave('WaNet_perturbed_image.jpg',perturbed_image)
            #     plt.figure()
            #     plt.subplot(1,2,1)
            #     plt.imshow(origin_image)
            #     plt.title(f'Original label: {labels[origin_label]}')
            #     plt.axis('off')
            #     # plt.subplot(1,2,2)
            #     # plt.imshow(perturbed_image)
            #     # plt.title(f'Perturbed label: {labels[perturbed_label]}')
            #     # plt.axis('off')
            #     plt.show()
            #     print(origin_label)
            #     # print(perturbed_label)
            #     exit(0)

            

            # Evaluate Backdoor
            grid_temps = (identity_grid + opt.s * noise_grid / opt.input_height) * opt.grid_rescale
            grid_temps = torch.clamp(grid_temps, -1, 1)

            ins = torch.rand(bs, opt.input_height, opt.input_height, 2).to(opt.device) * 2 - 1
            grid_temps2 = grid_temps.repeat(bs, 1, 1, 1) + ins / opt.input_height
            grid_temps2 = torch.clamp(grid_temps2, -1, 1)

            inputs_bd = F.grid_sample(inputs, grid_temps.repeat(bs, 1, 1, 1), align_corners=True)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets + 1, opt.num_classes)
            
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            # Evaluate Clean
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            if batch_idx <= 0:
                print(f'target: {targets[0]} tagetbd: {targets_bd[0]}')
                dump_pic(inputs[0, :, :, :],torch.argmax(preds_clean, 1)[0],inputs_bd[0, :, :, :],torch.argmax(preds_bd, 1)[0])
            

            # Evaluate cross
            if opt.cross_ratio:
                inputs_cross = F.grid_sample(inputs, grid_temps2, align_corners=True)
                preds_cross = netC(inputs_cross)
                total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

                acc_cross = total_cross_correct * 100.0 / total_sample

                info_string = "Clean Acc: {:.4f} | Bd Acc: {:.4f} | Cross: {:.4f}".format(acc_clean, acc_bd, acc_cross)
            else:
                info_string = "Clean Acc: {:.4f} - Best: {:.4f} | Bd Acc: {:.4f} - Best: {:.4f}".format(
                    acc_clean, best_clean_acc, acc_bd, best_bd_acc
                )
            progress_bar(batch_idx, len(test_dl), info_string)


def main():
    opt = config.get_arguments().parse_args()

    if opt.dataset in ["mnist", "cifar10"]:
        opt.num_classes = 10
    elif opt.dataset == "gtsrb":
        opt.num_classes = 43
    elif opt.dataset == "celeba":
        opt.num_classes = 8
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "cifar10":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "gtsrb":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    elif opt.dataset == "mnist":
        opt.input_height = 28
        opt.input_width = 28
        opt.input_channel = 1
    elif opt.dataset == "celeba":
        opt.input_height = 64
        opt.input_width = 64
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "{}_{}_morph.pth.tar".format(opt.dataset, mode))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")

    if os.path.exists(opt.ckpt_path):
        state_dict = torch.load(opt.ckpt_path)
        netC.load_state_dict(state_dict["netC"])
        identity_grid = state_dict["identity_grid"]
        noise_grid = state_dict["noise_grid"]
    else:
        print("Pretrained model doesnt exist")
        exit()

    eval(
        netC,
        optimizerC,
        schedulerC,
        test_dl,
        noise_grid,
        identity_grid,
        opt,
    )


if __name__ == "__main__":
    main()