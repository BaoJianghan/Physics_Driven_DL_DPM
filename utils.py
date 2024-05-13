import os
import torch
import math
import random
from torch.autograd import Variable
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from PIL import Image
import copy
import random
import numpy as np
import scipy.io as scio
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

def to_tensor(array, cuda = True):
    if cuda:
        return torch.tensor(array).cuda()
    else:
        return torch.tensor(array)

def check_make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def normal_imgs(imgs):
    max_img = torch.max(torch.max(imgs, -1)[0], -1)[0]
    max_img = max_img.reshape(imgs.shape[0], 1, 1, 1)
    return imgs / max_img


def get_dual_input_by_name(tgt_path, probe_1, probe_2, args):
    trans = transforms.Compose([
            transforms.Resize([args.size, args.size]),
            transforms.ToTensor()
        ])

    img_0 = Image.open(tgt_path + probe_1 + '.jpg')
    img = trans(img_0)
    img = img[0, :, :].unsqueeze(0).unsqueeze(0).cuda()

    sec_beam_0 = Image.open(tgt_path + probe_2 + '.jpg')
    sec_beam = trans(sec_beam_0)
    sec_beam = sec_beam[0, :, :].unsqueeze(0).unsqueeze(0).cuda()

    probe_name = probe_1 + '_' + probe_2 + '.jpg'
    
    img = img + sec_beam
    img = normal_imgs(img)

    img_ay = np.array(img_0)[:, :, 0] + np.array(sec_beam_0)[:, :, 0]
    return img_ay, img


def get_single_input_by_name(tgt_path, probe_1, args):
    trans = transforms.Compose([
            transforms.Resize([args.size, args.size]),
            transforms.ToTensor()
        ])

    img_0 = Image.open(tgt_path + probe_1 + '.jpg')
    img = trans(img_0)
    img = img[0, :, :].unsqueeze(0).unsqueeze(0).cuda()

    img = normal_imgs(img)
    img_ay = np.array(img_0)[:, :, 0]
    return img_ay, img