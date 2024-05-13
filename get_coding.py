import argparse
import os
import time
import torch

import torch.nn as nn
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from resnet import *
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--size', type = int, default = 64)
parser.add_argument('--test', type = int, default = 5)
parser.add_argument('--M', type = int, default = 18)
parser.add_argument('--N', type = int, default = 18)

args = parser.parse_args()
torch.autograd.set_detect_anomaly(True)

tgt_path = './target_uv/'
sec_path = './target_uv/'

model_dir = './saved_model/'
model_name = 'resnet34.pt'
saved_state_dict = torch.load(model_dir + model_name)

resnet = ResNet34(1, args.M * args.N).cuda() 
resnet.load_state_dict(saved_state_dict)

# change the orientation of target beam here
# -------------------- #
probe_1 = '6_180'
probe_2 = '9_0'
# -------------------- #

radi_ata_rot_k = 3
coding_rot_k = 3

input_img_ay, input_img = get_dual_input_by_name(tgt_path, probe_1, probe_2, args)
coding = resnet(input_img)

coding_disc = coding.clone()
coding_disc = coding_disc.reshape((args.M, args.N))
coding_disc[coding_disc <= 0] = -1
coding_disc[coding_disc > 0] = 1
coding_disc_ay = coding_disc.detach().cpu().numpy()
coding_ay = coding_disc_ay.copy()
coding_ay = np.rot90(coding_ay, k = coding_rot_k)
coding_ay = coding_ay.copy()

color1 = (1, 1, 1)
color2 = (255 / 255, 165 / 255, 0 / 255)
color3 = (255 / 255, 0 / 255, 0 / 255)

coding_cmap = LinearSegmentedColormap.from_list('coding_cmap', [color1, color2], 2)

plt.plot()
plt.axis('off')
plt.pcolormesh(coding_ay, edgecolors = 'k', linewidth = 1, cmap = coding_cmap)
ax = plt.gca()
ax.set_aspect('equal')

plt.show()