import numpy as np
import torch
from torch.autograd import Variable
from model.CLAN_G import Res_Deeplab
from dataset.cityscapes_dataset import cityscapesDataSet
from torch.utils import data
import os
from PIL import Image
import torch.nn as nn

model_path = './snapshots/GTA2Cityscapes_norm_00015_Damping15_normal_weight_loss/GTA5_40000.pth'

print(torch.cuda)

model = Res_Deeplab(num_classes=19)

saved_state_dict = torch.load(model_path, map_location="cuda:2")
model.load_state_dict(saved_state_dict)

print("Model's state_dict:")
for param_tensor in model.state_dict():
    if('layer5' in param_tensor.split('.')):
        print(param_tensor, "\t", model.state_dict()[param_tensor])

