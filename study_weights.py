import numpy as np
import torch
from torch.autograd import Variable
from model.deeplabv2_G import Res_Deeplab
from dataset.cityscapes_dataset import cityscapesDataSet
from torch.utils import data
import os
from PIL import Image
import torch.nn as nn

def get_L2norm_loss_self_driven(x):

    radius = x.norm(p=2, dim=1).detach()
    assert radius.requires_grad == False
    radius = radius + 0.15
    n = x.norm(p=2, dim=1)
    l = ((n - radius) ** 2).mean()
    return l

model_path = './snapshots/GTA2Cityscapes_norm_00015_Damping15_normal_weight_loss/GTA5_40000.pth'

print(torch.cuda)

model = Res_Deeplab(num_classes=19)

saved_state_dict = torch.load(model_path, map_location="cuda:5")
model.load_state_dict(saved_state_dict)
W5 = None
W6 = None
W7 = None
for w7 in model.layer4.parameters():
    if W7 is None:
        W7 = w7.view(-1)
    else:
        W7 = torch.cat((W7, w7.view(-1)), 0)

print("layer4 l2 norm = {0}".format(W7.norm(p=2, dim=1)))

#print("layer5 weights = {0}\n".format(W5))

#print("layer6 weights = {0}\n".format(W6))






#print("Model's state_dict:")
#for param_tensor in model.state_dict():
#    if('layer5' in param_tensor.split('.')):
#        print(param_tensor, "\t", model.state_dict()[param_tensor])

