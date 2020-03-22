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

for (w5, w6) in zip(model.layer5.parameters(), model.layer6.parameters()):
    if W5 is None and W6 is None:
        W5 = w5.view(-1)
        W6 = w6.view(-1)
    else:
        W5 = torch.cat((W5, w5.view(-1)), 0)
        W6 = torch.cat((W6, w6.view(-1)), 0)

print("layer5 weights = {0}\n".format(W5))

print("layer6 weights = {0}\n".format(W6))






print("Model's state_dict:")
for param_tensor in model.state_dict():
    if('layer5' in param_tensor.split('.')):
        print(param_tensor, "\t", model.state_dict()[param_tensor])

