import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils import data, model_zoo
from model.CLAN_G import Res_Deeplab
from dataset.cityscapes_dataset import cityscapesDataSet
from model import deeplab_multi
from model import ResNet101
import os
from PIL import Image
import torch.nn as nn

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = './data/CitySpaces'
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt'
SAVE_PATH = './result/cityscapes'

MODEL = 'ResNet' #Vgg
IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
SET = 'val'

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : ResNet, Vgg")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--restore-from-second", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--multi", type=str, default=False,
                        help="is restored model multi or not.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()


def main():
    """Create the model and start the evaluation process."""

    args = get_arguments()

    gpu0 = args.gpu

    if(args.multi == 'True'):
        multi = True
    else:
        multi = False

    if not os.path.exists(args.save):
        os.makedirs(args.save)
    
    if args.model == 'ResNet':
        model = Res_Deeplab(num_classes=args.num_classes)
        if(multi):
            model2 = deeplab_multi.DeeplabMulti()
        else:
            model2 = Res_Deeplab(num_classes=args.num_classes)
    
    if args.restore_from[:4] == 'http' :
        saved_state_dict = model_zoo.load_url(args.restore_from)
    else:
        saved_state_dict = torch.load(args.restore_from, map_location="cuda:{0}".format(args.gpu))
        saved_state_dict_2 = torch.load(args.restore_from_second, map_location="cuda:{0}".format(args.gpu))
        saved_state_dict_2 = saved_state_dict_2["state_dict"]
        keys_1 = saved_state_dict.keys()
        keys_2 = saved_state_dict_2.keys()
        keys_2 = list(keys_2)
        keys_1 = list(keys_1)
        print(keys_2)
        print(keys_1)
        exit(1)
        saved_state_dict_2[saved_state_dict.keys()] = saved_state_dict_2[keys_2]
        del saved_state_dict_2[keys_2]

    model.load_state_dict(saved_state_dict)
    model2.load_state_dict(saved_state_dict_2)
    
    model.eval()
    model.cuda(gpu0)
    model2.eval()
    model2.cuda(gpu0)
    testloader = data.DataLoader(cityscapesDataSet(args.data_dir, args.data_list, crop_size=(1024, 512), mean=IMG_MEAN, scale=False, mirror=False, set=args.set),
                                    batch_size=1, shuffle=False, pin_memory=True)

    interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

    with torch.no_grad():
        avg = 0
        c=0
        for index, batch in enumerate(testloader):
            if index % 100 == 0:
                print('%d processd' % index)
            image, _, _, name = batch
            output1, output2 , norm_dims = model(Variable(image).cuda(gpu0))
            output1_2, output2_2, _ = model2(Variable(image).cuda(gpu0))

            c+=1

            n = norm_dims.norm(p=2, dim=1).cuda(gpu0)

            temp = torch.dist(n.flatten(), torch.zeros(n.flatten().shape).cuda(gpu0))

            avg += temp
            print("L2 norm of pic {0} = {1}".format(c, temp))

            output_final = (output2+output1)*0.2 + 0.7*(output1_2 + output2_2)

            output = interp(output_final).cpu().data[0].numpy()
            
            output = output.transpose(1,2,0)
            output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
    
            output_col = colorize_mask(output)
            output = Image.fromarray(output)
    
            name = name[0].split('/')[-1]
            output.save('%s/%s' % (args.save, name))

            output_col.save('%s/%s_color.png' % (args.save, name.split('.')[0]))

        avg = avg/c
        print("average L2 norm = {0}".format(avg))


if __name__ == '__main__':
    main()
