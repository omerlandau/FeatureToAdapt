import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

image = Image.open('/Users/omerlandau/munster_000150_000019_leftImg8bit.png')
label = Image.open('/Users/omerlandau/munster_000150_000019_gtFine_color.png')

def split_all_imgaes(images_p,labels_p):

    r_images = []

    for image_p, label_p in zip(images_p, labels_p):

        image = Image.open(image_p)
        label = Image.open(labels_p)
        image = np.asarray(image, np.float32)
        image2 = image.reshape((image.shape[0]*image.shape[1], 3))
        image1 = image2.T
        label = np.asarray(label, np.float32)
        label1 = label.reshape((label.shape[0] * label.shape[1], 4))
        label2 = label1.T
        label3 = 3 * label2[0] + 5 * label2[1] + 7 * label2[2] + 11 * label2[3]
        ids = np.unique(label3)
        ids = np.sort(ids)
        splitted_imagesdict = []
        for i in ids:
            imaget = np.copy(image1)
            mapfig = np.isin(label3, i)
            mapfigx = np.copy(mapfig)
            mapfigy = np.copy(mapfig)
            mapfigz = np.copy(mapfig)
            np.copyto(imaget[0], 0, where=np.invert(mapfigx))
            np.copyto(imaget[1], 0, where=np.invert(mapfigy))
            np.copyto(imaget[2], 0, where=np.invert(mapfigz))
            imaget = imaget.T
            imaget = imaget.flatten()
            splitted_imagesdict.append(imaget)
        r_images.append(splitted_imagesdict)





