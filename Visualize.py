import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as osp
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from model.deeplabv2_G import Res_Deeplab
from torch.autograd import Variable
import pickle as pkl


def split_all_imgaes(images_p, labels_p, type, direct_l, direct_i, test_adaptation, model_path, gpu0, cropsize):
    if(test_adaptation):
        IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
        model = Res_Deeplab(num_classes=19)
        saved_state_dict = torch.load(model_path, map_location="cuda:{0}".format(gpu0))
        model.load_state_dict(saved_state_dict)
        model.eval()
        model.cuda(gpu0)


    r_images = []
    c=0
    print(len(images_p))
    splitted_imagesdict = []
    total_id = []
    for image_p, label_p in zip(images_p, labels_p):

        image = Image.open(osp.join(direct_i, image_p))
        label = Image.open(osp.join(direct_l, label_p))
        image = np.asarray(image, np.float32)
        shape_x = image.shape[0]
        shape_y = image.shape[1]
        image2 = image.reshape((image.shape[0] * image.shape[1], 3))
        image1 = image2.T
        label = np.asarray(label, np.float32)
        if (type == 'GTA'):
            label = np.asarray(label, np.float32)
            label3 = label.flatten()
            ids = np.unique(label3)
            ids = np.sort(ids)
        else:
            label1 = label.reshape((label.shape[0] * label.shape[1], 4))
            label2 = label1.T
            label3 = 3 * label2[0] + 5 * label2[1] + 7 * label2[2] + 11 * label2[3]
            ids = np.unique(label3)
            ids = np.sort(ids)
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
            imaget = imaget.reshape((shape_x, shape_y, 3))
            if(test_adaptation):
                torch.no_grad()
                imaget = np.asarray(imaget, np.uint8)
                imaget = Image.fromarray(imaget)
                imaget = imaget.resize(cropsize, Image.BICUBIC)
                imaget = np.asarray(imaget, np.float32)
                imaget = imaget[:, :, ::-1]  # change to BGR
                #imaget -= IMG_MEAN
                imaget = imaget.transpose((2, 0, 1))
                print(imaget)
                _, _, imaget = model(torch.unsqueeze(torch.from_numpy(imaget.copy()),dim=0).cuda(gpu0))
                imaget = imaget.data.cpu().numpy()
            else:
                imaget = imaget.reshape((shape_x, shape_y * 3))
            print(imaget.shape)
            imaget = np.array(torch.squeeze(torch.Tensor(imaget),0))
            imaget = imaget.reshape((imaget.shape[0],imaget.shape[1]*imaget.shape[2]))
            print(imaget.shape)
            ipca = PCA(n_components=64, svd_solver='randomized').fit(imaget)
            imaget = ipca.transform(imaget)
            imaget = imaget.flatten()
            splitted_imagesdict.append(imaget)
        total_id.append(ids)
        c +=1
        if(c%10==0):
            print('done with:{0} images'.format(c))

    pca = PCA(64)
    pca.fit(splitted_imagesdict)
    X = pca.transform(splitted_imagesdict)
    print("done PCA")
    tsne = TSNE(n_components=2, learning_rate=130, perplexity=30, angle=0.2, verbose=2, n_iter=6000, early_exaggeration=10).fit_transform(X)
    return tsne, total_id


def main():

    gta_ids = ['00201.png',
               '00202.png',
               '00203.png',
               '00204.png',
               '00205.png',
               '00206.png',
               '00207.png',
               '00208.png',
               '00209.png',
               '00210.png',
               '00211.png',
               '00212.png',
               '00213.png',
               '00214.png',
               '00215.png',
               '00216.png',
               '00217.png',
               '00218.png',
               '00219.png',
               '00220.png',
               '00221.png',
               '00222.png',
               '00223.png',
               '00224.png',
               '00225.png',
               '00226.png',
               '00227.png',
               '00228.png',
               '00229.png',
               '00230.png',
               '00231.png',
               '00232.png',
               '00233.png',
               '00234.png',
               '00235.png',
               '00236.png',
               '00237.png',
               '00238.png',
               '00239.png',
               '00240.png',
               '00241.png',
               '00242.png',
               '00243.png',
               '00244.png',
               '00245.png',
               '00246.png',
               '00247.png',
               '00248.png',
               '00249.png',
               '00250.png',
               '00251.png',
               '00252.png',
               '00253.png',
               '00254.png',
               '00255.png',
               '00256.png',
               '00257.png',
               '00258.png',
               '00259.png',
               '00260.png',
               '00261.png',
               '00262.png',
               '00263.png',
               '00264.png',
               '00265.png',
               '00266.png',
               '00267.png',
               '00268.png',
               '00269.png',
               '00270.png',
               '00271.png',
               '00272.png',
               '00273.png',
               '00274.png',
               '00275.png',
               '00276.png',
               '00277.png',
               '00278.png',
               '00279.png',
               '00280.png',
               '00281.png',
               '00282.png',
               '00283.png',
               '00284.png',
               '00285.png',
               '00286.png',
               '00287.png',
               '00288.png',
               '00289.png',
               '00290.png',
               '00291.png',
               '00292.png',
               '00293.png',
               '00294.png',
               '00295.png',
               '00296.png',
               '00297.png',
               '00298.png',
               '00299.png',
               '00300.png',
               '00301.png',
               '00302.png',
               '00303.png',
               '00304.png',
               '00305.png',
               '00306.png',
               '00307.png',
               '00308.png',
               '00309.png',
               '00310.png',
               '00311.png',
               '00312.png',
               '00313.png',
               '00314.png',
               '00315.png',
               '00316.png',
               '00391.png',
               '00392.png',
               '00393.png',
               '00394.png',
               '00395.png',
               '00396.png',
               '00397.png',
               '00699.png',
               '00700.png']

    city_ids_i = ['frankfurt/frankfurt_000000_003357_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_020880_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_062396_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_046272_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_062509_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_054415_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_021406_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_030310_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_014480_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_005410_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_022797_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_035144_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_014565_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_065850_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_000576_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_065617_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_005543_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_055709_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_027325_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_011835_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_046779_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_064305_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_012738_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_048355_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_019969_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_080091_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_011007_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_015676_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_044227_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_055387_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_038245_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_059642_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_030669_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_068772_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_079206_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_055306_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_012699_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_042384_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_054077_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_010830_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_052120_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_032018_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_051737_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_028335_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_049770_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_054884_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_019698_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_011461_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_001016_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_062250_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_004736_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_068682_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_006589_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_011810_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_066574_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_048654_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_049209_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_042098_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_031416_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_009969_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_038645_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_020046_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_054219_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_002759_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_066438_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_020321_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_002646_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_046126_leftImg8bit.png',
                  'frankfurt/frankfurt_000000_002196_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_057954_leftImg8bit.png',
                  'frankfurt/frankfurt_000001_011715_leftImg8bit.png',
                  'lindau/lindau_000001_000019_leftImg8bit.png',
                  'lindau/lindau_000036_000019_leftImg8bit.png',
                  'lindau/lindau_000035_000019_leftImg8bit.png',
                  'lindau/lindau_000003_000019_leftImg8bit.png',
                  'lindau/lindau_000034_000019_leftImg8bit.png',
                  'lindau/lindau_000010_000019_leftImg8bit.png',
                  'lindau/lindau_000055_000019_leftImg8bit.png',
                  'lindau/lindau_000006_000019_leftImg8bit.png',
                  'lindau/lindau_000019_000019_leftImg8bit.png',
                  'lindau/lindau_000029_000019_leftImg8bit.png',
                  'lindau/lindau_000039_000019_leftImg8bit.png',
                  'lindau/lindau_000051_000019_leftImg8bit.png',
                  'lindau/lindau_000020_000019_leftImg8bit.png',
                  'lindau/lindau_000057_000019_leftImg8bit.png',
                  'lindau/lindau_000041_000019_leftImg8bit.png',
                  'lindau/lindau_000040_000019_leftImg8bit.png',
                  'lindau/lindau_000044_000019_leftImg8bit.png',
                  'lindau/lindau_000028_000019_leftImg8bit.png',
                  'lindau/lindau_000058_000019_leftImg8bit.png',
                  'lindau/lindau_000008_000019_leftImg8bit.png',
                  'munster/munster_000005_000019_leftImg8bit.png',
                  'munster/munster_000102_000019_leftImg8bit.png',
                  'munster/munster_000160_000019_leftImg8bit.png',
                  'munster/munster_000107_000019_leftImg8bit.png',
                  'munster/munster_000095_000019_leftImg8bit.png',
                  'munster/munster_000106_000019_leftImg8bit.png',
                  'munster/munster_000034_000019_leftImg8bit.png',
                  'munster/munster_000143_000019_leftImg8bit.png',
                  'munster/munster_000017_000019_leftImg8bit.png',
                  'munster/munster_000040_000019_leftImg8bit.png',
                  'munster/munster_000152_000019_leftImg8bit.png',
                  'munster/munster_000154_000019_leftImg8bit.png',
                  'munster/munster_000100_000019_leftImg8bit.png',
                  'munster/munster_000004_000019_leftImg8bit.png',
                  'munster/munster_000141_000019_leftImg8bit.png',
                  'munster/munster_000011_000019_leftImg8bit.png',
                  'munster/munster_000055_000019_leftImg8bit.png',
                  'munster/munster_000134_000019_leftImg8bit.png',
                  'munster/munster_000054_000019_leftImg8bit.png',
                  'munster/munster_000064_000019_leftImg8bit.png',
                  'munster/munster_000039_000019_leftImg8bit.png',
                  'munster/munster_000103_000019_leftImg8bit.png',
                  'munster/munster_000092_000019_leftImg8bit.png',
                  'munster/munster_000172_000019_leftImg8bit.png',
                  'munster/munster_000042_000019_leftImg8bit.png',
                  'munster/munster_000124_000019_leftImg8bit.png',
                  'munster/munster_000069_000019_leftImg8bit.png',
                  'munster/munster_000026_000019_leftImg8bit.png',
                  'munster/munster_000120_000019_leftImg8bit.png',
                  'munster/munster_000031_000019_leftImg8bit.png',
                  'munster/munster_000162_000019_leftImg8bit.png',
                  'munster/munster_000056_000019_leftImg8bit.png',
                  'munster/munster_000081_000019_leftImg8bit.png',
                  'munster/munster_000123_000019_leftImg8bit.png',
                  'munster/munster_000125_000019_leftImg8bit.png',
                  'munster/munster_000082_000019_leftImg8bit.png',
                  'munster/munster_000133_000019_leftImg8bit.png',
                  'munster/munster_000126_000019_leftImg8bit.png',
                  'munster/munster_000063_000019_leftImg8bit.png',
                  'munster/munster_000008_000019_leftImg8bit.png',
                  'munster/munster_000149_000019_leftImg8bit.png',
                  'munster/munster_000076_000019_leftImg8bit.png',
                  'munster/munster_000091_000019_leftImg8bit.png']

    city_ids_l = ['frankfurt/frankfurt_000000_003357_gtFine_color.png',
                  'frankfurt/frankfurt_000000_020880_gtFine_color.png',
                  'frankfurt/frankfurt_000001_062396_gtFine_color.png',
                  'frankfurt/frankfurt_000001_046272_gtFine_color.png',
                  'frankfurt/frankfurt_000001_062509_gtFine_color.png',
                  'frankfurt/frankfurt_000001_054415_gtFine_color.png',
                  'frankfurt/frankfurt_000001_021406_gtFine_color.png',
                  'frankfurt/frankfurt_000001_030310_gtFine_color.png',
                  'frankfurt/frankfurt_000000_014480_gtFine_color.png',
                  'frankfurt/frankfurt_000001_005410_gtFine_color.png',
                  'frankfurt/frankfurt_000000_022797_gtFine_color.png',
                  'frankfurt/frankfurt_000001_035144_gtFine_color.png',
                  'frankfurt/frankfurt_000001_014565_gtFine_color.png',
                  'frankfurt/frankfurt_000001_065850_gtFine_color.png',
                  'frankfurt/frankfurt_000000_000576_gtFine_color.png',
                  'frankfurt/frankfurt_000001_065617_gtFine_color.png',
                  'frankfurt/frankfurt_000000_005543_gtFine_color.png',
                  'frankfurt/frankfurt_000001_055709_gtFine_color.png',
                  'frankfurt/frankfurt_000001_027325_gtFine_color.png',
                  'frankfurt/frankfurt_000001_011835_gtFine_color.png',
                  'frankfurt/frankfurt_000001_046779_gtFine_color.png',
                  'frankfurt/frankfurt_000001_064305_gtFine_color.png',
                  'frankfurt/frankfurt_000001_012738_gtFine_color.png',
                  'frankfurt/frankfurt_000001_048355_gtFine_color.png',
                  'frankfurt/frankfurt_000001_019969_gtFine_color.png',
                  'frankfurt/frankfurt_000001_080091_gtFine_color.png',
                  'frankfurt/frankfurt_000000_011007_gtFine_color.png',
                  'frankfurt/frankfurt_000000_015676_gtFine_color.png',
                  'frankfurt/frankfurt_000001_044227_gtFine_color.png',
                  'frankfurt/frankfurt_000001_055387_gtFine_color.png',
                  'frankfurt/frankfurt_000001_038245_gtFine_color.png',
                  'frankfurt/frankfurt_000001_059642_gtFine_color.png',
                  'frankfurt/frankfurt_000001_030669_gtFine_color.png',
                  'frankfurt/frankfurt_000001_068772_gtFine_color.png',
                  'frankfurt/frankfurt_000001_079206_gtFine_color.png',
                  'frankfurt/frankfurt_000001_055306_gtFine_color.png',
                  'frankfurt/frankfurt_000001_012699_gtFine_color.png',
                  'frankfurt/frankfurt_000001_042384_gtFine_color.png',
                  'frankfurt/frankfurt_000001_054077_gtFine_color.png',
                  'frankfurt/frankfurt_000001_010830_gtFine_color.png',
                  'frankfurt/frankfurt_000001_052120_gtFine_color.png',
                  'frankfurt/frankfurt_000001_032018_gtFine_color.png',
                  'frankfurt/frankfurt_000001_051737_gtFine_color.png',
                  'frankfurt/frankfurt_000001_028335_gtFine_color.png',
                  'frankfurt/frankfurt_000001_049770_gtFine_color.png',
                  'frankfurt/frankfurt_000001_054884_gtFine_color.png',
                  'frankfurt/frankfurt_000001_019698_gtFine_color.png',
                  'frankfurt/frankfurt_000000_011461_gtFine_color.png',
                  'frankfurt/frankfurt_000000_001016_gtFine_color.png',
                  'frankfurt/frankfurt_000001_062250_gtFine_color.png',
                  'frankfurt/frankfurt_000001_004736_gtFine_color.png',
                  'frankfurt/frankfurt_000001_068682_gtFine_color.png',
                  'frankfurt/frankfurt_000000_006589_gtFine_color.png',
                  'frankfurt/frankfurt_000000_011810_gtFine_color.png',
                  'frankfurt/frankfurt_000001_066574_gtFine_color.png',
                  'frankfurt/frankfurt_000001_048654_gtFine_color.png',
                  'frankfurt/frankfurt_000001_049209_gtFine_color.png',
                  'frankfurt/frankfurt_000001_042098_gtFine_color.png',
                  'frankfurt/frankfurt_000001_031416_gtFine_color.png',
                  'frankfurt/frankfurt_000000_009969_gtFine_color.png',
                  'frankfurt/frankfurt_000001_038645_gtFine_color.png',
                  'frankfurt/frankfurt_000001_020046_gtFine_color.png',
                  'frankfurt/frankfurt_000001_054219_gtFine_color.png',
                  'frankfurt/frankfurt_000001_002759_gtFine_color.png',
                  'frankfurt/frankfurt_000001_066438_gtFine_color.png',
                  'frankfurt/frankfurt_000000_020321_gtFine_color.png',
                  'frankfurt/frankfurt_000001_002646_gtFine_color.png',
                  'frankfurt/frankfurt_000001_046126_gtFine_color.png',
                  'frankfurt/frankfurt_000000_002196_gtFine_color.png',
                  'frankfurt/frankfurt_000001_057954_gtFine_color.png',
                  'frankfurt/frankfurt_000001_011715_gtFine_color.png',
                  'lindau/lindau_000001_000019_gtFine_color.png',
                  'lindau/lindau_000036_000019_gtFine_color.png',
                  'lindau/lindau_000035_000019_gtFine_color.png',
                  'lindau/lindau_000003_000019_gtFine_color.png',
                  'lindau/lindau_000034_000019_gtFine_color.png',
                  'lindau/lindau_000010_000019_gtFine_color.png',
                  'lindau/lindau_000055_000019_gtFine_color.png',
                  'lindau/lindau_000006_000019_gtFine_color.png',
                  'lindau/lindau_000019_000019_gtFine_color.png',
                  'lindau/lindau_000029_000019_gtFine_color.png',
                  'lindau/lindau_000039_000019_gtFine_color.png',
                  'lindau/lindau_000051_000019_gtFine_color.png',
                  'lindau/lindau_000020_000019_gtFine_color.png',
                  'lindau/lindau_000057_000019_gtFine_color.png',
                  'lindau/lindau_000041_000019_gtFine_color.png',
                  'lindau/lindau_000040_000019_gtFine_color.png',
                  'lindau/lindau_000044_000019_gtFine_color.png',
                  'lindau/lindau_000028_000019_gtFine_color.png',
                  'lindau/lindau_000058_000019_gtFine_color.png',
                  'lindau/lindau_000008_000019_gtFine_color.png',
                  'munster/munster_000005_000019_gtFine_color.png',
                  'munster/munster_000102_000019_gtFine_color.png',
                  'munster/munster_000160_000019_gtFine_color.png',
                  'munster/munster_000107_000019_gtFine_color.png',
                  'munster/munster_000095_000019_gtFine_color.png',
                  'munster/munster_000106_000019_gtFine_color.png',
                  'munster/munster_000034_000019_gtFine_color.png',
                  'munster/munster_000143_000019_gtFine_color.png',
                  'munster/munster_000017_000019_gtFine_color.png',
                  'munster/munster_000040_000019_gtFine_color.png',
                  'munster/munster_000152_000019_gtFine_color.png',
                  'munster/munster_000154_000019_gtFine_color.png',
                  'munster/munster_000100_000019_gtFine_color.png',
                  'munster/munster_000004_000019_gtFine_color.png',
                  'munster/munster_000141_000019_gtFine_color.png',
                  'munster/munster_000011_000019_gtFine_color.png',
                  'munster/munster_000055_000019_gtFine_color.png',
                  'munster/munster_000134_000019_gtFine_color.png',
                  'munster/munster_000054_000019_gtFine_color.png',
                  'munster/munster_000064_000019_gtFine_color.png',
                  'munster/munster_000039_000019_gtFine_color.png',
                  'munster/munster_000103_000019_gtFine_color.png',
                  'munster/munster_000092_000019_gtFine_color.png',
                  'munster/munster_000172_000019_gtFine_color.png',
                  'munster/munster_000042_000019_gtFine_color.png',
                  'munster/munster_000124_000019_gtFine_color.png',
                  'munster/munster_000069_000019_gtFine_color.png',
                  'munster/munster_000026_000019_gtFine_color.png',
                  'munster/munster_000120_000019_gtFine_color.png',
                  'munster/munster_000031_000019_gtFine_color.png',
                  'munster/munster_000162_000019_gtFine_color.png',
                  'munster/munster_000056_000019_gtFine_color.png',
                  'munster/munster_000081_000019_gtFine_color.png',
                  'munster/munster_000123_000019_gtFine_color.png',
                  'munster/munster_000125_000019_gtFine_color.png',
                  'munster/munster_000082_000019_gtFine_color.png',
                  'munster/munster_000133_000019_gtFine_color.png',
                  'munster/munster_000126_000019_gtFine_color.png',
                  'munster/munster_000063_000019_gtFine_color.png',
                  'munster/munster_000008_000019_gtFine_color.png',
                  'munster/munster_000149_000019_gtFine_color.png',
                  'munster/munster_000076_000019_gtFine_color.png',
                  'munster/munster_000091_000019_gtFine_color.png']

    gta_images,gta_cmap = split_all_imgaes(gta_ids,gta_ids,'GTA',direct_i='./data/GTA5/images', direct_l='./data/GTA5/labels', test_adaptation=True, model_path='./snapshots/GTA2Cityscapes_norm_00015_Damping15_normal_weight_loss_restore_from_40000_G_38_D_numsteps_fixed/GTA5_40000.pth', gpu0=3, cropsize=(1280,720))


    with open("./Adapted_GTA_p20_exagg", 'wb') as pfile:
        pkl.dump(gta_images, pfile, protocol=3)

    with open("./GTA_cmap", 'wb') as pfile:
        pkl.dump(gta_cmap, pfile, protocol=3)

    print('Dumped GTA pickle')

    city_images, city_cmap = split_all_imgaes(city_ids_i,city_ids_l,type='city', direct_l='./data/CitySpaces/gtFine/val', direct_i='./data/CitySpaces/leftImg8bit/val', test_adaptation=True, model_path='./snapshots/GTA2Cityscapes_norm_00015_Damping15_normal_weight_loss_restore_from_40000_G_38_D_numsteps_fixed/GTA5_40000.pth', gpu0=1, cropsize=(1024,512))


    with open("./Adapted_City_p20_exagg", 'wb') as pfile:
        pkl.dump(city_images, pfile, protocol=3)

    with open("./City_cmap", 'wb') as pfile:
        pkl.dump(city_cmap, pfile, protocol=3)

    print('Dumped City pickle')


if __name__ == '__main__':
    print('start')
    main()
    print('end')