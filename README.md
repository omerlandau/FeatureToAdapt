# Boosting Adaptation on Pixel Level Segmantation using Larger Norm Feature Extraction
This is a [pytorch](http://pytorch.org/) implementation of [Feature To Adapt](http://openaccess.thecvf.com/content_CVPR_2019/papers/Luo_Taking_a_Closer_Look_at_Domain_Shift_Category-Level_Adversaries_for_CVPR_2019_paper.pdf).

### Prerequisites
- Python 3.6
- GPU Memory >= 11G

### Getting started

- Download [The GTA5 Dataset]( https://download.visinf.tu-darmstadt.de/data/from_games/ )

- Download [The Cityscapes Dataset]( https://www.cityscapes-dataset.com/ )

- Download [The imagenet pretraind model]( https://drive.google.com/open?id=13kjtX481LdtgJcpqD3oROabZyhGLSBm2 )

The data folder is structured as follows:
```
├── data/
│   ├── Cityscapes/     
|   |   ├── gtFine/
|   |   ├── leftImg8bit/
│   ├── GTA5/
|   |   ├── images/
|   |   ├── labels/			
└── model/
│   ├── DeepLab_resnet_pretrained.pth
...
```

### Train
```
python train.py --gpu 6 --adv True --snapshot-dir ./GTA_snapshots/GTA2Cityscapes_adv 
```

### Evaluate
```
python evaluate.py --gpu=2 --ensemble True --restore-from-second "./model/GTA5_82000_orig_clan.pth" --restore-from ./snapshots/GTA2Cityscapes_only_norm/GTA5_100000.pth --multi False --flip True --save ./GTA_results/GTA2Cityscapes_ORIG_CLAN_and_NORM_flip_44_56
```
- Restore from second, multi and ensemble flags are for ensemble use only. If ensemble flag is set to False, only first (--restore-from) model would be evaluated and multi \ restore-from-second flags would be ignored. 

- Multi flag reffers to models which uses multi backbone layers during training s.e MaxSquareLoss and AdaptSegNet models. 

- If using multi model for ensembling, you should be using it as second one (--restore-from-second). 

- One can adjust the weight of each output in the ensemble via direct changes in code.

- Flip means using the image and it's flipped version for prediction (may improve results). 

pretrained models are available via [Google Drive]( https://drive.google.com/drive/folders/1LdTSOw80Nd5fHsMiosP187QDo5LR6Rnf?usp=sharing )

### Compute IoU
```
python iou.py ./data/Cityscapes/gtFine/val result/GTA2Cityscapes_100000
```

#### In order to Evaluate and compute iou of number of models you may use evaluate_bulk.py and iou_bulk.py, the results will be saved in a csv format. Complex evaluation s.e flipping and ensemble is not supported in evaluate_bulk mode.
```
python evaluate_bulk.py
python iou_bulk.py
```

### Visualization Results
<p align="left">
	<img src="https://github.com/omerlandau/FeatureToAdapt/blob/master/results_visualization.png"  alt="(a)"/>

</p>


#### This code is heavily borrowed from the baseline [AdaptSegNet]( https://github.com/wasidennis/AdaptSegNet ) and [CLAN]

