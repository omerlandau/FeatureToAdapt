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

```
pretrained models are available via [Google Drive]( https://drive.google.com/open?id=1Hl7r6fIbNfyA9A8wGUJIMOwzXVQ61ik8 )

### Compute IoU
```
python iou.py ./data/Cityscapes/gtFine/val result/GTA2Cityscapes_100000
```

#### Tip: The best-performance model might not be the final one in the last epoch. If you want to evaluate every saved models in bulk, please use CLAN_evaluate_bulk.py and CLAN_iou_bulk.py, the result will be saved in an Excel sheet.
```
python evaluate_bulk.py
python iou_bulk.py
```

### Visualization Results
<p align="left">
	<img src="https://github.com/RoyalVane/CLAN/blob/master/gifs/video_1.gif"  width="420" height="210" alt="(a)"/>

  <img src="https://github.com/RoyalVane/CLAN/blob/master/gifs/video_2.gif"  width="420" height="210" alt="(b)"/>
</p>


#### This code is heavily borrowed from the baseline [AdaptSegNet]( https://github.com/wasidennis/AdaptSegNet ) and [CLAN]

