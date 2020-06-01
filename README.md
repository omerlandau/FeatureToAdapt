# Boosting Adaptation on Pixel Level Segmantation using Larger Norm Feature Extraction
This is a [pytorch](http://pytorch.org/) implementation of [Feature To Adapt](http://openaccess.thecvf.com/content_CVPR_2019/papers/Luo_Taking_a_Closer_Look_at_Domain_Shift_Category-Level_Adversaries_for_CVPR_2019_paper.pdf).

### Prerequisites
- Python >=3.6
- GPU Memory >= 11G
- Pytorch
- Cuda 
- PIL

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
- If adv is set to be True, an adverserial model (based on CLAN) with additional L2 norm feature loss would be trained. If adv is set to be False, The version of deeplab_V2 + only L2 norm loss would be trained. 

### Evaluate
```
python evaluate.py --gpu=2 --ensemble True --restore-from-second "$second_model_path" --restore-from "$first_model_path" --multi False --flip True --save "$path_for_generated_predictions"
```
- Restore from second, multi and ensemble flags are for ensemble use only. If ensemble flag is set to False, only first (--restore-from) model would be evaluated and multi \ restore-from-second flags would be ignored. 

- Multi flag reffers to models which uses multi backbone layers during training s.e MaxSquareLoss and AdaptSegNet models. 

- If using multi model for ensembling, you should be using it as second one (--restore-from-second). 

- One can adjust the weight of each output in the ensemble via direct changes in code.

- Flip means using the image and it's flipped version for prediction (may improve results). 

pretrained models are available via [Google Drive]( https://drive.google.com/drive/folders/1LdTSOw80Nd5fHsMiosP187QDo5LR6Rnf?usp=sharing )

### Compute IoU
```
python iou.py ./data/Cityscapes/gtFine/val "$path_to_predictions"
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

