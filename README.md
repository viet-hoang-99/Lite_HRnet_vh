# Lite-HRNet-vh: An Easier Implementation of Lightweight High-Resolution Network

## Introduction
This is an easier pytorch implementation (not using mmcv apis) of [Lite-HRNet: A Lightweight High-Resolution Network](https://arxiv.org/abs/2104.06403). This is easier for developer and reseacher to customize networks, dataset, ...

<img width="512" height="512" src="/resources/litehrnet_block.png"/>

## Results and models

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset

| Arch  | Input Size | #Params | FLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt |
| :----------------- | :-----------: | :------: | :-----------: | :------: |:------: | :------: | :------: | :------: | :------: |
| [Naive Lite-HRNet-18](/configs/top_down/naive_litehrnet/coco/naive_litehrnet_18_coco_256x192.py)  | 256x192 | 0.7M | 194.8M | 0.628 | 0.855 | 0.699 | 0.691 | 0.901 | update soon |
| [Wider Naive Lite-HRNet-18](/configs/top_down/naive_litehrnet/coco/wider_naive_litehrnet_18_coco_256x192.py)  | 256x192 | 1.3M | 311.1M | 0.660 | 0.871 | 0.737 | 0.721 | 0.913 | update soon |
| [Lite-HRNet-18](/configs/top_down/lite_hrnet/coco/litehrnet_18_coco_256x192.py)  | 256x192 | 1.1M | 205.2M |0.648 | 0.867 | 0.730 | 0.712 | 0.911 | update soon |
| [Lite-HRNet-18](/configs/top_down/lite_hrnet/coco/litehrnet_18_coco_384x288.py)  | 384x288 | 1.1M | 461.6M | 0.676 | 0.878 | 0.750 | 0.737 | 0.921 | update soon |
| [Lite-HRNet-30](/configs/top_down/lite_hrnet/coco/litehrnet_30_coco_256x192.py)  | 256x192 | 1.8M | 319.2M | 0.672 | 0.880 | 0.750 | 0.733 | 0.922 | update soon |
| [Lite-HRNet-30](/configs/top_down/lite_hrnet/coco/litehrnet_30_coco_384x288.py)  | 384x288 | 1.8M | 717.8M | 0.704 | 0.887 | 0.777 | 0.762 | 0.928 | update soon |

### Results on MPII val set

| Arch  | Input Size | #Params | FLOPs | Mean | Mean@0.1   | ckpt |
| :--- | :--------: | :------: | :--------: | :------: | :------: | :------: |
| [Naive Lite-HRNet-18](/configs/top_down/naive_litehrnet/mpii/naive_litehrnet_18_mpii_256x256.py) | 256x256 | 0.7M | 259.6M | 0.853 | 0.305 | update soon |
| [Wider Naive Lite-HRNet-18](/configs/top_down/naive_litehrnet/mpii/wider_naive_litehrnet_18_mpii_256x256.py) | 256x256 | 1.3M | 418.7M | 0.868 | 0.311 | update soon |
| [Lite-HRNet-18](/configs/top_down/lite_hrnet/mpii/litehrnet_18_mpii_256x256.py) | 256x256 | 1.1M | 273.4M | 0.854 | 0.295 | update soon |
| [Lite-HRNet-30](/configs/top_down/lite_hrnet/mpii/litehrnet_30_mpii_256x256.py) | 256x256 | 1.8M | 425.3M | 0.870 | 0.313 | update soon |


## Enviroment
The code is developed using python 3.8 on Ubuntu 18.04. NVIDIA GPUs are needed. The code is developed and tested using 3 NVIDIA T4 GPU cards. Other platforms or GPU cards are not fully tested.
## Quick Start

### Requirements

- Linux (Windows is not officially supported)
- Python 3.8+
- PyTorch 1.8+
- CUDA 11.1+
- GCC 7+
- Numpy
- cv2
- json_tricks


## Quick start
### Installation
1. Install pytorch >= v1.8.0 following [official instruction](https://pytorch.org/).
   **Note that if you use pytorch's version < v1.0.0, you should following the instruction at <https://github.com/Microsoft/human-pose-estimation.pytorch> to disable cudnn's implementations of BatchNorm layer. We encourage you to use higher pytorch's version(>=v1.0.0)**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Make libs:
   ```
   cd ${POSE_ROOT}/lib
   make
   ```
5. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
4. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```
### Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/). The original annotation files are in matlab format. We have converted them into json format, you also need to download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW00SqrairNetmeVu4) or [GoogleDrive](https://drive.google.com/drive/folders/1En_VqmStnsXMdldXA6qpqEyDQulnmS3a?usp=sharing).
Extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- mpii
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. We also provide person detection result of COCO val2017 and test-dev2017 to reproduce our multi-person pose estimation results. Please download from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzzDXoz5BeFl8sWM-) or [GoogleDrive](https://drive.google.com/drive/folders/1fRUDNUDxe9fjqcRZ2bnF_TKMlO0nB_dk?usp=sharing).
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        |-- person_detection_results
        |   |-- COCO_val2017_detections_AP_H_56_person.json
        |   |-- COCO_test-dev2017_detections_AP_H_609_person.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training and Testing
#### Testing on MPII dataset
 

```
python tools/test.py \
    --cfg experiments/mpii/lite_hrnet/lite_hrnet_30_384x288.yaml \
    TEST.MODEL_FILE path/to/your_model.pth
```

#### Training on MPII dataset

```
python tools/train.py \
    --cfg experiments/mpii/lite_hrnet/lite_hrnet_30_384x288.yaml 
```

#### Testing on COCO val2017 dataset
 

```
python tools/test.py \
    --cfg experiments/coco/lite_hrnet/lite_hrnet_30_384x288.yaml \
    TEST.MODEL_FILE path/to/your_model.pth \
    TEST.USE_GT_BBOX False
```

#### Training on COCO train2017 dataset

```
python tools/train.py \
    --cfg experiments/coco/lite_hrnet/lite_hrnet_30_384x288.yaml \
```
### Demo and visualize
```
python demo/demo.py \
     --cfg experiments/coco/lite_hrnet_30_384x288.yaml \
     TEST.MODEL_FILE path/to/your_model.pth \
     --image path/to/your/image.jpg

*note: you can set TEST.MODEL_FILE in your config file 
```

## Convert to ONNX
```
python demo/convert2onnx.py \
     --cfg experiments/coco/lite_hrnet_30_384x288.yaml \
     TEST.MODEL_FILE path/to/your_model.pth

*note: you can set TEST.MODEL_FILE in your config file 
```

## Acknowledgement

Thanks to:

- [MMPose](https://github.com/open-mmlab/mmpose)
- [HRNet](https://github.com/HRNet/deep-high-resolution-net.pytorch)
- [Lite_HRNet](https://github.com/HRNet/Lite-HRNet)

## Citation
```
@inproceedings{Yulitehrnet21,
  title={Lite-HRNet: A Lightweight High-Resolution Network},
  author={Yu, Changqian and Xiao, Bin and Gao, Changxin and Yuan, Lu and Zhang, Lei and Sang, Nong and Wang, Jingdong},
  booktitle={CVPR},
  year={2021}
}

@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal={TPAMI}
  year={2019}
}

```
