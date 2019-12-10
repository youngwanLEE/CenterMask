# [CenterMask](https://arxiv.org/abs/1911.06667) : Real-Time Anchor-Free Instance Segmentation


![architecture](figures/architecture.png)

## Abstract

We propose a simple yet efficient anchor-free instance segmentation, called *CenterMask*, that adds a novel spatial attention-guided mask (SAG-Mask) branch to anchor-free one stage object detector (FCOS) in the same vein with Mask R-CNN. Plugged into the FCOS object detector, the SAG-Mask branch predicts a segmentation mask on each box with the spatial attention map that helps to focus on informative pixels and suppress noise. We also present an improved VoVNetV2 with two effective strategies: adds (1) residual connection for alleviating the saturation problem of larger VoVNet and (2) effective Squeeze-Excitation (eSE) deals with the information loss problem of original SE. With SAG-Mask and VoVNetV2, we deign CenterMask and CenterMask-Lite that are targeted to large and small models, respectively. CenterMask outperforms all previous state-of-the-art models at a much faster speed. CenterMask-Lite also achieves 33.4% mask AP / 38.0% box AP, outperforming YOLACT by 2.6 / 7.0 AP gain, respectively, at over 35fps on Titan Xp. We hope that CenterMask and VoVNetV2 can serve as a solid baseline of real-time instance segmentation and backbone network for various vision tasks, respectively. 


## Updates
- Open the official repo and code will be released after refactoring. (05/12/2019)
- Release code and MobileNetV2 & ResNet backbone models shown in the [[`paper`]](https://arxiv.org/abs/1911.06667).(10/12/2019)


## Notes
- The release of VoVNetV2 models is delayed due to the internal affairs concerning patent.



### Environment
- V100 or Titan Xp GPU
- CUDA 10.0 
- cuDNN7.3 
- pytorch1.1
- Implemented on [fcos](https://github.com/tianzhi0549/FCOS) and [maskrcn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) 

### coco test-dev results

|Detector | Backbone |  epoch |   Mask AP (AP/APs/APm/APl) | Box AP (AP/APs/APm/APl) |  Time (ms) | GPU |Weight |
|----------|----------|:--------------:|:-------------------:|:------------------------:|:--------------------------:| :---:|:---:|
| [ShapeMask](https://arxiv.org/abs/1904.03239)     | R-101-FPN   |N/A |            37.4/16.1/40.1/53.8                  | 42.2/24.9/45.2/52.7      | 125| V100| - |
 | [TensorMask](https://arxiv.org/abs/1903.12174)     | R-101-FPN  | 72 |  37.1/17.4/39.1/51.6         | -                  | 380      |V100| - |
 [RetinaMask](https://arxiv.org/abs/1901.03353)    | R-101-FPN   |  24 |    34.7/14.3/36.7/50.5     | 41.4/23.0/44.5/53.0                  | 98  |V100| - |
| [Mask R-CNN](https://arxiv.org/abs/1703.06870)     | R-101-FPN   | 24 |   37.9/18.1/40.3/53.3       | 42.2/24.9/45.2/52.7                  |  94     |V100| -                          |[link](https://dl.dropbox.com/s/rs1rgl5lupw576a/FRCN-V-57-FPN-2x-norm.pth?dl=1)|
| **CenterMask**    | R-101-FPN   |    24 |   38.3/17.7/40.8/54.5|     43.1/25.2/46.1/54.4              | **72**      |V100| [link](https://dl.dropbox.com/s/9w17k9iiihob8vx/centermask-R-101-ms-2x.pth?dl=1)|
||
| [YOLACT-400](https://arxiv.org/abs/1904.02689)     | R-101-FPN   |    48 |    24.9/5.0/25.3/45.0    |         28.4/10.7/28.9/43.1          |  22   | Xp |-|
| **CenterMask-Lite**    | MV2-FPN   |   24 |  25.2/8.6/25.8/38.2     |         28.8/14.0/30.7/37.8          | **20**      | Xp |[link](https://dl.dropbox.com/s/gsrxx63p0wtxsa3/centermask-lite-M-v2-ms-bs32-1x.pth?dl=1)|
||
| [YOLACT-550](https://arxiv.org/abs/1904.02689)     | R-50-FPN   |   48 |    28.2/9.2/29.3/44.8    | 30.3/14.0/31.2/43.0                  |   23    |Xp|-|
| [YOLACT-550](https://arxiv.org/abs/1904.02689)     | R-101-FPN   |   48 |     29.8/9.9/31.3/47.7   | 31.0/14.4/31.8/43.7                  |   30    | Xp| - |
| **CenterMask-Lite**     | R-50-FPN   |   24 |     31.9/12.4/33.8/47.3   | 35.3/18.2/38.6/46.2                  |   29    | Xp                         |[link](https://dl.dropbox.com/s/2enqxenccz4xy6l/centermask-lite-R-50-ms-bs32-1x.pth?dl=1)|

*Note that RetinaMask, Mask R-CNN, and CenterMask are implemented by using same baseline code([maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)) and all models are trained using multi-scale training augmentation.*

*We expect that if we implement our CenterMask based on [detectron2](https://github.com/facebookresearch/detectron2), it will get better performance.*


## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions which is orginate from [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark).

## Training
Follow [the instructions](https://github.com/facebookresearch/maskrcnn-benchmark#multi-gpu-training) of  [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) guides.

If you want multi-gpu (e.g.,8) training,

```bash
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_net.py --config-file "configs/centermask/centermask_R_50_FPN_1x.yaml" 
```


## Evaluation

Follow [the instruction](https://github.com/facebookresearch/maskrcnn-benchmark#evaluation) of [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)

First of all, you have to download the weight file you want to inference.

For examaple (CenterMask-Lite-R-50),
##### multi-gpu evaluation & test batch size 16,
```bash
wget https://dl.dropbox.com/s/2enqxenccz4xy6l/centermask-lite-R-50-ms-bs32-1x.pth
export NGPUS=8
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_net.py --config-file "configs/centermask/centermask_R_50_FPN_lite_res600_ms_bs32_1x.yaml"   TEST.IMS_PER_BATCH 16 MODEL.WEIGHT centermask-lite-R-50-ms-bs32-1x.pth
```

##### For single-gpu evaluation & test batch size 1,
```bash
wget https://dl.dropbox.com/s/2enqxenccz4xy6l/centermask-lite-R-50-ms-bs32-1x.pth
CUDA_VISIBLE_DEVICES=0
python tools/test_net.py --config-file "configs/centermask/centermask_R_50_FPN_lite_res600_ms_bs32_1x.yaml" TEST.IMS_PER_BATCH 1 MODEL.WEIGHT centermask-lite-R-50-ms-bs32-1x.pth
```


## TODO
 - [ ] train-time augmentation + 3x schedule for comparing with detectron2 models
 - [ ] quick-demo
 - [ ] ResNet-50 & ResNeXt-101-32x8d
 - [ ] arxiv paper update
 - [ ] VoVNetV2 backbones



## Performance
![vizualization](figures/quality.png)
![results_table](figures/results.png)

## Citing CenterMask

Please cite our paper in your publications if it helps your research:

    @article{lee2019centermask,
      title={CenterMask: Real-Time Anchor-Free Instance Segmentation},
      author={Lee, Youngwan and Park, Jongyoul},
      journal={arXiv preprint arXiv:1911.06667},
      year={2019}
    }
    
