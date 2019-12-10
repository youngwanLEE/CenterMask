# Mask Scoring R-CNN
# Wriiten by zhaojin.huang, 2018-12.

from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.layers import Conv2d
from maskrcnn_benchmark.layers import ConvTranspose2d


class MaskIoUPredictor(nn.Module):
    def __init__(self, cfg):
        super(MaskIoUPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES

        self.maskiou = nn.Linear(1024, num_classes)

        nn.init.normal_(self.maskiou.weight, mean=0, std=0.01)
        nn.init.constant_(self.maskiou.bias, 0)


    def forward(self, x):
        maskiou = self.maskiou(x)
        return maskiou

class MaskIoUGAPPredictor(nn.Module):
    def __init__(self, cfg):
        super(MaskIoUGAPPredictor, self).__init__()
        num_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        in_channel = cfg.MODEL.ROI_MASK_HEAD.CONV_LAYERS[0]

        self.maskiou = nn.Linear(in_channel, num_classes)

        nn.init.normal_(self.maskiou.weight, mean=0, std=0.01)
        nn.init.constant_(self.maskiou.bias, 0)


    def forward(self, x):
        maskiou = self.maskiou(x)
        return maskiou

_ROI_MASKIOU_PREDICTOR = {"MaskIoUPredictor": MaskIoUPredictor,
                          "MaskIoUGAPPredictor": MaskIoUGAPPredictor,
                          }


def make_roi_maskiou_predictor(cfg):
    func = _ROI_MASKIOU_PREDICTOR[cfg.MODEL.ROI_MASKIOU_HEAD.PREDICTOR]
    return func(cfg)
