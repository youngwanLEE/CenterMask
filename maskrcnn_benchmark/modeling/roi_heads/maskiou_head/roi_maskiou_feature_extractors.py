# Mask Scoring R-CNN
# Wriiten by zhaojin.huang, 2018-12.
# Modified by Youngwan Lee, 2019-10.

import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling.make_layers import make_conv3x3


class MaskIoUFeatureExtractor(nn.Module):
    """
    MaskIou head feature extractor.
    """

    def __init__(self, cfg, in_channels):
        super(MaskIoUFeatureExtractor, self).__init__()

        layers = cfg.MODEL.ROI_MASKIOU_HEAD.CONV_LAYERS
        resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION // 2
        input_features = in_channels + 1
        fc_input_size = layers[0] * resolution * resolution

        self.blocks = []
        stride = 1
        for layer_idx, layer_features in enumerate(layers, 1):
            layer_name = "maskiou_fcn{}".format(layer_idx)
            if layer_idx == len(layers):
                stride = 2
            module = make_conv3x3(input_features, layer_features, stride=stride)
            self.add_module(layer_name, module)
            input_features = layer_features
            self.blocks.append(layer_name)

        self.maskiou_fc1 = nn.Linear(fc_input_size, 1024)
        self.maskiou_fc2 = nn.Linear(1024, 1024)

        for l in [self.maskiou_fc1, self.maskiou_fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)

    def forward(self, x, mask):
        mask_pool = F.max_pool2d(mask, kernel_size=2, stride=2)
        x = torch.cat((x, mask_pool), 1)
        for layer_name in self.blocks:
            x = F.relu(getattr(self, layer_name)(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.maskiou_fc1(x))
        x = F.relu(self.maskiou_fc2(x))
        return x


_ROI_MASKIOU_FEATURE_EXTRACTORS = {
    "MaskIoUFeatureExtractor": MaskIoUFeatureExtractor,
}


def make_roi_maskiou_feature_extractor(cfg, in_channels):
    # func = MaskIoUFeatureExtractor
    func = _ROI_MASKIOU_FEATURE_EXTRACTORS[
    cfg.MODEL.ROI_MASKIOU_HEAD.FEATURE_EXTRACTOR]
    return func(cfg, in_channels)
