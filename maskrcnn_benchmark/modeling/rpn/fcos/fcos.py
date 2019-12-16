# Modified by Youngwan Lee (ETRI). All Rights Reserved.
import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_fcos_postprocessor
from .loss import make_fcos_loss_evaluator

from maskrcnn_benchmark.layers import Scale
from .fcos_head import FCOSHead, FCOSSharedHead

_FCOS_HEAD = {
    "FCOSHead": FCOSHead,
    "FCOSSharedHead":FCOSSharedHead
}

class FCOSModule(torch.nn.Module):
    """
    Module for FCOS computation. Takes feature maps from the backbone and
    FCOS outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(FCOSModule, self).__init__()

        # head = FCOSHead(cfg, in_channels)
        fcos_head = _FCOS_HEAD[cfg.MODEL.FCOS.HEAD]
        head = fcos_head(cfg, in_channels)

        box_selector_train = make_fcos_postprocessor(cfg, is_train=True)
        box_selector_test = make_fcos_postprocessor(cfg, is_train=False)

        loss_evaluator = make_fcos_loss_evaluator(cfg)
        self.head = head
        self.box_selector_train = box_selector_train
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.dense_points = cfg.MODEL.FCOS.DENSE_POINTS

    def forward(self, images, features, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness = self.head(features)
        locations = self.compute_locations(features)

        if self.training:
            return self._forward_train(
                locations, box_cls,
                box_regression,
                centerness, targets, images.image_sizes
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression,
                centerness, images.image_sizes
            )


    def _forward_train(self, locations, box_cls, box_regression, centerness, targets, image_sizes):
        loss_box_cls, loss_box_reg, loss_centerness = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }
        boxes = self.box_selector_train(
            locations, box_cls, box_regression, 
            centerness, image_sizes, targets
        )
        return boxes, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, image_sizes):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression,
            centerness, image_sizes
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        locations = self.get_dense_locations(locations, stride, device)
        return locations

    def get_dense_locations(self, locations, stride, device):
        if self.dense_points <= 1:
            return locations
        center = 0
        step = stride // 4
        l_t = [center - step, center - step]
        r_t = [center + step, center - step]
        l_b = [center - step, center + step]
        r_b = [center + step, center + step]
        if self.dense_points == 4:
            points = torch.cuda.FloatTensor([l_t, r_t, l_b, r_b], device=device)
        elif self.dense_points == 5:
            points = torch.cuda.FloatTensor([l_t, r_t, [center, center], l_b, r_b], device=device)
        else:
            print("dense points only support 1, 4, 5")
        points.reshape(1, -1, 2)
        locations = locations.reshape(-1, 1, 2).to(points)
        dense_locations = points + locations
        dense_locations = dense_locations.view(-1, 2)
        return dense_locations


def build_fcos(cfg, in_channels):
    return FCOSModule(cfg, in_channels)
