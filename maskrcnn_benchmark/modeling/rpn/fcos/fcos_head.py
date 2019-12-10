# Modified by Youngwan Lee (ETRI). All Rights Reserved.
import math
import torch
from torch import nn

from maskrcnn_benchmark.layers import Scale


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            cls_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.dense_points = cfg.MODEL.FCOS.DENSE_POINTS
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            logits.append(self.cls_logits(cls_tower))
            box_tower = self.bbox_tower(feature)
            centerness.append(self.centerness(box_tower))
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(box_tower)
            )))

        return logits, bbox_reg, centerness

class FCOSSharedHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSSharedHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES - 1
        self.identity = cfg.MODEL.FCOS.RESIDUAL_CONNECTION
        shared_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            shared_tower.append(
                nn.Conv2d(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            shared_tower.append(nn.GroupNorm(32, in_channels))
            shared_tower.append(nn.ReLU())


        self.add_module('shared_tower', nn.Sequential(*shared_tower))
        self.dense_points = cfg.MODEL.FCOS.DENSE_POINTS
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1 * self.dense_points, kernel_size=3, stride=1,
            padding=1
        )

        # initialization
        for modules in [self.shared_tower, self.cls_logits, 
                          self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            if self.identity:
                shared_tower = self.shared_tower(feature) + feature
            else:
                shared_tower = self.shared_tower(feature)
            logits.append(self.cls_logits(shared_tower))
            centerness.append(self.centerness(shared_tower))
            bbox_reg.append(torch.exp(self.scales[l](
                self.bbox_pred(shared_tower)
            )))

        return logits, bbox_reg, centerness