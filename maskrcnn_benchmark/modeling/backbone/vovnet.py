# Copyright (c) Youngwan Lee (ETRI) All Rights Reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from maskrcnn_benchmark.utils.registry import Registry
from maskrcnn_benchmark.layers import FrozenBatchNorm2d, DFConv2d
from maskrcnn_benchmark.modeling.make_layers import group_norm

_GN = False

VoVNet19_eSE_FPNStagesTo5 = {
    'config_stage_ch': [128, 160, 192, 224],
    'config_concat_ch': [256, 512, 768, 1024],
    'layer_per_block': 3,
    'block_per_stage': [1, 1, 1, 1],
    'eSE' : True
}

VoVNet39_eSE_FPNStagesTo5 = {
    'config_stage_ch': [128, 160, 192, 224],
    'config_concat_ch': [256, 512, 768, 1024],
    'layer_per_block': 5,
    'block_per_stage': [1, 1, 2, 2],
    'eSE' : True
}

VoVNet57_eSE_FPNStagesTo5 = {
    'config_stage_ch': [128, 160, 192, 224],
    'config_concat_ch': [256, 512, 768, 1024],
    'layer_per_block': 5,
    'block_per_stage': [1, 1, 4, 3],
    'eSE' : True
}

VoVNet99_eSE_FPNStagesTo5 = {
    'config_stage_ch': [128, 160, 192, 224],
    'config_concat_ch': [256, 512, 768, 1024],
    'layer_per_block': 5,
    'block_per_stage': [1, 3, 9, 3],
    'eSE' : True
}

_STAGE_SPECS = Registry({
    "V-19-eSE-FPN-RETINANET": VoVNet19_eSE_FPNStagesTo5,
    "V-39-eSE-FPN-RETINANET": VoVNet39_eSE_FPNStagesTo5,
    "V-57-eSE-FPN-RETINANET": VoVNet57_eSE_FPNStagesTo5,
    "V-99-eSE-FPN-RETINANET": VoVNet99_eSE_FPNStagesTo5,
    "V-19-eSE-FPN": VoVNet19_eSE_FPNStagesTo5,
    "V-39-eSE-FPN": VoVNet39_eSE_FPNStagesTo5,
    "V-57-eSE-FPN": VoVNet57_eSE_FPNStagesTo5,
    "V-99-eSE-FPN": VoVNet99_eSE_FPNStagesTo5
})

def freeze_bn_params(m):
    """Freeze all the weights by setting requires_grad to False
    """
    m.eval()
    for p in m.parameters():
        p.requires_grad = False

def conv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, padding=1):
    """3x3 convolution with padding"""
    return [
        (f'{module_name}_{postfix}/conv',
         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False)),
        (f'{module_name}_{postfix}/norm',
            group_norm(out_channels) if _GN else FrozenBatchNorm2d(out_channels)
        ),
        (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))
    ]

def DFConv3x3(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=3, 
                with_modulated_dcn=None, deformable_groups=None):
    """3x3 convolution with padding"""
    return [
        (f'{module_name}_{postfix}/conv',
         DFConv2d(in_channels, out_channels, with_modulated_dcn=with_modulated_dcn,
            kernel_size=kernel_size, stride=stride, groups=groups, 
            deformable_groups=deformable_groups, bias=False)),
        (f'{module_name}_{postfix}/norm',
            group_norm(out_channels) if _GN else FrozenBatchNorm2d(out_channels)
        ),
        (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))
    ]

def conv1x1(in_channels, out_channels, module_name, postfix, stride=1, groups=1, kernel_size=1, padding=0):
    """1x1 convolution with padding"""
    return [
        (f'{module_name}_{postfix}/conv',
         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                   bias=False)),
        (f'{module_name}_{postfix}/norm',
            group_norm(out_channels) if _GN else FrozenBatchNorm2d(out_channels)
        ),
        (f'{module_name}_{postfix}/relu', nn.ReLU(inplace=True))
    ]

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class eSEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channel,channel, kernel_size=1,
                             padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x


class _OSA_module(nn.Module):

    def __init__(self, 
                 in_ch, 
                 stage_ch, 
                 concat_ch, 
                 layer_per_block, 
                 module_name,
                 SE=False,
                 identity=False,
                 dcn_config={}
                 ):

        super(_OSA_module, self).__init__()

        self.identity = identity
        self.layers = nn.ModuleList()
        in_channel = in_ch
        with_dcn = dcn_config.get("stage_with_dcn", False)
        for i in range(layer_per_block):
            if with_dcn:
                deformable_groups = dcn_config.get("deformable_groups", 1)
                with_modulated_dcn = dcn_config.get("with_modulated_dcn", False)
                self.layers.append(nn.Sequential(OrderedDict(DFConv3x3(in_channel, stage_ch, module_name, i, 
                    with_modulated_dcn=with_modulated_dcn, deformable_groups=deformable_groups))))
            else:
                self.layers.append(nn.Sequential(OrderedDict(conv3x3(in_channel, stage_ch, module_name, i))))
            in_channel = stage_ch

        # feature aggregation
        in_channel = in_ch + layer_per_block * stage_ch
        self.concat = nn.Sequential(OrderedDict(conv1x1(in_channel, concat_ch, module_name, 'concat')))

        self.ese = eSEModule(concat_ch)


    def forward(self, x):

        identity_feat = x

        output = []
        output.append(x)
        for layer in self.layers:
            x = layer(x)
            output.append(x)

        x = torch.cat(output, dim=1)
        xt = self.concat(x)

        xt = self.ese(xt)

        if self.identity:
            xt = xt + identity_feat

        return xt


class _OSA_stage(nn.Sequential):

    def __init__(self, 
                 in_ch, 
                 stage_ch, 
                 concat_ch, 
                 block_per_stage, 
                 layer_per_block, 
                 stage_num,
                 SE=False,
                 dcn_config={},
                ):
        super(_OSA_stage, self).__init__()

        if not stage_num == 2:
            self.add_module('Pooling', nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

        if block_per_stage !=1:
            SE = False
        module_name = f'OSA{stage_num}_1'
        self.add_module(module_name, _OSA_module(in_ch, 
                                                 stage_ch, 
                                                 concat_ch, 
                                                 layer_per_block, 
                                                 module_name,
                                                 SE=SE,
                                                 dcn_config=dcn_config
                                                 ))
        for i in range(block_per_stage - 1):
            if i != block_per_stage -2: #last block
                SE = False
            module_name = f'OSA{stage_num}_{i + 2}'
            self.add_module(module_name,
                            _OSA_module(concat_ch, 
                                        stage_ch, 
                                        concat_ch, 
                                        layer_per_block, 
                                        module_name, 
                                        SE=SE,
                                        identity=True,
                                        dcn_config=dcn_config
                                        ))



class VoVNet(nn.Module):

    def __init__(self, cfg):

        super(VoVNet, self).__init__()

        global _GN
        _GN = cfg.MODEL.VOVNET.USE_GN
        stage_specs = _STAGE_SPECS[cfg.MODEL.BACKBONE.CONV_BODY]

        config_stage_ch = stage_specs['config_stage_ch']
        config_concat_ch = stage_specs['config_concat_ch']
        block_per_stage = stage_specs['block_per_stage']
        layer_per_block = stage_specs['layer_per_block']
        SE = stage_specs['eSE']

        # self.stem = nn.Sequential()
        # Stem module
        stem = conv3x3(3, 64, 'stem', '1', 2)
        stem += conv3x3(64, 64, 'stem', '2', 1)
        stem += conv3x3(64, 128, 'stem', '3', 2)
        self.add_module('stem', nn.Sequential((OrderedDict(stem))))

        stem_out_ch = [128]
        in_ch_list = stem_out_ch + config_concat_ch[:-1]
        # OSA stages
        self.stage_names = []
        for i in range(4):  # num_stages
            name = 'stage%d' % (i + 2)
            self.stage_names.append(name)
            self.add_module(name, _OSA_stage(in_ch_list[i],
                                             config_stage_ch[i],
                                             config_concat_ch[i],
                                             block_per_stage[i],
                                             layer_per_block,
                                             i + 2,
                                             SE,
                                            dcn_config = {
                                                "stage_with_dcn": cfg.MODEL.VOVNET.STAGE_WITH_DCN[i],
                                                "with_modulated_dcn": cfg.MODEL.VOVNET.WITH_MODULATED_DCN,
                                                "deformable_groups": cfg.MODEL.VOVNET.DEFORMABLE_GROUPS,
                                            }
            ))

        # initialize weights
        self._initialize_weights()
        # Optionally freeze (requires_grad=False) parts of the backbone
        self._freeze_backbone(cfg.MODEL.BACKBONE.FREEZE_CONV_BODY_AT)


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        # freeze BN layers
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                freeze_bn_params(m)
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem # stage 0 is the stem
            else:
                m = getattr(self, "stage" + str(stage_index+1))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        x = self.stem(x)
        outputs = []
        for name in self.stage_names:
            x = getattr(self, name)(x)
            outputs.append(x)

        return outputs