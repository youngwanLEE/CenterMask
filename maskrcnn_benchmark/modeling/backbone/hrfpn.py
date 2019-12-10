"""
MIT License

Copyright (c) 2019 Microsoft

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class HRFPN(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 normalize=None,
                 pooling='AVG',
                 share_conv=False,
                 conv_stride=1,
                 num_level=5,
                 with_checkpoint=False):
        super(HRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_bias = normalize is None
        self.share_conv = share_conv
        self.num_level = num_level
        self.reduction_conv = nn.Sequential(
            nn.Conv2d(in_channels=sum(in_channels),
                      out_channels=out_channels,
                      kernel_size=1),
        )

        if self.share_conv:
            self.fpn_conv = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3, 
                stride=conv_stride,
                padding=1,
            )
        else:
            self.fpn_conv = nn.ModuleList()
            for i in range(self.num_level):
                self.fpn_conv.append(nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=conv_stride,
                    padding=1
                ))
        if pooling == 'MAX':
            print("Using AVG Pooling")
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d
        self.with_checkpoint = with_checkpoint

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,  a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outs = []
        outs.append(inputs[0])
        for i in range(1, len(inputs)):
            outs.append(F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))
        out = torch.cat(outs, dim=1)
        if out.requires_grad and self.with_checkpoint:
            out = checkpoint(self.reduction_conv, out)
        else:
            out = self.reduction_conv(out)
        outs = [out]
        for i in range(1, self.num_level):
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        outputs = []
        if self.share_conv:
            for i in range(self.num_level):
                outputs.append(self.fpn_conv(outs[i]))
        else:
            for i in range(self.num_level):
                if outs[i].requires_grad and self.with_checkpoint:
                    tmp_out = checkpoint(self.fpn_conv[i], outs[i])
                else:
                    tmp_out = self.fpn_conv[i](outs[i])
                outputs.append(tmp_out)
        return tuple(outputs)