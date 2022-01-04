# Copyright (c) 2018-2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from util.loss import FocalLoss
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
def get_channels(layers):
    ch = 0
    for name, layer in layers.named_modules():
        if isinstance(layer, nn.BatchNorm2d):
            ch = max(ch, layer.num_features)
    return ch

class up_conv(nn.Module):
    def __init__(self, ch_in):
        super(up_conv, self).__init__()
        self.Conv1x1 = nn.Conv2d(ch_in, 256, kernel_size=1, stride=1, bias=False)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.Conv3x3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
            )
    def forward(self, c, p): 
        up_p = self.up(p)
        down_c = self.Conv1x1(c)
        out = self.Conv3x3(up_p+down_c)
        return out


class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        if backbone == 'resnet18':
            backbone = resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            backbone = resnet34(pretrained=pretrained)
        elif backbone == 'resnet50':
            backbone = resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            backbone = resnet101(pretrained=pretrained)
        else:  # backbone == 'resnet152':
            backbone = resnet152(pretrained=pretrained)

        self.out_channels = [256,256,256,256]
        self.layers = list(backbone.children())

        self.block1 = nn.Sequential(*self.layers[:3])
        self.block2 = nn.Sequential(*self.layers[3:5])
        self.block3 = self.layers[5]
        self.block4 = self.layers[6]
        self.block5 = self.layers[7]
        self.maxpool = nn.MaxPool2d(2,2)
        block5_out = get_channels(self.block5) #2048
        block4_out = get_channels(self.block4) #1024
        block3_out = get_channels(self.block3) #512
        block2_out = get_channels(self.block2) #256
        
        self.P2_out = nn.Sequential(
                      nn.Conv2d(block5_out, 256, kernel_size=1, stride=1, bias=False),
                      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True),
                      nn.BatchNorm2d(256),
                      nn.ReLU(inplace=True))
        self.P3_out = up_conv(ch_in=block4_out)
        self.P4_out = up_conv(ch_in=block3_out)
        self.P5_out = up_conv(ch_in=block2_out)

        self._init_weights_list([self.P2_out, self.P3_out, self.P4_out, self.P5_out])

    def forward(self, x):
        c1 = self.block1(x) 
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)

        p2 = self.P2_out(self.maxpool(c5))
        p3 = self.P3_out(self.maxpool(c4),p2)
        p4 = self.P4_out(self.maxpool(c3),p3)
        p5 = self.P5_out(self.maxpool(c2),p4)
        return p2,p3,p4,p5

    def _init_weights_list(self,layer_list):
        for layers in layer_list:
            layers.apply(self._init_weights)

    def _init_weights(self,m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)

class SSD(nn.Module):
    def __init__(self, backbone=ResNet('resnet50'), n_classes=10):
        super().__init__()

        self.feature_extractor = backbone

        self.n_classes = n_classes
        self.num_defaults = [4,4,4,4]
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            self.loc.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(nn.Conv2d(oc, nd * self.n_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _init_weights(self):
        layers = [*self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1: nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.n_classes, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, x):
        p2,p3,p4,p5 = self.feature_extractor(x)
        
        detection_feed = [p2,p3,p4,p5]
        # Feature Map 8x8x4, 16x16x4, 32x32x4,64x64x4
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        return locs, confs


class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, dboxes, method='L2'):
        super(Loss, self).__init__()
        self.scale_xy = 1.0/dboxes.scale_xy
        self.scale_wh = 1.0/dboxes.scale_wh
        self.method = method
        self.bbox_loss = nn.MSELoss(reduction='none') if method == 'L2' else nn.SmoothL1Loss(reduction='none')
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)
        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.con_loss = FocalLoss(gamma=2, alpha=0.25, reduction='none')

    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.dboxes[:, :2, :])/self.dboxes[:, 2:, ]
        gwh = self.scale_wh*((loc[:, 2:, :]+1e-6)/self.dboxes[:, 2:, :]).log() # prevent for -inf ( (0+1e6)/number )
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def forward(self, ploc, plabel, gloc, glabel):
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """
        mask = glabel > 0
        pos_num = mask.sum(dim=1)
        vec_gd = self._loc_vec(gloc)
        # sum on four coordinates, and mask
        b_loss = self.bbox_loss(ploc, vec_gd).sum(dim=1)
        b_loss = (mask.float()*b_loss).sum(dim=1)

        out_bbox_loss = (b_loss.detach() / (pos_num+1e-6)).mean().item()
        # hard negative mining
        con = self.con_loss(plabel, glabel)
        # postive mask will never selected
        con_neg = con.clone()
        con_neg[mask] = 0
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)

        # number of negative three times positive
        neg_num = torch.clamp(3*pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num

        #print(con.shape, mask.shape, neg_mask.shape)
        closs = (con*((mask + neg_mask).float())).sum(dim=1)
        # avoid no object detected
        total_loss = b_loss + closs
        num_mask = (pos_num > 0).float()
        pos_num = pos_num.float().clamp(min=1e-6)
        ret = (total_loss*num_mask/pos_num).mean(dim=0)
        out_class_loss = (closs.detach()/pos_num).mean().item()

        return ret, out_bbox_loss, out_class_loss
