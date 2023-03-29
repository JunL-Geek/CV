from classifier import Resnet50RoIHead
from rpn import RegionProposalNetwork
from ResNet50 import resnet50

import torch
import torch.nn as nn

class FasterRCNN(nn.Module):
    def __init__(self, num_classes, mode='training',
                 feat_stride=16, ratios=[0.5, 1, 2], anchor_scales = [8, 16, 32], backbone = 'resnet50', pretrained=False):
        super(FasterRCNN, self).__init__()
        self.feat_stride = feat_stride
        self.extractor, self.classifier = resnet50()
        self.rpn = RegionProposalNetwork(in_channels=1024,
                                         mid_channels=512,
                                         ratios=ratios,
                                         anchor_scales=anchor_scales,
                                         feat_stride=self.feat_stride,
                                         mode=mode)
        self.head = Resnet50RoIHead(n_classes=num_classes,
                                    roi_size=14,
                                    spatial_scale=1,
                                    classifier=self.classifier)

    def forward(self, x, scale=1., mode='forward'):
        if mode == 'forward':
            img_size = x.shape[2:]
            base_feature = self.extractor(x)
            _, _, rois, roi_indices, _ = self.rpn.forward(base_feature, img_size, scale)
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores, rois, roi_indices
        elif mode == 'extractor':
            base_feature = self.extractor(x)
            return base_feature
        elif mode == 'rpn':
            base_feature, img_size = x
            rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn.forward(base_feature, img_size, scale)
            return rpn_locs, rpn_scores, rois, roi_indices, anchor
        elif mode == 'head':
            base_feature, rois, roi_indices, img_size = x
            roi_cls_locs, roi_scores = self.head.forward(base_feature, rois, roi_indices, img_size)
            return roi_cls_locs, roi_scores

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == '__main__':
    faster_rcnn = FasterRCNN(num_classes=91).cuda()
    x = torch.randn(1, 3, 640, 640).cuda()
    rois_cls_locs, roi_scores, rois, roi_indices = faster_rcnn(x)
    print(rois_cls_locs.shape)
    print(roi_scores.shape)
    print(rois.shape)
    print(roi_indices.shape)