import torch
from torch import nn
from torchvision.ops import RoIPool
from rpn import normal_init

class Resnet50RoIHead(nn.Module):
    def __init__(self, n_classes, roi_size, spatial_scale, classifier):
        self.classifier = classifier

        # 对每个类别进行位置预测
        self.cls_loc = nn.Linear(2048, n_classes * 4)
        # 对每个类别进行分类回归
        self.score = nn.Linear(2048, n_classes)

        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)


    def forward(self, x, rois, roi_indices, img_size):
        """

        :param x: feature map, shape=(n, c, h, w)
        :param rois: prior anchor，shape=(n, num_anchors, 4)
        :param roi_indices: prior anchor indices to indicate which batch_index the anchor belong to，shape=(n, num_anchors)
        :param img_size: shape=(h, w)
        :return:
        """
        n, _, _, _ = x.shape
        if x.is_cuda:
            rois = rois.cuda()
            roi_indices = roi_indices.cuda()

        # 对rois 进行归一化
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0, 2]] = rois[:, [0, 2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1, 3]] = rois[:, [1, 3]] / img_size[0] * x.size()[2]

        # 利用prior anchor 对特征图进行池化
        pool = self.roi(x, rois_feature_map) # shape=(n, c, output_size[0], output_size[1])

        fc7 = self.classifier(pool)
        fc7 = fc7.view(fc7.size(0), -1)

        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        roi_cls_locs = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores = roi_scores.view(n, -1, roi_scores.size(1))

        return roi_cls_locs, roi_scores
