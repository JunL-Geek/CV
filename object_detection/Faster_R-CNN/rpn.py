import numpy as np
import torch
import torch.nn as nn
from anchors import generate_anchor_base, _enumerate_shifted_anchor
import torch.nn.functional as F


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512, ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32], feat_stride=16, mode='training'):
        super(RegionProposalNetwork, self).__init__()
        self.mode = mode
        self.feat_stride = feat_stride
        self.proposal_layer = ProposalCreator(mode=self.mode)
        # generate anchor_base (9, 4)

        self.anchor_base = generate_anchor_base(anchor_scales=anchor_scales, ratios=ratios)
        n_anchor = self.anchor_base.shape[0]

        # aggregate features
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        # regression prediction to roi
        self.los = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

        # classification whether the roi contain an object
        self.score = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)

        normal_init(self.conv1, 0, 0.01)
        normal_init(self.los, 0, 0.01)
        normal_init(self.score, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        # x.shape = (batch_num, c, h, w)
        n, _, h, w = x.shape

        x = F.relu(self.conv1(x))
        rpn_locs = self.loc(x)
        # rpn_locs.shape = (n, h * w * 9, 4)
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)

        rpn_scores = self.score(x)
        # rpn_scores.shape = (n, h * w * 9, 2)
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)

        rpn_prob = F.softmax(rpn_scores, dim=-1)
        # rpn_fg_prob.shape = (n, w * h * 9)
        rpn_fg_prob = rpn_prob[:, :, 1].contiguous().view(n, -1)

        # generate standard prior anchor for each feature map, shape = (h * w * 9, 4)
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(loc=rpn_locs, score=rpn_scores, anchor=anchor,
                                      img_size=img_size, scale=scale)
            rois.append(roi)
            batch_index = i * torch.ones((len(roi),))
            roi_indices.append(batch_index)

        rois = torch.cat(rois, dim=0)
        roi_indices = torch.cat(roi_indices, dim=0)

        return rpn_locs, rpn_scores, rois, roi_indices, anchor


class ProposalCreator():
    def __init__(self, mode, nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=600,
                 n_test_pre_nms=3000,
                 n_test_post_nms=300,
                 min_size=16):
        super(ProposalCreator, self).__init__()
        self.mode = mode
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.):
        if self.mode == "training":
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # ndarry to tensor
        anchor = torch.from_numpy(anchor)

        roi = loc2bbox(anchor, loc)

        # prevent the roi from going beyond the edge of the image
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min=0, max=img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min=0, max=img_size[0])

        # remove the roi have inappropriate width and height
        min_size = self.min_size * scale
        keep = torch.where(((roi[:, 2] - roi[:, 0] + 1) >= min_size) & ((roi[:, 3] - roi[:, 1] + 1) >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # order the roi and choose top n_pre_nms
        order = torch.argsort(score, descending=True)
        if n_pre_nms > 0:
            keep = order[:n_pre_nms]
        roi = roi[keep, :]
        score = score[keep]

        # nms
        keep = nms(roi, score, self.nms_thresh)
        keep = keep[:n_post_nms]
        roi = roi[keep, :]
        score = score[keep]

        return roi


def loc2bbox(src_bbox, loc):
    """

    :param src_bbox: (38 * 38 * 9, 4)
    :param loc: (38 * 38, 9 * 4)
    :return: dst_bbox: (38 * 38, 9 * 4)
    """
    src_width = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)
    src_height = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)
    src_crt_x = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width
    src_crt_y = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height

    dx = loc[:, 0::4]
    dy = loc[:, 1::4]
    dw = loc[:, 2::4]
    dh = loc[:, 3::4]

    crt_x = dx * src_width + src_crt_x
    crt_y = dy * src_height + src_crt_y
    w = torch.exp(dw) * src_width
    h = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(loc)
    dst_bbox[:, 0::4] = crt_x - 0.5 * w
    dst_bbox[:, 1::4] = crt_y - 0.5 * h
    dst_bbox[:, 2::4] = crt_x + 0.5 * w
    dst_bbox[:, 3::4] = crt_y + 0.5 * h

    return dst_bbox

def normal_init(module, mean, stddev, truncated=False):
    if truncated:
        module.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)
    else:
        module.weight.data.normal_(mean, stddev)
        module.bias.data.zero_()

