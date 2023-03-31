import numpy as np


def bbox_iou(bbox_a, bbox_b):
    """
    :param bbox_a: pridicted anchor shape = (n, 4)
    :param bbox_b: truth anchor shape = (k, 4)
    :return:
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        print("bbox_iou error")
        raise IndexError
    # shape = (n, k, 2)
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.maximum(bbox_b[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br-tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[2:] - bbox_a[:2], axis=1)
    area_b = np.prod(bbox_b[2:] - bbox_a[:2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i) # shape = (n, k)


def bbox2loc(src_bbox, dst_bbox):
    width = src_bbox[:, 2] - src_bbox[:, 0]
    height = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_x = src_bbox[:, 0] + width / 2
    ctr_y = src_bbox[:, 1] + height / 2

    dst_width = dst_bbox[:, 2] - dst_bbox[:, 0]
    dst_height = dst_bbox[:, 3] - dst_bbox[:, 1]
    dst_ctr_x = dst_bbox[:, 0] + dst_width / 2
    dst_ctr_y = dst_bbox[:, 1] + dst_height / 2

    # 防止除零错误
    eps = np.finfo(height.dtype).eps
    width = np.maximum(width, eps)
    height = np.maximum(height, eps)

    dx = (dst_ctr_x - ctr_x) / width
    dy = (dst_ctr_y - ctr_y) / height
    dw = np.log(dst_width / width)
    dh = np.log(dst_height / height)

    loc = np.vstack((dx, dy, dw, dh)).transpose()
    return loc


class AnchorTargetCreator(object):
    def __init__(self, n_samples=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, ratio_pos = 0.5):
        self.n_samples = n_samples
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.ratio_pos = ratio_pos

    def __call__(self, anchor, bbox):
        argmax_ious, labels = self._create_label(anchor, bbox)
        if (labels > 0).any():
            loc = bbox2loc(anchor, bbox[argmax_ious])
            return loc, labels
        else:
            return np.zeros_like(anchor), labels

    def _cals_ious(self, anchor, bbox):
        if len(bbox) == 0:
            return np.zeros(len(anchor), dtype=np.int32), np.zeros(len(anchor)), np.zeros(len(bbox))
        ious = bbox_iou(anchor, bbox)
        argmax_ious = np.argmax(ious, axis=1)
        max_ious = np.max(ious, axis=1)
        gt_argmax_ious = np.argmax(ious, axis=0)
        for i in range(len(anchor)):
            gt_argmax_ious[argmax_ious[i]] = i
        return argmax_ious, max_ious, gt_argmax_ious

    def _create_label(self, anchor, bbox):
        argmax_ious, max_ious, gt_argmax_ious = self._cals_ious(anchor, bbox)

        labels = np.zeros((len(anchor),), dtype=np.int32)
        labels.fill(-1)

        labels[max_ious < self.neg_iou_thresh] = 0
        labels[max_ious >= self.pos_iou_thresh] = 1
        if len(gt_argmax_ious) > 0:
            labels[gt_argmax_ious] = 1 # 保证每个真实框都对应一个正样本

        n_pos = self.ratio_pos * self.n_samples
        pos_index = np.where(labels == 1)[0]
        if len(pos_index) > n_pos:
            exclude_index = np.random.choice(pos_index, len(pos_index) - n_pos, replace=False)
            labels[exclude_index] = -1

        n_neg = self.n_samples - np.sum(labels==1)
        neg_index = np.where(labels == 0)[0]
        if len(neg_index) > n_neg:
            exclude_index = np.random.choice(neg_index, len(neg_index) - n_neg, replace=False)
            labels[exclude_index] = -1

        return argmax_ious, labels