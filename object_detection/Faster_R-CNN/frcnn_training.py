from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
    def __init__(self, n_samples=256, pos_iou_thresh=0.7, neg_iou_thresh=0.3, ratio_pos=0.5):
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


class ProposalTargetCreator(object):
    """

    """
    def __init__(self, n_sample=128, pos_ratio=0.5, pos_iou_thresh=0.5, neg_iou_thresh_high=0.5, neg_iou_thresh_low=0.0):
        self.n_sample = n_sample
        self.pos_ratio = pos_ratio
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh_high = neg_iou_thresh_high
        self.neg_iou_thresh_low = neg_iou_thresh_low

    def __call__(self, roi, bbox, label, loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        roi = np.concatenate((roi.detach().cpu().numpy(), bbox), axis=0) # ?????????????
        pos_roi_per_image = self.pos_ratio * self.n_sample

        iou = bbox_iou(roi, bbox)

        if len(bbox)==0:
            gt_assignment = np.zeros(len(roi), np.int32)
            iou_max = np.zeros(len(roi))
            gt_roi_label = np.zeros(len(roi))
        else:
            gt_assignment = iou.argmax(axis=1)
            iou_max = iou.max(axis=1)
            gt_roi_label = label(gt_assignment) + 1

        pos_index = np.where(iou_max >= self.pos_iou_thresh)[0]
        pos_roi_per_image = int(min(pos_roi_per_image, pos_index.size))
        if pos_index.size > 0:
            pos_index = np.random.choice(pos_index, size=pos_roi_per_image, replace=False)

        neg_index = np.where((iou_max < self.neg_iou_thresh_high) & (iou_max >= self.neg_iou_thresh_low))[0]
        neg_roi_per_image = self.n_sample - pos_roi_per_image
        neg_roi_per_image = int(min(neg_roi_per_image, neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(neg_index, size=neg_roi_per_image, replace=False)

        keep_index = np.append(pos_index, neg_index)

        sample_roi = roi[keep_index]
        if len(bbox) == 0:
            return sample_roi, np.zeros_like(sample_roi), gt_roi_label[keep_index]

        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]])
        gt_roi_loc = (gt_roi_loc / np.array(loc_normalize_std, np.float32))

        gt_roi_label = gt_roi_label[keep_index]
        gt_roi_label[pos_roi_per_image:] = 0
        return sample_roi, gt_roi_loc, gt_roi_label


class FasterRCNNTrainer(nn.Module):
    def __init__(self, model_train, optimizer):
        super(FasterRCNNTrainer, self).__init__()
        self.model_train = model_train
        self.optimizer = optimizer

        self.rpn_sigma = 1
        self.roi_sigma = 1

        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()


        self.loc_normalize_std = [0.1, 0.1, 0.2, 0.2]

    def _fast_rcnn_loc_loss(self, pred_loc, gt_loc, gt_label, sigma):
        pred_loc = pred_loc[gt_label > 0] # 正样本（认定为包含检测目标）的位置调整预测值
        gt_loc = gt_loc[gt_label > 0] # 正样本（认定为包含检测物体）的位置调整真实值

        sigma_squared = sigma ** 2
        regression_diff = gt_loc - pred_loc
        regression_diff = regression_diff.abs().float()
        regression_loss = torch.where(
            regression_diff < (1. / sigma_squared),
            0.5 * sigma_squared * regression_diff ** 2,
            regression_diff - 0.5 / sigma_squared
        )
        regression_loss = regression_loss.sum()
        num_pos = (gt_label > 0).sum().float()

        regression_loss /= torch.max(num_pos, torch.ones_like(num_pos))
        return regression_loss

    def forward(self, imgs, bboxes, labels, scale):
        n = imgs.shape[0]
        img_size = imgs.shape[2:]

        base_feature = self.model_train(imgs, mode = 'extractor')

        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.model_train(x = [base_feature, img_size], scale = scale, mode = 'rpn')

        rpn_loc_loss_all, rpn_cls_loss_all, roi_loc_loss_all, roi_cls_loss_all = 0, 0, 0, 0
        sample_rois, sample_indices, gt_roi_locs, gt_roi_labels = [], [], [], []

        for i in range(n):
            bbox = bboxes[i] # (n_t, 4)
            label = labels[i] # (n_t, 1)
            rpn_loc = rpn_locs[i] # (9 * h * w, 4) = (12996, 4)
            rpn_score = rpn_scores[i] # (12996, 2)
            roi = rois[i] # (600, 4)

            # (256, 4), (256, 1)
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(anchor[0].cpu().numpy(), bbox)
            gt_rpn_loc = torch.Tensor(gt_rpn_loc).type_as(rpn_locs)
            gt_rpn_label = torch.Tensor(gt_rpn_label).type_as(rpn_locs).long()

            rpn_loc_loss = self._fast_rcnn_loc_loss(rpn_loc, gt_rpn_loc, gt_rpn_label)
            rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label, ignore_index=-1)

            rpn_loc_loss_all += rpn_loc_loss
            rpn_cls_loss_all += rpn_cls_loss

            # (128, 4), (128, 4), (128, 1)
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi, bbox, label, self.loc_normalize_std)
            sample_rois.append(torch.Tensor(sample_roi).type_as(rpn_locs))
            sample_indices.append(torch.ones(len(sample_roi)).type_as(rpn_locs) * roi_indices[i][0])
            gt_roi_locs.append(torch.Tensor(gt_roi_loc).type_as(rpn_locs))
            gt_roi_label.append(torch.Tensor(gt_roi_label).type_as(rpn_locs).long())

        # n_sample = 128
        sample_rois = torch.stack(sample_rois, dim=0)   # (n * 128, 4)
        sample_indices = torch.stack(sample_indices, dim=0)     # (n * 128, 1)
        # (n, 128, 4 * n_classes), (n, 128, n_classes)
        roi_cls_locs, roi_scores = self.model_train([base_feature, sample_rois, sample_indices, img_size], mode='head')

        for i in range(n):
            n_sample = roi_cls_locs.size()[1]

            roi_cls_loc = roi_cls_locs[i]   # (128, 4 * n_classes)
            roi_score = roi_scores[i]   # (128, n_classes)
            gt_roi_loc = gt_roi_locs[i] # (128, 4)
            gt_roi_label = gt_roi_labels[i] # (128, 1)

            roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
            roi_loc = roi_cls_loc[torch.arange(0, n_sample), gt_roi_label] # (128, 4) 为每一个样本框选取对应的标签的调整参数

            roi_loc_loss = self._fast_rcnn_loc_loss(roi_loc, gt_roi_loc, gt_roi_label.data, self.roi_sigma)
            roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label)

            roi_cls_loss_all += roi_cls_loss
            roi_loc_loss_all += roi_loc_loss

        losses = [rpn_loc_loss_all/n, rpn_cls_loss_all/n, roi_loc_loss_all/n, roi_cls_loss_all/n]
        losses = losses + [sum(losses)] # (5, 1)
        return losses

    def train_step(self, imgs, bboxes, labels, scale, fp16=False, scaler=None):
        """

        :param imgs: (n, h, w)
        :param bboxes: (n, n_t, 4)
        :param labels: (n, n_t, 1)
        :param scale:
        :param fp16:
        :param scaler:
        :return:
        """
        self.optimizer.zero_grad()
        if not fp16:
            losses = self.forward(imgs, bboxes, labels, scale)
            losses[-1].backward()
            self.optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                losses = self.forward(imgs, bboxes, labels, scale)

                # ??????? scaler ????????
                scaler.scale(losses[-1]).backward()
                scaler.step(self.optimizer)
                scaler.update()

        return losses


def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize net with %s type' % init_type)
    net.apply(init_func)


def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1, no_aug_iter_ratio=0.05, step_num=10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi * (iters -warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n = iters // step_size
        lr = lr * decay_rate ** n
        return  lr

    if lr_decay_type == 'cos':
        warmup_total_iters = min(max((warmup_iters_ratio * total_iters),1), 3)
        warmup_lr_start = max(lr * warmup_lr_ratio, 1e-6)
        no_aug_iter = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr, lr, min_lr, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate = (min_lr / lr) ** (1 / (step_num-1))
        step_size = total_iters / step_num
        func = partial(step_lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_schedule_func, epoch):
    lr = lr_schedule_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr