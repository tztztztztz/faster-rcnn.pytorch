import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .anchor_target import build_anchor_target
from .proposal import get_proposals
from ..util.network import _smooth_l1_loss


from config.config import config as cfg


class RPN(nn.Module):
    def __init__(self, n_feature):
        super(RPN, self).__init__()

        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE
        self.n_anchors = len(self.anchor_ratios) * len(self.anchor_scales)

        self.n_feature = n_feature
        self.rpn_conv = nn.Conv2d(n_feature, 512, 3, 1, 1)

        self.rpn_cls_score = nn.Conv2d(512, self.n_anchors * 2, 1, 1, 0)
        self.rpn_box_reg = nn.Conv2d(512, self.n_anchors * 4, 1, 1, 0)

    def forward(self, features, gt_boxes, im_info):
        bsize, _, h, w = features.size()
        rpn_feat = F.relu(self.rpn_conv(features), inplace=True)

        # rpn score
        rpn_cls_score = self.rpn_cls_score(rpn_feat)
        rpn_cls_score_reshape = rpn_cls_score.view(bsize, 2, self.n_anchors, h, w)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob = rpn_cls_prob_reshape.view(bsize, self.n_anchors * 2, h, w)

        # rpn boxes
        rpn_box_reg = self.rpn_box_reg(rpn_feat)

        cfg_key = 'TRAIN' if self.training else 'TEST'

        rois = get_proposals(rpn_cls_prob.data, rpn_box_reg.data, cfg_key, im_info,
                             self.anchor_scales, self.anchor_ratios, self.feat_stride)


        rpn_cls_loss = 0
        rpn_box_loss = 0
        _train_info = {}

        if self.training:
            rpn_data = build_anchor_target(rpn_cls_prob.data, gt_boxes, im_info,
                                           self.anchor_scales, self.anchor_ratios, self.feat_stride)

            label_target, boxes_target, outside_boxes_weights, inside_boxes_weights = rpn_data

            # cls cross entropy loss
            keep = torch.nonzero(label_target != -1).view(-1)
            label_target_keep = label_target[keep]
            keep = Variable(keep)
            label_target_keep = Variable(label_target_keep).long()
            rpn_cls_score = rpn_cls_score_reshape.permute(0, 3, 4, 2, 1).contiguous().view(-1, 2)
            rpn_cls_score_keep = rpn_cls_score[keep, :]
            rpn_cls_loss = F.cross_entropy(rpn_cls_score_keep, label_target_keep)

            # box smooth l1 loss
            boxes_target = Variable(boxes_target.view(bsize, h, w, self.n_anchors * 4).permute(0, 3, 1, 2))
            inside_boxes_weights = Variable(inside_boxes_weights.view(bsize, h, w, self.n_anchors * 4).permute(0, 3, 1, 2))
            outside_boxes_weights = Variable(outside_boxes_weights.view(bsize, h, w, self.n_anchors * 4).permute(0, 3, 1, 2))

            # build rpn loss
            rpn_box_loss = _smooth_l1_loss(rpn_box_reg, boxes_target, inside_boxes_weights, outside_boxes_weights, dim=[1,2,3])

            # collect training info
            if cfg.VERBOSE:
                rpn_fg_inds = torch.nonzero(label_target == 1).view(-1)
                rpn_bg_inds = torch.nonzero(label_target == 0).view(-1)
                rpn_pred_data = rpn_cls_prob_reshape.permute(0, 3, 4, 2, 1).contiguous().view(-1, 2)[:, 1]
                rpn_pred_info = (rpn_pred_data >= 0.4).long()
                rpn_tp = torch.sum(rpn_pred_info[rpn_fg_inds] == label_target[rpn_fg_inds].long())
                rpn_tn = torch.sum(rpn_pred_info[rpn_bg_inds] == label_target[rpn_bg_inds].long())
                _train_info['rpn_num_fg'] = rpn_fg_inds.size(0)
                _train_info['rpn_num_bg'] = rpn_bg_inds.size(0)
                _train_info['rpn_tp'] = rpn_tp.item()
                _train_info['rpn_tn'] = rpn_tn.item()

        return rois, rpn_cls_loss, rpn_box_loss, _train_info













