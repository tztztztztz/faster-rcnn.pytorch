import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import config as cfg
from faster_rcnn.rpn.proposal_target import build_proposal_target
from faster_rcnn.resnet import resnet50, resnet101
from faster_rcnn.rpn.rpn import RPN
from faster_rcnn.roi_pooling.modules.roi_pool import _RoIPooling
from faster_rcnn.util.network import _smooth_l1_loss, normal_init


class FasterRCNN(nn.Module):
    n_classes = 21 # extra one for __background__

    def __init__(self, backbone='res50', pretrained=None):
        super(FasterRCNN, self).__init__()

        if backbone == 'res50':
            resnet = resnet50()
        elif backbone == 'res101':
            resnet = resnet101()
        else:
            raise NotImplementedError

        if pretrained:
            print('Loading pretrained weights from %s' % pretrained)
            state_dict = torch.load(pretrained)
            resnet.load_state_dict({k: v for k, v in state_dict.items() if k in resnet.state_dict()})

        self.backbone = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
                                      resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

        self.classifier = nn.Sequential(resnet.layer4)

        self.n_feature = 1024
        self.spatial_scale = 1. / 16
        self.pooled_height = cfg.POOLING_SIZE
        self.pooled_width = cfg.POOLING_SIZE

        self.rpn = RPN(self.n_feature)
        self.roi_pooling = _RoIPooling(self.pooled_height, self.pooled_width, self.spatial_scale)
        self.rcnn_cls = nn.Linear(2048, self.n_classes)
        self.rcnn_box = nn.Linear(2048, self.n_classes * 4)

        # Fix blocks
        for p in self.backbone[0].parameters(): p.requires_grad = False
        for p in self.backbone[1].parameters(): p.requires_grad = False
        for p in self.backbone[4].parameters(): p.requires_grad = False

        def set_bn_fix(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in m.parameters(): p.requires_grad = False

        self.backbone.apply(set_bn_fix)
        self.classifier.apply(set_bn_fix)

        # initial weights
        normal_init(self.rpn.rpn_conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn.rpn_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rpn.rpn_box_reg, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rcnn_cls, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.rcnn_box, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def forward(self, x, gt_boxes, im_info):
        features = self.backbone(x)
        rois, rpn_cls_loss, rpn_box_loss, _rpn_train_info = self.rpn(features, gt_boxes, im_info)

        # ## TMP DEBUG
        # return rois, rpn_cls_loss, rpn_box_loss, _rpn_train_info

        if self.training:
            rois_data = build_proposal_target(rois, gt_boxes, im_info, self.n_classes)
            rois, label_target, boxes_target, boxes_inside_weight, boxes_outside_weight = rois_data
        else:
            label_target = None
            boxes_target = None
            boxes_inside_weight = None
            boxes_outside_weight = None

        pooled_feature = self.roi_pooling(features, rois) # N x n_feature x 7 x 7
        out = self.classifier(pooled_feature)

        # avg pooling
        out = out.mean(3).mean(2)

        rcnn_cls = self.rcnn_cls(out)
        rcnn_box = self.rcnn_box(out)

        rcnn_cls_prob = F.softmax(rcnn_cls, 1)

        rcnn_cls_loss = 0
        rcnn_box_loss = 0
        _train_info = {}

        if self.training:
            # build loss
            rcnn_cls_loss = F.cross_entropy(rcnn_cls, label_target)
            rcnn_box_loss = _smooth_l1_loss(rcnn_box, boxes_target, boxes_inside_weight, boxes_outside_weight)

            if cfg.VERBOSE:
                _rcnn_fg_inds = torch.nonzero(label_target != 0).view(-1)
                _rcnn_bg_inds = torch.nonzero(label_target == 0).view(-1)
                _rcnn_pred_info = torch.argmax(rcnn_cls, dim=1)
                rcnn_tp = torch.sum(_rcnn_pred_info[_rcnn_fg_inds] == label_target[_rcnn_fg_inds])
                rcnn_tn = torch.sum(_rcnn_pred_info[_rcnn_bg_inds] == label_target[_rcnn_bg_inds])
                _train_info['rcnn_num_fg'] = _rcnn_fg_inds.size(0)
                _train_info['rcnn_num_bg'] = _rcnn_bg_inds.size(0)
                _train_info['rcnn_tp'] = rcnn_tp.item()
                _train_info['rcnn_tn'] = rcnn_tn.item()
                _train_info['rpn_num_fg'] = _rpn_train_info['rpn_num_fg']
                _train_info['rpn_num_bg'] = _rpn_train_info['rpn_num_bg']
                _train_info['rpn_tp'] = _rpn_train_info['rpn_tp']
                _train_info['rpn_tn'] = _rpn_train_info['rpn_tn']

        return rois, rcnn_cls_prob, rcnn_box, rcnn_cls_loss, rcnn_box_loss, rpn_cls_loss, rpn_box_loss, _train_info

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode:
            self.backbone.eval()
            self.backbone[5].train()
            self.backbone[6].train()

            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.backbone.apply(set_bn_eval)
            self.classifier.apply(set_bn_eval)















