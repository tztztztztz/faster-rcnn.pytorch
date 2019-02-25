import numpy as np
import torch
from config.config import config as cfg
from ..util.box import bbox_overlaps, box_transform
from .generate_anchors import generate_anchors


def build_anchor_target(rpn_cls_prob, gt_boxes, im_info,
                        anchor_scales, anchor_ratios, feat_stride):
    """
    Assign anchors to ground-truth targets. Produce anchor classification labels and
    bounding-box regression targets.

    Arguments:

    rpn_cls_prob -- tensor of shape (B, 2*A, H, W)
    gt_boxes -- tensor of shape (B, N, 5) (x1, y1, x2, y2, cls)
    im_info --  tensor of shape (B, 3) [image_height, image_width, scale_ratio]
    cfg_key -- string of 'TRAIN' or 'TEST'
    anchor_scale -- list
    anchor_ratios -- list
    feat_stride -- list down-sampling ratio [16, ]

    Returns:

    label_target -- tensor of shape (B * H * W * A)
    boxes_target -- tensor of shape (B * H * W * A, 4)
    inside_boxes_weights -- tensor of shape (B * H * W * A, 4)
    outside_boxes_weights -- tensor of shape (B * H * W * A, 4)

    """

    # Algorithms:
    # generate all anchors over the grid
    # exclude anchors those are out of bound
    # calculate the overlap between anchors and ground-truth boxes
    # define positive anchors and negative anchors
    #   # each ground truth each match an anchor with max overlap
    #   # assign anchor with overlap larger than thresh to positive, less than thresh to negative
    #   # assign label
    # sampling positive and negative anchors
    #   # disable some samples, and set the related label to -1
    # assign box regression target

    bsize, _, h, w = rpn_cls_prob.size()

    A = len(anchor_scales) * len(anchor_ratios)

    K = h * w

    assert bsize == 1, 'Only support single batch'

    gt_boxes = gt_boxes[0]
    im_info = im_info[0]

    num_boxes = gt_boxes.size(0)

    image_height, image_width = im_info[0], im_info[1]

    assert bsize == 1, 'only single batch is supported'

    anchor = generate_anchors(feat_stride[0], ratios=anchor_ratios, scales=anchor_scales) # (A, 4)

    shift_x = np.arange(0, w) * feat_stride
    shift_y = np.arange(0, h) * feat_stride
    shifts_x, shifts_y = np.meshgrid(shift_x, shift_y)

    all_shifts = np.vstack((shifts_x.ravel(), shifts_y.ravel(),
                            shifts_x.ravel(), shifts_y.ravel())).transpose() # (K, 4)

    anchors = anchor.reshape(1, A, 4) + all_shifts.reshape(K, 1, 4) # (K, A, 4)
    anchors = anchors.reshape(-1, 4) # (K * A, 4)
    num_all_anchors = anchors.shape[0]

    # to torch tensor
    anchors = torch.from_numpy(anchors).type_as(rpn_cls_prob)

    inside_keep = (anchors[:, 0] >= 0) & \
                  (anchors[:, 1] >= 0) & \
                  (anchors[:, 2] <= image_width - 1) & \
                  (anchors[:, 3] <= image_height - 1)

    inside_inds = torch.nonzero(inside_keep).view(-1)

    num_inside_anchors = inside_inds.size(0)

    inside_anchors = anchors[inside_inds, :]

    if cfg.DEBUG:
        print(inside_anchors[:10, :])

    label_target = rpn_cls_prob.new(num_inside_anchors).fill_(-1)

    overlaps = bbox_overlaps(inside_anchors, gt_boxes[:, :4]) # N X G

    max_overlap, argmax_overlap = torch.max(overlaps, 1) # N
    gt_max_overlaps, _ = torch.max(overlaps, 0) # G

    # assign each gt box to one max overlap anchor
    # gt_argmax_overlaps, super trick! because there may be two anchors having the same IOU

    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        label_target[max_overlap < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    keep = torch.sum(gt_max_overlaps.view(1, -1).expand(num_inside_anchors, num_boxes) == overlaps, 1)

    if torch.sum(keep) > 0:
        if cfg.DEBUG:
            _keep_ind = torch.nonzero(keep > 0).view(-1)
            _keep_anchors = inside_anchors[_keep_ind, :]
            print(_keep_anchors)
            print(bbox_overlaps(_keep_anchors, gt_boxes))
        label_target[keep > 0] = 1

    label_target[max_overlap > cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        label_target[max_overlap < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    max_num_fg = int(cfg.TRAIN.RPN_BATCHSIZE * cfg.TRAIN.RPN_FG_FRACTION)

    num_fg = torch.sum(label_target == 1)
    num_bg = torch.sum(label_target == 0)

    # sample front and background if there are too many.
    if num_fg > max_num_fg:
        fg_inds = torch.nonzero(label_target == 1).view(-1)
        rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(fg_inds)
        disable_inds = fg_inds[rand_num[:(num_fg - max_num_fg)]]
        label_target[disable_inds] = -1

    max_num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum(label_target == 1)

    if num_bg > max_num_bg:
        bg_inds = torch.nonzero(label_target == 0).view(-1)
        rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(bg_inds)
        disable_inds = bg_inds[rand_num[:(num_bg - max_num_bg)]]
        label_target[disable_inds] = -1

    # Compute boxes target of all anchors
    boxes_target = box_transform(inside_anchors, gt_boxes[argmax_overlap, :])

    if cfg.DEBUG:
        __num_check = 20
        __fg_inds = torch.nonzero(label_target == 1).view(-1)
        __fg_anchors = inside_anchors[__fg_inds[__num_check:]]
        __fg_gt_boxes = gt_boxes[argmax_overlap, :][__fg_inds[__num_check:]]
        __fg_boxes_target = box_transform(__fg_anchors, __fg_gt_boxes)
        print(__fg_anchors)
        print(__fg_gt_boxes)
        print(__fg_boxes_target)

    inside_boxes_weights = rpn_cls_prob.new_zeros(num_inside_anchors, 4)
    outside_boxes_weights = rpn_cls_prob.new_zeros(num_inside_anchors, 1)

    # assign

    assert cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0

    num_examples = torch.sum(label_target >= 0)

    outside_weights = 1.0 / num_examples.float()

    inside_weights = torch.from_numpy(np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)).type_as(rpn_cls_prob)

    outside_boxes_weights[label_target >= 0, :] = outside_weights
    outside_boxes_weights = outside_boxes_weights.expand(num_inside_anchors, 4)
    inside_boxes_weights[label_target == 1, :] = inside_weights

    label_target = _unmap(label_target, num_all_anchors, inside_inds, fill=-1)
    boxes_target = _unmap(boxes_target, num_all_anchors, inside_inds, fill=0)
    outside_boxes_weights = _unmap(outside_boxes_weights, num_all_anchors, inside_inds, fill=0)
    inside_boxes_weights = _unmap(inside_boxes_weights, num_all_anchors, inside_inds, fill=0)

    return label_target, boxes_target, outside_boxes_weights, inside_boxes_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if data.dim() == 1:
        ret = data.new(count).fill_(fill)
        ret[inds] = data

    elif data.dim() == 2:
        ret = data.new(count, data.size(1)).fill_(fill)
        ret[inds, :] = data

    else:
        raise NotImplementedError

    return ret



















