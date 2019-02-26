import numpy as np
import torch

from ..util.box import box_transform_inv, clip_boxes
from .generate_anchors import generate_anchors
from ..nms.nms_wrapper import nms
from config.config import config as cfg


def get_proposals(rpn_cls_prob, rpn_box_reg, cfg_key, im_info, anchor_scales, anchor_ratios, feat_stride):
    """
    Generate proposals by using rpn output prediction and pre-defined anchors.

    Arguments:

    rpn_cls_prob -- tensor of shape (B, 2 * num_anchors, H, W)
    rpn_box_reg -- tensor of shape (B, 4 * num_anchors, H, W)
    cfg_key -- string, 'TRAIN' or 'TEST'.
    im_info -- tensor of shape (B, 3) [image_height, image_width, scale_ratio]
    anchor_scales -- list. scales to the basic window [16, 16]
    anchor_ratios -- list. ratios of anchors
    feat_stride -- list. down-sampling ratio of feature map to the original input image.

    Returns:

    rpn_rois: tensor of shape (N, 5). [batch_id, x1, y1, x2, y2]
    """

    # Algorithm:
    # generate all anchors over the grid
    # apply delta to anchors to generate boxes (proposals)
    # clip proposals
    # filter proposals with either height or width < min_size
    # take top pre_nms_topN (objectiveness score) proposals before NMS
    # apply NMS
    # take top post_nms_topN proposals after NMS
    # return those top proposals

    bsize, _, h, w = rpn_cls_prob.size()
    A = len(anchor_scales) * len(anchor_ratios)
    K = h * w

    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
    min_size = cfg[cfg_key].RPN_MIN_SIZE

    assert bsize == 1, 'Only support single batch'

    im_info = im_info[0]

    # enumerate all shifts

    shift_x = np.arange(0, w) * feat_stride
    shift_y = np.arange(0, h) * feat_stride
    shifts_x, shifts_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack([shifts_x.ravel(), shifts_y.ravel(),
                        shifts_x.ravel(), shifts_y.ravel()]).transpose()

    # apply shifts to base anchors

    anchor = generate_anchors(base_size=feat_stride[0], ratios=anchor_ratios, scales=anchor_scales)

    anchors = anchor.reshape(1, A, 4) + shifts.reshape(K, 1, 4)

    anchors = anchors.reshape(-1, 4)

    assert anchors.shape[0] == K * A

    # apply deltas to generate proposals

    anchors = torch.from_numpy(anchors).type_as(rpn_cls_prob)

    if cfg.DEBUG:
        _checkwidth = 25
        _checkheight = 19
        _start = _checkheight * w * 9 + _checkwidth * 9
        _end = _start + 9
        print(anchors[_start:_end, :])

    rpn_box_reg = rpn_box_reg.permute(0, 2, 3, 1).contiguous().view(-1, 4)

    proposals = box_transform_inv(anchors, rpn_box_reg)

    # clip and filter proposals

    proposals = clip_boxes(proposals, im_info)

    keep = _filter_boxes(proposals, min_size)

    proposals_keep = proposals[keep, :]

    # same for cls_prob
    # only choose prob for objectiveness (fg)
    rpn_scores = rpn_cls_prob[:, A:, :, :]
    rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(-1) # N

    rpn_scores_keep = rpn_scores[keep]

    # take top pre_nms_topN before nms

    sort_ind = torch.sort(rpn_scores_keep, descending=True)[1]

    top_keep = sort_ind[:pre_nms_topN]

    proposals_keep = proposals_keep[top_keep, :]
    rpn_scores_keep = rpn_scores_keep[top_keep]

    # ##
    # print(proposals_keep[:30, :])
    # print(rpn_scores_keep[:30])
    # ##
    #
    # __tmp_keep = torch.from_numpy(np.array([ 7458,  7467,  7908,  7917,  8358,  8367,  8808,  8817,  9258,  9267,
    #      9708,  9717, 10158, 10167, 10590, 10608, 10617, 11040, 11049, 11490,
    #     11967, 12417, 12867, 13083, 13533, 13683, 13983, 14133, 14583])).long()
    # __tmp = anchors[__tmp_keep, :]
    #
    # __tmp_score = rpn_scores[__tmp_keep]
    #
    # print(__tmp)
    # print(__tmp_score)
    #
    # # apply nms with threshold 0.7

    keep = nms(torch.cat([proposals_keep, rpn_scores_keep.view(-1, 1)], dim=1), thresh=nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
    keep = keep.long().view(-1)

    # pick post_nms_topN

    keep = keep[:post_nms_topN]

    proposals_keep = proposals_keep[keep, :]

    rpn_scores_keep = rpn_scores_keep[keep]

    # ### tmp
    # print(proposals_keep[:30, :])
    # print(rpn_scores_keep[:30])

    rois = proposals_keep.new_zeros((proposals_keep.size(0),5))

    rois[:, 1:] = proposals_keep

    return rois


def _filter_boxes(boxes, min_size):
    """
    Arguments:

    boxes -- tensor of shape (N, 4) (x1, y1, x2, y2)
    min_size -- int

    Returns:
    keep -- tensor of shape (N) uint8, byte tensor.
    """

    keep = ((boxes[:, 2] - boxes[:, 0] + 1) >= min_size) & \
           ((boxes[:, 3] - boxes[:, 1] + 1) >= min_size)

    return keep












