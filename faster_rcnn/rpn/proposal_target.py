import numpy as np
import torch
from config.config import config as cfg
from ..util.box import bbox_overlaps, box_transform


def build_proposal_target(all_rois, gt_boxes, im_info, n_classes):
    """
    Arguments:
    all_rois -- tensor of shape (N, 5) (0, x1, y1, x2, y2)
    gt_boxes -- tensor of shape (B, G, 5)
    im_info --  tensor of shape (B, 3) [image_height, image_width, scale_ratio]

    Returns:
    rois -- tensor of shape (S, 5)
    label_target -- tensor of shape (S)
    boxes_target -- tensor of shape (S, 4 * n_classes)
    boxes_inside_weights -- tensor of shape (S, 4 * n_classes)
    boxes_outside_weights -- tensor of shape (S, 4 * n_classes)
    """

    # Algorithm
    # merger gt boxes to all_rois
    # calculate the overlap between all rois and gt_boxes
    # sampling rois
    #     # define positive and negative rois
    #     # sample positive and negative rois if there are too many
    # assign label target
    # compute boxes target
    # assign inside_boxes_weights

    bsize, num_boxes, _ = gt_boxes.size()

    assert bsize == 1, 'Only support single batch'

    gt_boxes = gt_boxes[0]
    im_info = im_info[0]

    bbox_normalize_means = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).type_as(all_rois)
    bbox_normalize_stds = torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).type_as(all_rois)
    inside_weights = torch.FloatTensor(cfg.TRAIN.BBOX_INSIDE_WEIGHTS).type_as(all_rois)

    gt_boxes_append = all_rois.new_zeros((num_boxes, 5))
    gt_boxes_append[:, 1:] = gt_boxes[:, :4]

    merged_rois = torch.cat([all_rois, gt_boxes_append], dim=0)

    rois_per_image = cfg.TRAIN.BATCH_SIZE

    fg_rois_per_image = int(rois_per_image * cfg.TRAIN.FG_FRACTION)

    # sample rois

    overlaps = bbox_overlaps(merged_rois[:, 1:], gt_boxes[:, :4])

    max_overlap, argmax_overlap = torch.max(overlaps, 1)

    fg_inds = torch.nonzero(max_overlap >= cfg.TRAIN.FG_THRESH).view(-1)
    bg_inds = torch.nonzero((max_overlap >= cfg.TRAIN.BG_THRESH_LO) &
                            (max_overlap < cfg.TRAIN.BG_THRESH_HI)).view(-1)

    num_fg = fg_inds.numel()
    num_bg = bg_inds.numel()

    # num_fg always > 0
    if num_bg > 0:
        num_fg_this_image = min(num_fg, fg_rois_per_image)
        rand_num = torch.from_numpy(np.random.permutation(num_fg_this_image)).type_as(all_rois).long()
        fg_inds = fg_inds[rand_num[:num_fg_this_image]]

        num_bg_this_image = rois_per_image - num_fg_this_image

        rand_num = np.floor(np.random.rand(num_bg_this_image) * num_bg)
        rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
        bg_inds = bg_inds[rand_num]
    else:
        rand_num = np.floor(np.random.rand(rois_per_image) * num_fg)
        rand_num = torch.from_numpy(rand_num).type_as(gt_boxes).long()
        fg_inds = fg_inds[rand_num]
        num_fg_this_image = rois_per_image
        num_bg_this_image = 0

    keep_inds = torch.cat([fg_inds, bg_inds], dim=0)

    # assign label target
    rois = merged_rois[keep_inds, :]
    if cfg.DEBUG:
        print(merged_rois[fg_inds])
    label_target = gt_boxes[argmax_overlap[keep_inds], 4].long()
    if num_bg_this_image > 0:
        label_target[num_fg_this_image:] = 0

    # Compute box target
    boxes_target_data = box_transform(rois[:, 1:], gt_boxes[argmax_overlap[keep_inds], :4])

    # Normalize regression target
    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS:
        boxes_target_data = (boxes_target_data - bbox_normalize_means.view(1, 4)) \
                       / bbox_normalize_stds.view(1, 4)  # use broad-casting (1, 4) -> (N, 4)

    # make boxes target data class specific
    boxes_target = all_rois.new_zeros(rois.size(0), n_classes * 4)
    boxes_inside_weight = all_rois.new_zeros(rois.size(0), n_classes * 4)

    for i in range(label_target.size(0)):
        cls_id = int(label_target[i].item())
        if cls_id == 0:
            continue
        start = cls_id * 4
        end = start + 4
        boxes_target[i, start:end] = boxes_target_data[i, :]
        boxes_inside_weight[i, start:end] = inside_weights

    boxes_outside_weights = (boxes_inside_weight > 0).float()

    return rois, label_target, boxes_target, boxes_inside_weight, boxes_outside_weights




















































