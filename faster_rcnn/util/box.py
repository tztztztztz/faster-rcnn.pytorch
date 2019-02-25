import torch

def bbox_overlaps(box1, box2):
    """
    Implement the intersection over union (IoU) between box1 and box2 (x1, y1, x2, y2)

    Arguments:
    box1 -- tensor of shape (N, 4), first set of boxes
    box2 -- tensor of shape (K, 4), second set of boxes

    Returns:
    ious -- tensor of shape (N, K), ious between boxes
    """

    N = box1.size(0)
    K = box2.size(0)

    # when torch.max() takes tensor of different shape as arguments, it will broadcasting them.
    xi1 = torch.max(box1[:, 0].view(N, 1), box2[:, 0].view(1, K))
    yi1 = torch.max(box1[:, 1].view(N, 1), box2[:, 1].view(1, K))
    xi2 = torch.min(box1[:, 2].view(N, 1), box2[:, 2].view(1, K))
    yi2 = torch.min(box1[:, 3].view(N, 1), box2[:, 3].view(1, K))

    # we want to compare the compare the value with 0 elementwise. However, we can't
    # simply feed int 0, because it will invoke the function torch(max, dim=int) which is not
    # what we want.
    # To feed a tensor 0 of same type and device with box1 and box2
    # we use tensor.new().fill_(0)

    iw = torch.max(xi2 - xi1, box1.new(1).fill_(0))
    ih = torch.max(yi2 - yi1, box1.new(1).fill_(0))

    inter = iw * ih

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    box1_area = box1_area.view(N, 1)
    box2_area = box2_area.view(1, K)

    union_area = box1_area + box2_area - inter

    ious = inter / union_area

    return ious


def xxyy2xywh(box):
    """
    Convert the box (x1, y1, x2, y2) encoding format to (c_x, c_y, w, h) format

    Arguments:
    box: tensor of shape (N, 4), boxes of (x1, y1, x2, y2) format

    Returns:
    xywh_box: tensor of shape (N, 4), boxes of (c_x, c_y, w, h) format
    """

    c_x = (box[:, 2] + box[:, 0]) / 2
    c_y = (box[:, 3] + box[:, 1]) / 2
    w = box[:, 2] - box[:, 0] + 1
    h = box[:, 3] - box[:, 1] + 1

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    xywh_box = torch.cat([c_x, c_y, w, h], dim=1)
    return xywh_box


def xywh2xxyy(box):
    """
    Convert the box encoding format form (c_x, c_y, w, h) to (x1, y1, x2, y2)

    Arguments:
    box -- tensor of shape (N, 4), box of (c_x, c_y, w, h) format

    Returns:
    xxyy_box -- tensor of shape (N, 4), box of (x1, y1, x2, y2) format
    """

    x1 = box[:, 0] - (box[:, 2] - 1) / 2
    y1 = box[:, 1] - (box[:, 3] - 1) / 2
    x2 = box[:, 0] + (box[:, 2] - 1) / 2
    y2 = box[:, 1] + (box[:, 3] - 1) / 2

    x1 = x1.view(-1, 1)
    y1 = y1.view(-1, 1)
    x2 = x2.view(-1, 1)
    y2 = y2.view(-1, 1)

    xxyy_box = torch.cat([x1, y1, x2, y2], dim=1)
    return xxyy_box


def box_transform(anchors, gt_boxes):
    """

    Compute the deltas from anchors to gt_boxes

    Arguments:
    anchors -- tensor of shape (N, 4)
    gt_boxes -- tensor of shape (N, 4)

    Return:
    deltas -- tensor of shape (N, 4)
    """

    anchors_xywh = xxyy2xywh(anchors)
    gt_boxes_xywh = xxyy2xywh(gt_boxes)

    t_x = (gt_boxes_xywh[:, 0] - anchors_xywh[:, 0]) / anchors_xywh[:, 2]
    t_y = (gt_boxes_xywh[:, 1] - anchors_xywh[:, 1]) / anchors_xywh[:, 3]
    t_w = torch.log(gt_boxes_xywh[:, 2] / anchors_xywh[:, 2])
    t_h = torch.log(gt_boxes_xywh[:, 3] / anchors_xywh[:, 3])

    t_x = t_x.view(-1, 1)
    t_y = t_y.view(-1, 1)
    t_w = t_w.view(-1, 1)
    t_h = t_h.view(-1, 1)

    deltas = torch.cat([t_x, t_y, t_w, t_h], dim=1)

    return deltas


def box_transform_inv(anchors, deltas):
    """
    Convert anchors to predicted boxes

    Arguments:
    anchors -- tensor of shape (N, 4) (x1, y1, x2, y2)
    deltas -- tensor of shape (N, 4) (t_x, t_y, t_w, t_h)

    Returns:
    pred_boxes -- shape of shape (N, 4) (x1, y1, x2, y2)
    """

    anchors_xywh = xxyy2xywh(anchors)
    c_x = deltas[:, 0] * anchors_xywh[:, 2] + anchors_xywh[:, 0]
    c_y = deltas[:, 1] * anchors_xywh[:, 3] + anchors_xywh[:, 1]
    w = torch.exp(deltas[:, 2]) * anchors_xywh[:, 2]
    h = torch.exp(deltas[:, 3]) * anchors_xywh[:, 3]

    c_x = c_x.view(-1, 1)
    c_y = c_y.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    pred_boxes = torch.cat([c_x, c_y, w, h], dim=1)

    return xywh2xxyy(pred_boxes)


def clip_boxes(boxes, im_info):
    """
    Arguments:

    boxes -- tensor of shape (N, 4) (x1, y1, x2, y2)
    im_info -- list of [image_height, image_width, scale_ratio]

    Returns:

    boxes -- tensor of shape (None, 4)
    """

    height, width = im_info[0], im_info[1]

    boxes[:, 0::2].clamp_(0, width - 1)
    boxes[:, 1::2].clamp_(0, height - 1)

    return boxes



