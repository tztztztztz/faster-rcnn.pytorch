import os
import sys
import argparse
import time
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset.roidb import combined_roidb, RoiDataset
from config.config import config as cfg
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.util.box import bbox_transform_inv_cls, clip_boxes_cls
from faster_rcnn.nms.nms_wrapper import nms
from util.visualize import draw_detection_boxes


def parse_args():

    parser = argparse.ArgumentParser('Faster-RCNN test')

    parser.add_argument('--cuda', dest='use_cuda',
                        action='store_true', default=None)

    parser.add_argument('--net', dest='net',
                        default='res50', type=str)

    parser.add_argument('--vis', dest='vis',
                        action='store_true', default=None)

    parser.add_argument('--dataset', dest='dataset',
                        default='voc07test', type=str)

    parser.add_argument('--check_epoch', dest='check_epoch',
                        default='10', type=str)

    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)

    args = parser.parse_args()
    return args


def test():

    args = parse_args()

    print('Call with ', args)

    model_name = 'faster_{}_epoch_{}.pth'.format(args.net, args.check_epoch)
    model_path = os.path.join(args.output_dir, model_name)

    # load dataset

    if args.dataset == 'voc07test':
        args.imdbval_name = 'voc_2007_test'
    elif args.dataset == 'voc07trainval':
        args.imdbval_name = 'voc_2007_trainval'
    else:
        raise NotImplementedError

    cfg.TRAIN.USE_FLIPPED = False

    imdb, roidb = combined_roidb(args.imdbval_name)

    val_dataset = RoiDataset(roidb)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    data_iter = iter(val_dataloader)

    print('load {}, roidb len is {}'.format(args.imdbval_name, len(roidb)))

    # load model

    model = FasterRCNN(backbone=args.net)

    model.load_state_dict(torch.load(model_path)['model'])

    print('model loaded from {}'.format(model_path))

    if args.use_cuda:
        model = model.cuda()

    if args.vis:
        thresh = 0.05
    else:
        thresh = 0.0

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    start = time.time()
    max_per_image = 100
    det_file = os.path.join(args.output_dir, 'detections.pkl')

    num_images = len(imdb.image_index)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(imdb.num_classes)]

    model.eval()

    empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

    torch.set_grad_enabled(False)

    for i in range(num_images):
        im_data, gt_boxes, im_info = next(data_iter)
        if args.use_cuda:
            im_data = im_data.cuda()
            gt_boxes = gt_boxes.cuda()
            im_info = im_info.cuda()

        im_data_variable = Variable(im_data)

        det_tic = time.time()
        output = model(im_data_variable, gt_boxes, im_info)
        rois, cls_prob, bbox_pred = output[:3]

        scores = cls_prob.data
        boxes = rois.data[:, 1:]

        box_deltas = bbox_pred.data

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(-1, 4 * imdb.num_classes)

        pred_boxes = bbox_transform_inv_cls(boxes, box_deltas)

        pred_boxes = clip_boxes_cls(pred_boxes, im_info[0])

        pred_boxes /= im_info[0][2].item()

        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()

        if args.vis:
            im2show = Image.open(imdb.image_path_at(i))

        for j in range(1, imdb.num_classes):
            inds = torch.nonzero(scores[:, j] > thresh).view(-1)
            if inds.numel() > 0:
                cls_scores = scores[:, j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if args.vis:
                    clsname_dets = np.repeat(j, cls_dets.size(0))
                    im2show = draw_detection_boxes(im2show, cls_dets.cpu().numpy(), clsname_dets, imdb.classes, 0.5)
                all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                all_boxes[j][i] = empty_array

        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in range(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        if args.vis:
            plt.imshow(im2show)
            plt.show()

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic
        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                         .format(i + 1, num_images, detect_time, nms_time))
        sys.stdout.flush()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    imdb.evaluate_detections(all_boxes, args.output_dir)

    end = time.time()
    print("test time: %0.4fs" % (end - start))


if __name__ == '__main__':
    test()

















