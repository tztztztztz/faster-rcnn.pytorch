from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import SGD
from tensorboardX import SummaryWriter
from config.config import config as cfg
import pickle

from faster_rcnn.faster_rcnn import FasterRCNN
from dataset.roidb import combined_roidb, RoiDataset


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def parse_args():
    parser = argparse.ArgumentParser(description='faster rcnn')
    parser.add_argument('--dataset', dest='dataset',
                        default='voc07trainval', type=str)
    parser.add_argument('--batch_size', dest='batch_size',
                        default=1, type=int)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=None, action='store_true')
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--max_epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=7, type=int)
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        default=True, type=bool)
    parser.add_argument('--output_dir', dest='output_dir',
                        default='output', type=str)
    parser.add_argument('--exp_name', dest='exp_name',
                        default='default', type=str)
    parser.add_argument('--display_interval', dest='display_interval',
                        default=100, type=int)
    parser.add_argument('--save_interval', dest='save_interval',
                        default=1, type=int)
    args = parser.parse_args()
    return args


def train():
    args = parse_args()
    args.pretrained_model = os.path.join('data', 'pretrained', 'resnet50-caffe.pth')
    args.decay_lrs = cfg.TRAIN.DECAY_LRS

    cfg.USE_GPU_NMS = True if args.use_cuda else False

    assert args.batch_size == 1, 'Only support single batch'

    lr = cfg.TRAIN.LEARNING_RATE
    momentum = cfg.TRAIN.MOMENTUM
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    gamma = cfg.TRAIN.GAMMA

    # initial tensorboardX writer
    if args.use_tfboard:
        if args.exp_name == 'default':
            writer = SummaryWriter()
        else:
            writer = SummaryWriter('runs/' + args.exp_name)

    if args.dataset == 'voc07trainval':
        args.imdb_name = 'voc_2007_trainval'
        args.imdbval_name = 'voc_2007_test'

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # dataset_cachefile = os.path.join(output_dir, 'dataset.pickle')
    # if not os.path.exists(dataset_cachefile):
    #     imdb, roidb = combined_roidb(args.imdb_name)
    #     cache = [imdb, roidb]
    #     with open(dataset_cachefile, 'wb') as f:
    #         pickle.dump(cache, f)
    #     print('save dataset cache')
    # else:
    #     with open(dataset_cachefile, 'rb') as f:
    #         cache = pickle.load(f)
    #         imdb, roidb = cache[0], cache[1]
    #         print('loaded dataset from cache')

    imdb, roidb = combined_roidb(args.imdb_name)

    # roidb = roidb[:1]

    train_dataset = RoiDataset(roidb)
    train_dataloader = DataLoader(train_dataset, args.batch_size, shuffle=True)

    model = FasterRCNN(pretrained=args.pretrained_model)
    print('model loaded')

    model.load_state_dict(torch.load('output/faster_rcnn_epoch_8.pth')['model'])

    # optimizer
    params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': cfg.TRAIN.BIAS_DECAY and weight_decay or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': weight_decay}]

    optimizer = SGD(params, momentum=momentum)

    if args.use_cuda:
        model = model.cuda()

    model.train()

    iters_per_epoch = int(len(train_dataset) / args.batch_size)

    # start training
    for epoch in range(args.start_epoch, args.max_epochs+1):
        loss_temp = 0
        rpn_tp, rpn_tn, rpn_fg, rpn_bg = 0, 0, 0, 0
        rcnn_tp, rcnn_tn, rcnn_fg, rcnn_bg = 0, 0, 0, 0
        tic = time.time()
        train_data_iter = iter(train_dataloader)

        if epoch in args.decay_lrs:
            lr = lr * gamma
            adjust_learning_rate(optimizer, lr)
            print('adjust learning rate to {}'.format(lr))

        for step in range(iters_per_epoch):
            im_data, gt_boxes, im_info = next(train_data_iter)
            if args.use_cuda:
                im_data = im_data.cuda()
                gt_boxes = gt_boxes.cuda()
                im_info = im_info.cuda()

            im_data_variable = Variable(im_data)

            output = model(im_data_variable, gt_boxes, im_info)
            # rois, rcnn_cls_loss, rcnn_box_loss, rpn_cls_loss, rpn_box_loss, _train_info = output
            #
            # loss = rcnn_cls_loss.mean() + rcnn_box_loss.mean() +\
            #        rpn_cls_loss.mean() + rpn_box_loss.mean()

            rois, rpn_cls_loss, rpn_box_loss, _train_info = output

            loss = rpn_cls_loss.mean() + rpn_box_loss.mean()

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            loss_temp += loss.item()

            if cfg.VERBOSE:
                rpn_tp += _train_info['rpn_tp']
                rpn_tn += _train_info['rpn_tn']
                rpn_fg += _train_info['rpn_num_fg']
                rpn_bg += _train_info['rpn_num_bg']
                # rcnn_tp += _train_info['rcnn_tp']
                # rcnn_tn += _train_info['rcnn_tn']
                # rcnn_fg += _train_info['rcnn_num_fg']
                # rcnn_bg += _train_info['rcnn_num_bg']

            if (step + 1) % args.display_interval == 0:
                toc = time.time()
                loss_temp /= args.display_interval
                rpn_cls_loss_v = rpn_cls_loss.mean().item()
                rpn_box_loss_v = rpn_box_loss.mean().item()
                rcnn_cls_loss_v = 0
                rcnn_box_loss_v = 0
                # rcnn_cls_loss_v = rcnn_cls_loss.mean().item()
                # rcnn_box_loss_v = rcnn_box_loss.mean().item()

                print("[epoch %2d][step %4d/%4d] loss: %.4f, lr: %.2e, time cost %.1fs" \
                      % (epoch, step+1, iters_per_epoch, loss_temp, lr, toc - tic))
                print("\t\t\t rpn_cls_loss_v: %.4f, rpn_box_loss_v: %.4f\n\t\t\t "
                      "rcnn_cls_loss_v: %.4f, rcnn_box_loss_v: %.4f" \
                      % (rpn_cls_loss_v, rpn_box_loss_v, rcnn_cls_loss_v, rcnn_box_loss_v))
                if cfg.VERBOSE:
                    print('\t\t\t RPN : [FG/BG] [%d/%d], FG: %.4f, BG: %.4f'
                          % (rpn_fg, rpn_bg, float(rpn_tp) / rpn_fg, float(rpn_tn) / rpn_bg))
                    # print('\t\t\t RCNN: [FG/BG] [%d/%d], FG: %.4f, BG: %.4f' %
                    #       (rcnn_fg, rcnn_bg, float(rcnn_tp) / rcnn_fg, float(rcnn_tn) / rcnn_bg))

                if args.use_tfboard:
                    n_iter = (epoch - 1) * iters_per_epoch + step + 1
                    writer.add_scalar('losses/loss', loss_temp, n_iter)
                    writer.add_scalar('losses/rpn_cls_loss_v', rpn_cls_loss_v, n_iter)
                    writer.add_scalar('losses/rpn_box_loss_v', rpn_box_loss_v, n_iter)
                    # writer.add_scalar('losses/rcnn_cls_loss_v', rcnn_cls_loss_v, n_iter)

                    if cfg.VERBOSE:
                        writer.add_scalar('rpn/fg_acc', float(rpn_tp) / rpn_fg, n_iter)
                        writer.add_scalar('rpn/bg_acc', float(rpn_tn) / rpn_bg, n_iter)
                        # writer.add_scalar('rcnn/fg_acc', float(rcnn_tp) / rcnn_fg, n_iter)
                        # writer.add_scalar('rcnn/bg_acc', float(rcnn_tn) / rcnn_bg, n_iter)

                loss_temp = 0
                rpn_tp, rpn_tn, rpn_fg, rpn_bg = 0, 0, 0, 0
                rcnn_tp, rcnn_tn, rcnn_fg, rcnn_bg = 0, 0, 0, 0
                tic = time.time()

        if epoch % args.save_interval == 0:
            save_name = os.path.join(output_dir, 'faster_rcnn_epoch_{}.pth'.format(epoch))
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'lr': lr
                }, save_name)

if __name__ == '__main__':
    train()







