import datetime
from distutils.version import LooseVersion
import math
import os
import os.path as osp
import shutil

import numpy as np
import pytz
import skimage.io
import torch
import torchvision
from torch.autograd import Variable

"""Use this to log data in the tensorboard for observation"""
#from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import tqdm

from utils.loss import cross_entropy2d
from utils.metrics import label_accuracy_score, visualize_segmentation, get_tile_image


class Trainer(object):

    def __init__(self, cuda, model, optimizer, train_loader, val_loader, out, max_iter, size_average = False, interval_validate = None):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = datetime.datetime.now(pytz.timezone('Asia/Kolkata'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_iter = max_iter
        self.best_mean_iu = 0


    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        label_trues, label_preds = [], []
        for batch_idx, (orig_data, orig_target, data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Valid iteration=%d' % self.iteration, ncols=80,
                leave=False):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            with torch.no_grad():
                score = self.model(data)

            loss = cross_entropy2d(score, target,
                                   size_average=self.size_average)

            loss_data = loss.data.item()

            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')
            val_loss += loss_data / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()

            pred = score.clone().detach()
            pred,idxs = torch.max(pred, dim=1)

            pred = pred.cpu()
            idxs = idxs.cpu()

            orig_target = torch.unsqueeze((orig_target*255), dim=-1)
            idxs = torch.unsqueeze((idxs*255), dim=-1)

            orig_target = orig_target.type(torch.ByteTensor)
            idxs = idxs.type(torch.ByteTensor)

            """if batch_idx == 0:
              with summary_writer.as_default():
                summary.scalar('val_loss', loss_data, step=self.iteration)
                summary.image('val_image', orig_data, step=self.iteration)
                summary.image('val_output_prediction', idxs, step=self.iteration)
                summary.image('val_target_prediction', orig_target, step=self.iteration)"""

            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransform(img, lt)
                label_trues.append(lt)
                label_preds.append(lp)
                if len(visualizations) < 9:
                    viz = visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
        
        metrics = label_accuracy_score(label_trues, label_preds, n_class)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'iter%012d.jpg' % self.iteration)
        skimage.io.imsave(out_file, get_tile_image(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Asia/Kolkata')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        #print(mean_iu)
        if training:
            self.model.train()


    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (orig_data, orig_target, data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80, leave=False):
            iteration = batch_idx + self.epoch * len(self.train_loader)
            
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                continue  # for resuming
            
            self.iteration = iteration

            if self.iteration % self.interval_validate == 0:
                self.validate()
                torch.save(self.model.state_dict(),f"./train/training_models/fcn_cvppp_{self.epoch}.pth")

            assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optim.zero_grad()
            
            score = self.model(data)

            loss = cross_entropy2d(score, target, size_average=self.size_average)
            loss /= len(data)
            loss_data = loss.data.item()

            pred = score.clone().detach()
            pred,idxs = torch.max(pred, dim=1)

            pred = pred.cpu()
            idxs = idxs.cpu()

            orig_target = torch.unsqueeze((orig_target*255), dim=-1)
            idxs = torch.unsqueeze((idxs*255), dim=-1)

            orig_target = np.reshape(orig_target, (-1, 410, 410, 1))
            orig_data = np.reshape(orig_data, (-1, 410, 410, 3))
            idxs = np.reshape(idxs, (-1, 410, 410, 1))

            orig_target = orig_target.type(torch.ByteTensor)
            idxs = idxs.type(torch.ByteTensor)

            """with summary_writer.as_default():
              summary.scalar('train_loss', loss_data, step=iteration)
              summary.image('train_images', orig_data, max_outputs = 4, step=iteration)
              summary.image('train_output_predictions', idxs, max_outputs = 4, step=iteration)
              summary.image('train_target_predictions', orig_target, max_outputs = 4, step=iteration)"""
            
            
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()

            acc, acc_cls, mean_iu, fwavacc = label_accuracy_score(lbl_true, lbl_pred, n_class=n_class)
            """with summary_writer.as_default():
              summary.scalar('train_accuracy', acc, step=iteration)
              summary.scalar('train_mean_iu', mean_iu, step=iteration)"""
            
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Asia/Kolkata')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')

            if self.iteration >= self.max_iter:
                break


    def train(self):
        max_epoch = int(math.ceil(1. * self.max_iter / len(self.train_loader)))
        for epoch in tqdm.trange(self.epoch, max_epoch,
                                 desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.iteration >= self.max_iter:
                break
