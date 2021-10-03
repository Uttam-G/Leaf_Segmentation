#!/usr/bin/env python
import argparse
import datetime
import os
import os.path as osp

import torch
import yaml

from model.fcn8s import FCN8s
from model.vgg16 import VGG16
from dataloader.Dataloader_CVPPP import CVPPPDataset
from train import Trainer


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))



def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-f')
    parser.add_argument('--gpu', type=int, default = 0, help='gpu id')
    parser.add_argument('--resume', default = "", help='checkpoint path')     
    """start training using pre-trained VGG16 model or with saved training models"""

    parser.add_argument('--max-iteration', type=int, default=4035, help='max iteration') #(1076/4)*15 = 4035
    parser.add_argument('--lr', type=float, default=1.0e-10, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.99, help='momentum')
    
    args = parser.parse_args()
    args.model = 'FCN8s'

    now = datetime.datetime.now()
    args.out = osp.join('.', 'train', 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    
    if not os.path.exists(osp.join('.', 'train', 'training_models')):
        os.makedirs(osp.join('.', 'train', 'training_models'))
    
    
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    
    # 1. dataset
    root = './dataset'
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(CVPPPDataset(root, split='train', transform=True), batch_size=4, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(CVPPPDataset(root, split='val', transform=True), batch_size=1, shuffle=False, **kwargs)

    # 2. model
    model = FCN8s(n_class = 2)
    start_epoch = 0
    start_iteration = 0
    if args.resume != "":
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint) 
    else:
        vgg16 = VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)

    if cuda:
        model = model.cuda()

    # 3. optimizer
    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': args.lr * 2, 'weight_decay': 0},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)


    trainer = Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=args.out,
        max_iter=args.max_iteration,
        interval_validate=538,                  #validate after every 2 epochs(538 iterations)
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
