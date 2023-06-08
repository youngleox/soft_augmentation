# train.py
#!/usr/bin/env	python3

""" train network using pytorch

Adapted, original repo:
https://github.com/weiaicunzai/pytorch-cifar100

"""

import os
import sys
import argparse
import time
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softmax
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.autograd import Variable
from conf import settings
from utils import get_network, get_training_dataloader, \
                  get_test_dataloader, WarmUpLR, soft_target, \
                  smooth_crossentropy,\
                  mixup_data,mixup_criterion,\
                  ECE

import math
import sys
sys.path.append('../optim')
sys.path.append('../core')


def compute_loss(outputs, labels, lam=0, labels_b=None):
    if args.loss == 'st':
        loss = soft_target(outputs, labels,
                           other=args.other,
                           reweight=args.reweight,
                           soften_one_hot=args.soften_one_hot)

    elif args.loss == 'sce':
            loss = smooth_crossentropy(outputs, labels.long(), smoothing=1-args.max_p)

    elif args.loss == 'fl':
            ce_loss = torch.nn.functional.cross_entropy(outputs, labels.long(), reduction='none') # important to add reduction='none' to keep per-batch-item loss
            pt = torch.exp(-ce_loss)
            loss = ((1-pt)**args.g * ce_loss).mean() # mean over the batch

    else:
        if args.mixup == 1 :
            loss = mixup_criterion(loss_function, outputs, labels.long(), labels_b.long(), lam)
        else:
            loss = loss_function(outputs, labels.long())
    return loss

def train(epoch):
    pf = 0 #False
    start = time.time()
    net.train()
    loss_temp = 0
    count = 0
    if args.loss == 'ols':
        loss_function.train()
    for batch_index, (images, labels) in enumerate(cifar_training_loader):

        if args.decay == 'c' or args.decay == 'l':
            train_scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        
        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        alpha = args.a
        lam = 0
        labels_b = None
        if args.mixup == 1:
            images, labels, labels_b, lam = mixup_data(images, labels.long(),
                                                       alpha, True)

            images, labels, labels_b = map(Variable, (images,
                                                      labels, labels_b))

        optimizer.zero_grad()
        outputs = net(images)

        loss = compute_loss(outputs, labels, lam, labels_b)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar_training_loader) + batch_index + 1

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar_training_loader.dataset)
        ))
        loss_temp += loss.item()
        count += 1

    writer.add_scalar('Train/loss', loss_temp/count, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(dataloader=None, train=False, epoch=None):
    if args.task == 'cifar100':
        num_classes = 100
    else:
        num_classes = 10

    start = time.time()
    net.eval()

    test_loss = 0.0 
    correct = 0.0
    preds_all = None
    for (images, labels) in dataloader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
        
        outputs = net(images)
        if args.loss == 'st':
            loss = soft_target(outputs, labels) 
        elif args.loss == 'sce':
            loss = smooth_crossentropy(outputs, labels.long(), smoothing=1-args.max_p)
        elif args.loss == 'fl':
            ce_loss = torch.nn.functional.cross_entropy(outputs, labels.long(), reduction='none') # important to add reduction='none' to keep per-batch-item loss
            pt = torch.exp(-ce_loss)
            loss = ((1-pt)**args.g * ce_loss).mean() # mean over the batch   
        else:
            loss = loss_function(outputs, labels.long())
        test_loss += loss.item() * len(labels)

        if args.other:
            _, preds = outputs[:,:args.nclass].max(1)
        else:
            _, preds = outputs.max(1)

        if preds_all is None:
            preds_all = outputs.clone()
            labels_all = labels.clone().view(labels.size(0), -1)
        else:
            preds_all = torch.vstack((preds_all,outputs.clone()))
            labels_all = torch.vstack((labels_all,labels.clone().view(labels.size(0), -1)))

        correct += preds.eq(labels.long()).sum()
    ece, confs, accs, counts = ECE(labels_all,preds_all,num_bins=11)
    finish = time.time()
    print('Evaluating Network.....')
    acc = correct.float() / len(dataloader.dataset)
    mean_loss = test_loss / len(dataloader.dataset)

    name = "train" if train else "test"
    print('{} set: Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        name, mean_loss, acc, finish - start ))

    #add informations to tensorboard
    if train:
        writer.add_scalar('Train/Average loss', mean_loss, epoch)
        writer.add_scalar('Train/Accuracy', acc, epoch)
        writer.add_scalar('Train/ECE', ece, epoch)
    else:
        writer.add_scalar('Test/Average loss', mean_loss, epoch)
        writer.add_scalar('Test/Accuracy', acc, epoch)
        writer.add_scalar('Test/ECE', ece, epoch)

    return acc, mean_loss

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=-1, help='random seed')

    parser.add_argument('--net', type=str, default='resnet18', help='model type')
    
    parser.add_argument('--nogpu', action='store_false', default=True, dest="gpu", 
                        help='use gpu or not')

    parser.add_argument('--b', type=int, default=128, help='batch size for dataloader')

    parser.add_argument('--warm', type=int, default=0, help='warm up training phase')

    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')

    parser.add_argument('--momentum', default=0.9 , type=float,help='momentum/beta1')

    parser.add_argument('--prefix', default='', type=str)

    parser.add_argument('--gamma', default=0.2 , type=float,help='lr decay')

    parser.add_argument('--wd', default=0.0005 , type=float)

    parser.add_argument('--optimizer', default='sgd', type=str)

    parser.add_argument('--task', default='cifar10' , type=str)

    parser.add_argument('--da', type=int, default=0, 
                        help='data augmentation options, \
                            -1: no DA, 0: standard, \
                            1: cutout, 2: custom')

    parser.add_argument('--ra', type=int, default=0, 
                        help='auto augmentation options,0: no RA, 1: RandAugment')

    parser.add_argument('--noise', type=float, default=0.0, 
                        help='sigma of noise distribution, default: no noise')

    parser.add_argument('--pnoise', type=float, default=1.0, 
                        help='power of noise to probability function')

    parser.add_argument('--bgnoise', type=float, default=1, 
                        help='background type, default: 0, zeros. 1, noise')

    parser.add_argument('--crop', type=float, default=12.0, 
                        help='sigma of crop offset, default: 12')

    parser.add_argument('--pcrop', type=float, default=2.0, 
                        help='power of crop overlap to probability function')

    parser.add_argument('--bgcrop', type=float, default=0.01, 
                        help='crop background type, default: 0, zeros. 1, noise')

    parser.add_argument('--l', type=float, default=0.0, 
                        help='mean cutout length')

    parser.add_argument('--max_p', type=float, default=1.0, 
                        help='roughly equal to 1 - smooth in label smoothing')

    parser.add_argument('--loss', default='ce', type=str,
                        help="crossentropy (ce) or soft target (st) loss" )

    parser.add_argument('--g', default=2, type=float,help='gamma for focal loss')

    parser.add_argument('--sch', type=float, default=2.5, 
                        help='learning schedule scaling, default 2.5: 500 epochs')
    
    parser.add_argument('--decay', default='c', 
                        help='Step (s), Linear(l), or Cosine(c) annealing scheduler')

    parser.add_argument('--n', type=int, default=16, help='number of workers')

    parser.add_argument('--no-nag', action='store_false', dest="nag", default=True, 
                        help='disable Nesterov for SGD')

    parser.add_argument('--mixup', type=int, default=0, help='0: no mixup, 1: mixup')

    parser.add_argument('--a', default=1.0, type=float,help='mixup alpha')

    parser.add_argument('--ft', default=20, type=int,help='fine tune epochs')

    parser.add_argument('--r', default=1000.0, type=float,
                        help='augmentation strength decay at the end')
    
    parser.add_argument('--other', action='store_true', dest="other", help='add other class')

    parser.add_argument('--rw', action='store_true', default=True, dest="reweight", 
                        help='wether to reweight other classes')

    parser.add_argument('--no-soften', action='store_false', default=True, dest="soften_one_hot", 
                        help='whether to soften one hot targets')



    args = parser.parse_args()
    if args.seed == -1:
        args.seed = random.randint(0,1000)
        print("Generating random seed!")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    settings.MILESTONES = [int(mst * args.sch) for mst in settings.MILESTONES]
    settings.EPOCH = int(settings.EPOCH * args.sch)
    print(settings.MILESTONES)

    #data preprocessing:
    if args.task == "cifar100":
        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
        args.nclass = 100
    elif args.task == "cifar10":
        # below from cutout official repo
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]]
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
        args.nclass = 10
    else:
        print("invalid task!!")

    if args.other:
        args.nclass += 1

    print(args)
    net = get_network(args)
    net = torch.nn.DataParallel(net).cuda()
    cifar_training_loader = get_training_dataloader(
        mean,
        std,
        num_workers=args.n,
        batch_size=args.b,
        shuffle=True,
        task = args.task,
        da = args.da,
        max_p = args.max_p,
        sigma_noise=args.noise, pow_noise=args.pnoise, bg_noise=args.bgnoise,
        sigma_crop=args.crop, pow_crop=args.pcrop, bg_crop=args.bgcrop,
        length_cut=args.l, mask_cut=0,
        aa=args.ra
    )

    cifar_test_loader = get_test_dataloader(
        mean,
        std,
        num_workers=args.n,
        batch_size=args.b,
        shuffle=False,
        task = args.task
    )
    #test training acc
    cifar_train_test_loader = get_test_dataloader(
        mean,
        std,
        num_workers=args.n,
        batch_size=args.b,
        shuffle=False,
        task = args.task,
        train = True
    )

    if args.loss == "ce":
        loss_function = nn.CrossEntropyLoss()
    elif args.loss == "fl":
        print("using focal loss!!")
        loss_function = nn.CrossEntropyLoss()
        args.prefix += args.loss + "_gamma_" + str(args.g)
    elif args.loss == "st":
        print("using soft target loss!!")
        args.prefix += args.loss 
        args.prefix += "_rw" if args.reweight else ""
        args.prefix += "_no_soften" if not args.soften_one_hot else ""

    else:
        print("undefined loss")

    if args.optimizer == 'sgd':
        print("using sgd!")
        optimizer = optim.SGD(net.parameters(), lr=args.lr, nesterov=args.nag, momentum=args.momentum, weight_decay=args.wd)
        base_optimizer = optim.SGD


    iter_per_epoch = len(cifar_training_loader)
    if args.decay == 'c':
        Max_it = iter_per_epoch * (settings.EPOCH+1) + 1
        train_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,Max_it)
    elif args.decay == 'l':
        Max_it = iter_per_epoch * (settings.EPOCH+1) + 1
        lambda1 = lambda it: (Max_it - it) / Max_it
        train_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    else:
        train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=args.gamma) #learning rate decay
        
    args.prefix += "_Mix{}_A{}_".format(args.mixup,args.a) if args.mixup else "_"

    if args.da == 0:
        args.prefix = "seed" + str(args.seed) + "_DA" + str(args.da) + \
                    "_MaxP" + str(args.max_p) + "_L" + str(args.l) + \
                    "_sch" + str(args.sch) + args.prefix + '_'

    elif args.da == 1:
        args.prefix = "seed" + str(args.seed) + "_DA" + str(args.da) + \
                    "_MaxP" + str(args.max_p) + \
                    "_L" + str(args.l) + \
                    "_sch" + str(args.sch) + args.prefix + '_'

    elif args.da == 2 or args.da == 3 or args.da >= 98:
        args.prefix = "seed" + str(args.seed) + "_DA" + str(args.da) + args.prefix
        args.prefix += "_other_" if args.other else ""
        args.prefix +=  "_MaxP" + str(args.max_p) if not args.max_p == 1 else ""
        args.prefix += "" if args.noise == 0 else "_Noise" + str(args.noise) + "P" + str(args.pnoise) + "BG" + str(args.bgnoise)
        args.prefix +=  "" if args.crop == 0 else "_Crop" + str(args.crop) + "P" + str(args.pcrop) + "BG" + str(args.bgcrop)
        args.prefix += "_sch" + str(args.sch) + "_DE_" + str(args.decay) + "_R_" + str(args.r) + '_'
    
    args.prefix += "_RA" if args.ra == 1 else "" 

    nag = 'NAG' if args.nag else ""
    if args.optimizer == "sgd":
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.task, args.net, 
                        args.prefix + args.optimizer + str(args.lr)+ nag +
                        '_momentum_'+str(args.momentum)+'_wd_'+str(args.wd),
                        settings.TIME_NOW)
    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.task, args.net, 
                        args.prefix + args.optimizer + str(args.lr)+ 
                        '_wd_'+str(args.wd),
                        settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
    if args.optimizer == "sgd":
        writer = SummaryWriter(log_dir=os.path.join(
                            settings.LOG_DIR, args.task, args.net,
                            args.prefix + args.optimizer + str(args.lr)+nag +#'_g_'+str(args.gamma)+
                            '_momentum_'+str(args.momentum)+'_wd_'+str(args.wd),
                            settings.TIME_NOW))

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    best_test_acc = 0.0
    best_test_acc_epoch = 0

    best_test_loss = 10.0
    best_test_loss_epoch = 0
        
    best_train_acc = 0.0
    best_train_acc_epoch = 0

    best_train_loss = 10.0
    best_train_loss_epoch = 0
    base_lr = optimizer.param_groups[0]['lr']
    base_a = args.a
    for epoch in range(0, settings.EPOCH+1):
        if args.decay == 's':
            train_scheduler.step(epoch)
        writer.add_scalar("learning/lr",optimizer.param_groups[0]['lr'],epoch)
        
        if epoch >= max(settings.EPOCH * 0.05, settings.EPOCH - args.ft):
            print("Switching to finetune augmentation", str(args.r),'!')
            args.a = 0
            if args.da > 0:
                aug_ratio = 1.0/float(args.r)
                cifar_training_loader = get_training_dataloader(
                    mean,
                    std,
                    num_workers=args.n,
                    batch_size=args.b,
                    shuffle=True,
                    task = args.task,
                    da = args.da,
                    max_p = args.max_p,
                    sigma_noise=args.noise*aug_ratio, pow_noise=args.pnoise, bg_noise=args.bgnoise,
                    sigma_crop=args.crop*aug_ratio, pow_crop=args.pcrop, bg_crop=args.bgcrop,
                    length_cut=args.l*aug_ratio, mask_cut=0,
                    aa=args.ra
                )
 
        train(epoch)
        if epoch % 5 == 0:
            test_acc, test_loss = eval_training(dataloader=cifar_test_loader,train=False,epoch=epoch)
            train_acc, train_loss = test_acc, test_loss
            
            print(writer.log_dir)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_epoch = epoch

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_loss_epoch = epoch
            
            if train_acc > best_train_acc:
                best_train_acc = train_acc
                best_train_acc_epoch = epoch

            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_train_loss_epoch = epoch

            #start to save best performance model after learning rate decay to 0.01
            writer.add_scalar('Test/Best Average loss', best_test_loss, epoch)
            writer.add_scalar('Test/Best Accuracy', best_test_acc, epoch)
            writer.add_scalar('Train/Best Average loss', best_train_loss, epoch)
            writer.add_scalar('Train/Best Accuracy', best_train_acc, epoch)

        if not epoch % settings.SAVE_EPOCH:
            torch.save(net.state_dict(), checkpoint_path.format(net=args.net, epoch=epoch, type='regular'))

    writer.close()
