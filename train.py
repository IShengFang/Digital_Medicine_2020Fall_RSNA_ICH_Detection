# -*- coding: utf-8 -*-
import os
import json
import random
import argparse
import matplotlib
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import data
import model
import utils
from radam import RAdam
from data.dataset import ICHDataset
from mixup import mixup_data, mixup_criterion

'''
# of different patient ID
硬膜外出血 epidural: 223
健康 healthy: 977
腦實質性出血 intraparenchymal: 807
腦室內出血 intraventricular: 712
蛛網膜下腔出血 subarachnoid: 717
硬膜下出血 subdural: 794
'''
matplotlib.use('AGG')


def load_args():
    parser = argparse.ArgumentParser()
    # data path
    parser.add_argument('--train_set', type=str,
                        default='dataset/TrainingData/',
                        help='Directory path of dataset for training')

    # split relate
    parser.add_argument('--split_file_dir', type=str,
                        default='split_file/',
                        help='Directory path for save split file')

    parser.add_argument('--name_split_file', action='store_true', default=False)
    parser.add_argument('--use_old_split_file', action='store_true', default=False)

    # save path
    parser.add_argument('--log_dir', type=str,
                        default='logs/',
                        help='Directory path for save tensorboard log')
    parser.add_argument('--cpt_dir', type=str,
                        default='cpts/',
                        help='Directory path for save model cpt')

    parser.add_argument('--cpt_num', type=int, default=1000)

    # model options
    parser.add_argument('--model_name', type=str,
                        default='resnet18',
                        help='resnet18, resnet34, resnet50, resnet101, resnet152, \
                              resnext50_32x4d, resnext101_32x8d, \
                              wide_resnet50_2, wide_resnet101_2, \
                              densenet121, densenet169, densenet161, densenet201, \
                              inception_v3, googlenet, ')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--fixed_weight', action='store_true', default=False)

    # data relate
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--split_ratio', type=float, default=0.7)
    parser.add_argument('--random_apply_aug', action='store_true', default=False)
    parser.add_argument('--random_horizontal_flip', action='store_true', default=False)
    parser.add_argument('--random_perspective', action='store_true', default=False)
    parser.add_argument('--random_rotation', action='store_true', default=False)
    parser.add_argument('--random_order', action='store_true', default=False)
    parser.add_argument('--random_erasing', action='store_true', default=False)
    parser.add_argument('--pre_kernel', type=str,  default='bsb',
                        help='bsb(defualt), meta, brain, all_channel_window, rainbow_window')

    # mixup
    parser.add_argument('--mixup_alpha', type=float, default=1.0,
                        help='mixup interpolation strength')

    # training options
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--beta_1', type=float, default=0.9)
    parser.add_argument('--beta_2', type=float, default=0.999)
    parser.add_argument('--radam', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_threads', type=int, default=8)

    parser.add_argument('--num_classes', type=int, default=6)

    # exp name
    parser.add_argument('--exp_name', default='ICH_detection',
                        help='Experiment name')

    parser.add_argument('--generate_exp_name', action='store_true', default=False)
    args = parser.parse_args()

    if args.generate_exp_name:
        args.exp_name = generate_exp_name(args)

    args.log_dir = os.path.join(args.log_dir, args.exp_name)
    utils.check_and_make_dir(args.log_dir)
    args.cpt_dir = os.path.join(args.cpt_dir, args.exp_name)
    utils.check_and_make_dir(args.cpt_dir)

    if args.name_split_file:
        args.split_file_dir = os.path.join(args.split_file_dir, args.exp_name)

    return args


def generate_exp_name(args):
    if args.pretrained:
        args.exp_name += '_Pretrain'
        if args.fixed_weight:
            args.exp_name += '_fixed'
    else:
        args.exp_name += '_Scratch'

    args.exp_name += '_{}_E{}_lr{}_b1_{}_b2_{}_bs_{}_splt_{}_prekeral_{}'.format(
        args.model_name, args.epochs, args.lr, args.beta_1, args.beta_2,
        args.batch_size, args.split_ratio, args.pre_kernel,
    )
    if args.radam:
        args.exp_name += '_radam'

    if args.random_apply_aug:
        args.exp_name += '_rand_app'
    if args.random_horizontal_flip:
        args.exp_name += '_rand_flip'
    if args.random_perspective:
        args.exp_name += '_rand_pers'
    if args.random_rotation:
        args.exp_name += '_rand_rota'
    if args.random_order:
        args.exp_name += '_rand_ord'
    if args.random_erasing:
        args.exp_name += '_rand_eras'
    return args.exp_name


def train(net, writer, class_dict, train_loader, valid_loader,
          optimizer, criterion, epochs, device, cpt_num, args):
    step = 0
    print('start training...')
    exp_pbar = tqdm(range(epochs))
    for epoch in exp_pbar:
        net.train()
        running_loss = 0
        batch_idx = 0
        epoch_pbar = tqdm(train_loader)
        for imgs, labels in epoch_pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, args.mixup_alpha, device)

            optimizer.zero_grad()

            output = net(imgs)

            loss_func = mixup_criterion(labels_a, labels_b, lam)
            loss = loss_func(criterion, output)
            # loss = criterion(output, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            writer.add_scalar('train/loss', loss.item(), step)
            step += 1
            epoch_pbar.set_description(f'epoch: {epoch:>2}/{epochs}, batch: {batch_idx:>3}/{len(train_loader)}, loss: {loss.item():.5f}')

            if step%cpt_num == 0:
                save_path = os.path.join(args.cpt_dir, '{}_E_{}_iter_{}.cpt'.format(args.model_name, epoch, step,))
                print('saving model cpt @ {}'.format(save_path))
                torch.save(net.state_dict(), save_path)

            batch_idx += 1

        running_loss /= batch_idx
        print('----------------------------')
        exp_pbar.set_description(f'epoch: {epoch:>2}/{epochs}, avg. loss: {running_loss:.5f}')
        evaluate(net, valid_loader, train_loader, criterion, writer, device, step, class_dict)
        print('============================')

        save_path = os.path.join(args.cpt_dir, '{}_E_{}_iter_{}.cpt'.format(
                                                args.model_name, epoch, step,))
        print('saving model cpt @ {}'.format(save_path))
        torch.save(net.state_dict(), save_path)


def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    fig, ax = plt.subplots()
    sn.heatmap(cm, annot=True, ax=ax, cmap='Blues', fmt='d')
    ax.set_xlabel('Prediction')
    ax.set_ylabel('Ground truth')
    ax.xaxis.set_ticklabels(labels, rotation=45)
    ax.yaxis.set_ticklabels(labels, rotation=0)
    plt.tight_layout()
    return fig


def evaluate(net, valid_loader, train_loader, criterion, logger, device, step, class_dict):
    net.eval()
    y_true = []
    y_pred = []
    print('start evaluating...')

    print('valid_set')
    eval_pbar = tqdm(valid_loader)
    correct = 0
    batch_idx = 0
    val_loss = 0
    for imgs, labels in eval_pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            output = net(imgs)
            loss = criterion(output, labels)
            val_loss += loss.item()

        _, pred = output.data.max(1)
        correct += (labels == pred).sum().item()
        y_true += labels.data.cpu()
        y_pred += pred.data.cpu()
        batch_idx += 1
    val_loss = val_loss / batch_idx
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(y_true, y_pred)
    val_acc = correct / len(valid_loader.dataset)
    val_cm = plot_confusion_matrix(y_true, y_pred, class_dict.values())

    eval_pbar = tqdm(train_loader)
    correct = 0
    batch_idx = 0
    for imgs, labels in eval_pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            output = net(imgs)
        _, pred = output.data.max(1)
        correct += (labels == pred).sum().item()
        y_true += labels.data.cpu()
        y_pred += pred.data.cpu()
        batch_idx += 1
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(y_true, y_pred)
    train_acc = correct / len(train_loader.dataset)
    train_cm = plot_confusion_matrix(y_true, y_pred, class_dict.values())
    for class_index in range(args.num_classes):
        class_name = class_dict[str(class_index)]
        logger.add_scalar(f'{class_name}/val/precision', val_precision[class_index], step)
        logger.add_scalar(f'{class_name}/val/recall', val_recall[class_index], step)
        logger.add_scalar(f'{class_name}/val/f1-score', val_f1[class_index], step)
        logger.add_scalar(f'{class_name}/train/precision', train_precision[class_index], step)
        logger.add_scalar(f'{class_name}/train/recall', train_recall[class_index], step)
        logger.add_scalar(f'{class_name}/train/f1-score', train_f1[class_index], step)
    logger.add_scalar('val/loss', val_loss, step)
    logger.add_scalar('val/total_acc', val_acc, step)
    logger.add_figure('val/cm', val_cm, step)
    logger.add_scalar('train/total_acc', train_acc, step)
    logger.add_figure('train/cm', train_cm, step)


def test(model, dataloader, device):
    res = []
    for batch_idx, imgs in enumerate(dataloader):
        imgs = imgs.to(device)
        output = net(imgs)
        _, pred = output.data.max(1)
        res += pred.tolist()
    return res


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


if __name__ == '__main__':
    args = load_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not args.use_old_split_file:
        data.split_data(args.train_set, args.split_file_dir, args.split_ratio)

    class_dict = json.load(open('label.json', 'r'))
    print(json.dumps(class_dict, indent=2))

    print('setting data aug')
    train_transforms = []
    if args.random_horizontal_flip:
        train_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
    if args.random_perspective:
        train_transforms.append(transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=2, fill=0))
    if args.random_rotation:
        train_transforms.append(transforms.RandomRotation(15))
    if args.random_erasing:
        train_transforms.append(transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False))

    if args.random_order:
        random.shuffle(train_transforms)
    if args.random_apply_aug:
        train_transforms = transforms.RandomApply(train_transforms, p=0.5)
    elif args.random_order:
        train_transforms = transforms.RandomOder(train_transforms)
    else:
        train_transforms = transforms.Compose(train_transforms)

    train_set = ICHDataset('{}/train.txt'.format(args.split_file_dir), args.img_size, kernel=args.pre_kernel, transform=train_transforms)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)
    valid_set = ICHDataset('{}/valid.txt'.format(args.split_file_dir), args.img_size,  kernel=args.pre_kernel)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_threads)

    net = model.Model(args.model_name, args.pretrained)
    criterion = nn.CrossEntropyLoss()
    net = net.to(device)

    if args.radam:
        if args.fixed_weight:
            optimizer = RAdam(net.fc.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        else:
            optimizer = RAdam(net.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
    else:
        if args.fixed_weight:
            optimizer = optim.Adam(net.fc.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))
        else:
            optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(args.beta_1, args.beta_2))

    writer = SummaryWriter(args.log_dir)
    train(net, writer, class_dict,
          train_loader, valid_loader,
          optimizer, criterion, args.epochs, device,
          args.cpt_num, args)
