# -*- coding: utf-8 -*-
import os
import json
import argparse
import collections
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import model
import utils
from data.dataset import ICHDataset

'''
# of different patient ID
硬膜外出血 epidural: 223
健康 healthy: 977
腦實質性出血 intraparenchymal: 807
腦室內出血 intraventricular: 712
蛛網膜下腔出血 subarachnoid: 717
硬膜下出血 subdural: 794
'''


def load_args():
    parser = argparse.ArgumentParser()
    # data path
    parser.add_argument('--test_set', type=str,
                        default='dataset/TestingData/',
                        help='Directory path of dataset for testing')

    # split relate
    parser.add_argument('--split_file_dir', type=str,
                        default='split_file/',
                        help='Directory path for saved split file')

    # save path
    parser.add_argument('--cpt_dir', type=str,
                        default='cpts/',
                        help='Directory path for saved model cpt')

    # model options
    parser.add_argument('--model_name', type=str,
                        default='resnet18',
                        help='resnet18, resnet34, resnet50, resnet101, resnet152, \
                              resnext50_32x4d, resnext101_32x8d, \
                              wide_resnet50_2, wide_resnet101_2, \
                              densenet121, densenet169, densenet161, densenet201, \
                              inception_v3, googlenet, ')

    # data relate
    parser.add_argument('--img_size', type=int, default=512)
    parser.add_argument('--pre_kernel', type=str,  default='bsb',
                        help='bsb(defualt), meta, brain, all_channel_window, rainbow_window')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_threads', type=int, default=8)

    parser.add_argument('--num_classes', type=int, default=6)

    # exp name
    parser.add_argument('--exp_name', default='ICH_detection',
                        help='Experiment name')
    parser.add_argument('--cpt_name', type=str)

    parser.add_argument('--output_dir', default='output',
                        help='Directory for saving inference result')

    args = parser.parse_args()

    args.cpt_name = os.path.join(os.path.join(args.cpt_dir, args.exp_name), args.cpt_name)
    args.split_file_dir = os.path.join(args.split_file_dir, args.exp_name)

    utils.check_and_make_dir(args.output_dir)

    return args


def test(net, class_dict, test_loader, device, args):
    print('start testing...')
    net.eval()
    y_pred = []
    filenames = []
    test_pbar = tqdm(test_loader)
    for imgs, fns in test_pbar:
        imgs = imgs.to(device)

        output = net(imgs)
        _, pred = output.data.max(1)
        y_pred += pred.data.cpu().tolist()
        filenames += fns

    result = {}
    for (fn, pred) in zip(filenames, y_pred):
        result[fn.split('/')[-1].replace('.dcm', '')] = class_dict[str(pred)]
    result = collections.OrderedDict(sorted(result.items()))
    testcase = []
    pred = []
    for k in result:
        testcase.append(k)
        pred.append(result[k])
    df = pd.DataFrame({
        'testcase': testcase,
        'result': pred
    })

    save_path = args.output_dir
    result_filename_json = os.path.join(save_path, f'{args.exp_name}.json')
    result_filename_xlsx = os.path.join(save_path, f'{args.exp_name}.xlsx')
    df.to_excel(result_filename_xlsx, header=False, index=False)
    json.dump(result, open(result_filename_json, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

    print('saving inference result @ {}'.format(save_path))


if __name__ == '__main__':
    args = load_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class_dict = json.load(open(os.path.join(args.split_file_dir, 'label.json'), 'r'))
    print(json.dumps(class_dict, indent=2))

    test_set = ICHDataset(args.test_set, args.img_size, kernel=args.pre_kernel)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_threads)

    net = model.Model(args.model_name, pretrained=True)
    net.load_state_dict(torch.load(args.cpt_name))
    net = net.to(device)

    test(net, class_dict, test_loader, device, args)
