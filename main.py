import os
import numpy as np
import traceback
import argparse as arg
from train_evaluate import Trainer
from dataset import get_dataset
import torch
from torch.utils.data import DataLoader
from model.model import Solution


def get_arguments():
    parser = arg.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=3)
    parser.add_argument('-b', '--batch', type=int, default=16)
    parser.add_argument('-l', '--lr', type=float, default=0.001)
    parser.add_argument('-p', '--patience', type=int, default=16, help='早停')
    parser.add_argument('-w', '--workers', type=int, default=4, help='读数据集的线程数')
    parser.add_argument('-c', '--cuda', default=0)
    parser.add_argument('-s', '--step', default=36, help='时间步长，目前还没用')
    parser.add_argument('-m', '--model', default='convlstm', help='模型名称')
    parser.add_argument('--infer', action='store_true', help='推理')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('--small-dataset',
                        action='store_true', dest='small_dataset', help='小数据集调试用')
    parser.add_argument('--model-path', type=str, default='output/model.tar.gz', help='训练模型保存路径')
    return parser.parse_args()


def predict(
    trainer: Trainer,
    debug: bool = False
):
    test_dir = 'tcdata/enso_round1_test_20210201'
    if not debug:
        test_dir = '/' + test_dir
    print('Predict from ' + os.getcwd())
    for i in os.listdir(test_dir):
        if i.endswith('.npy'):
            x = np.load(test_dir + '/' + i)
            np.save('result/' + i, trainer.infer(x))


if __name__ == '__main__':    
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_descr = 'GPU'
    else:
        device = torch.device('cpu')
        device_descr = 'CPU'
    print(f'RUN MODEL: {args.model.upper()}, Device: {device_descr}')
    print(f'Settings: {args}')
        
    cmip_loader = DataLoader(
        dataset=get_dataset('cmip', args.debug, args.small_dataset),
        batch_size=args.batch, shuffle=False, num_workers=args.workers
    )
    soda_loader = DataLoader(
        dataset=get_dataset('soda', args.debug, args.small_dataset),
        batch_size=50, shuffle=False, num_workers=args.workers
    )
    trainer = Trainer(
        model=Solution(device), 
        device=device, 
        train_loader=cmip_loader,
        val_loader=soda_loader,
        lr=args.lr,
        epoch=args.epoch,
        patience=args.patience,
        # lossf=NegativeScore(device=device)
    )
    trainer.fit()
    predict(trainer, args.debug)
