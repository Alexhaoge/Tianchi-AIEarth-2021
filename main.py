import os
import numpy as np
import traceback
import argparse as arg
from train_evaluate import Trainer
from dataset import get_dataset_old as get_dataset
import torch
from torch.utils.data import DataLoader, random_split
from model.model import Solution


def get_arguments():
    parser = arg.ArgumentParser()
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-b', '--batch', type=int, default=16)
    parser.add_argument('-l', '--lr', type=float, default=0.005)
    parser.add_argument('-p', '--patience', type=int, default=16, help='早停')
    parser.add_argument('-w', '--workers', type=int, default=4, help='读数据集的线程数')
    parser.add_argument('-c', '--cuda', default=0)
    # parser.add_argument('-t', '--timestep', default=36, help='时间步长，目前还没用')
    parser.add_argument('-m', '--model', default=2, type=int,
                        help='模型版本 1:错的ConvLSTM  2:对的ConvLSTM')
    # parser.add_argument('--infer', action='store_true', help='推理')
    parser.add_argument('--debug', action='store_true', help='调试模式')
    parser.add_argument('-s', '--small-dataset', type=int, default=-1,
                        dest='small_dataset', help='小数据集调试用')
    parser.add_argument('--model-path', type=str, default='output/model.tar.gz', help='训练模型保存路径')
    parser.add_argument('--loss', type=str, default='rmse', help='训练用损失函数')
    parser.add_argument('--val-loss', type=str, default='score', help='验证用损失函数')
    parser.add_argument('--no-stop', action='store_true', help='禁用早停')
    parser.add_argument('--refit', action='store_true', help='SODA重训练')
    return parser.parse_args()


def predict(
    trainer: Trainer,
    model_path: str,
    debug: bool = False
):
    map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    trainer.model.load_state_dict(torch.load(model_path, map_location))
    print('successully load model checkpoints')
    print('load model succeed')  
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
    print(f'Device: {device_descr}, Settings: {args}')
    cmip = get_dataset('cmip', args.debug, args.small_dataset)
    cmip_train, cmip_val = random_split(
        cmip, [int(len(cmip)*0.8), len(cmip)-int(len(cmip)*0.8)])
    # cmip_loader = DataLoader(
    #     dataset=get_dataset('cmip', args.debug, args.small_dataset),
    #     batch_size=args.batch, shuffle=False, num_workers=args.workers,
    # )
    cmip_train_loader = DataLoader(
        dataset=cmip_train, batch_size=args.batch,
        shuffle=False, num_workers=args.workers,
    )
    cmip_val_loader = DataLoader(
        dataset=cmip_val, batch_size=args.batch,
        shuffle=False, num_workers=args.workers,
    )
    soda_loader = DataLoader(
        dataset=get_dataset('soda', debug_mode=args.debug, small=-1),
        batch_size=args.batch, shuffle=False, num_workers=args.workers
    )
    trainer = Trainer(
        model_path=args.model_path,
        model=Solution(args.model, device=device), 
        device=device, 
        train_loader=cmip_train_loader,
        val_loader=cmip_val_loader,
        lr=args.lr,
        epoch=args.epoch,
        patience=args.patience,
        lossf=args.loss,
        val_lossf=args.val_loss,
        no_stop=args.no_stop
    )
    trainer.fit()
    if args.refit:
        trainer.refit_refresh(50, 0.0001)
        trainer.train_loader = soda_loader
        trainer.fit()
    predict(trainer, args.model_path, args.debug)
