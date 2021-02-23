from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
import torch
from utils import EarlyStopping
from torch.utils import data
from numpy import ndarray
# from datetime import datetime
from model.loss import LossFactory
import time


class Trainer:
    def __init__(
        self, 
        model: nn.Module,
        device: torch.device,
        model_path: str,
        train_loader: data.DataLoader = None,
        val_loader: data.DataLoader = None,
        lr: float = 0.001,
        epoch: int = 100,
        patience: int = 16,
        lossf: str = 'score',
        val_lossf: str = 'score',
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_path = model_path
        self.model = model.to(device)
        self.opt = RMSprop(self.model.parameters(), lr=lr)
        self.scheduler = ExponentialLR(optimizer=self.opt, gamma=0.94)
        self.device = device
        self.early = EarlyStopping(patience=patience)
        self.epochs = epoch
        self.lossf =  LossFactory(lossf, self.device)
        self.val_lossf = LossFactory(val_lossf, self.device)

    def fit(self):
        print('start fitting at '+time.strftime('%d-%H:%I:%M:%S', time.localtime()))
        train_losses, val_losses = [], []
        _ = torch.isnan(self.train_loader.dataset.tensors[0]).sum().item()
        if _ > 0:
            print('Input exists NaN {}'.format(_))
        for epoch in range(1, self.epochs+1):
            train_loss = self.__train()
            val_loss = self.__eval()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f'Epoch:{epoch}/{self.epochs} \
                    loss:{train_loss:.4f} \
                    val_loss:{val_loss:.4f} \
                    at ' + time.strftime('%d-%H:%I:%M:%S', time.localtime()))
            self.early(val_loss, self.model, self.model_path)
            if self.early.isToStop:
                print("=> Stopped")
                break
        return  train_losses, val_losses
        # plot([train_losses, val_losses], ['Training', 'Validation'], 'Epochs', 'Error', 'Error analysis')

    def __train(self) -> float:
        self.model.train()
        epoch_loss = .0
        for id, (inputs, target) in enumerate(self.train_loader):
            inputs, target = inputs.to(self.device), target.to(self.device)
            output = self.model(inputs)
            loss = self.lossf(output, target[:,12:])
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if id % 2:
                self.scheduler.step()
            epoch_loss += loss.item()
        return epoch_loss/len(self.train_loader)

    def __eval(self) -> float:
        # 这需要改吗？
        self.model.eval()
        loss_sum = .0
        with torch.no_grad():
            for id, (inputs, target) in enumerate(self.val_loader):
                inputs, target = inputs.to(self.device), target.to(self.device)
                output = self.model(inputs)
                assert torch.isnan(output).sum().item() == 0
                loss_sum += self.val_lossf(output, target[:,12:]).item()
        return loss_sum/len(self.val_loader)

    def eval(self) -> float:
        return self.__eval()

    def infer(self, inputs: ndarray) -> ndarray:
        self.model.eval()
        with torch.no_grad():
            x = torch.Tensor(inputs).reshape(1, 12, 24, 72, 4)\
                    .permute(0, 1, 4, 2, 3).to(self.device)
            output = self.model.infer(x)
            assert torch.isnan(output).sum().item() == 0
        return output.view(-1).cpu().numpy()


# def cross_validation(
#     ds: data.TensorDataset,
#     K: int = 3, batch: int = 16,
#     learning_rate: float = 0.001,
#     focal: bool = True
# ):
#     print('Total {} images, {} folds, batch size {}, use focal loss: {}'.format(
#         len(ds), K, batch, focal))
#     assert isinstance(ds, data.TensorDataset)
#     size = len(ds) // K
#     size_list = [size] * K
#     size_list[0] += len(ds) % K
#     folds = data.random_split(ds, size_list, torch.Generator().manual_seed(59))
#     import pandas as pd
#     result = pd.DataFrame(
#         columns=['loss_f', 'loss', 'tp', 'tn', 'fp', 'fn', 'plot'])
#     for i in range(K):
#         print('=======> Fold '+str(i)+' <========')
#         train = None
#         for j in range(K):
#             if j != i:
#                 train = folds[j] if train is None else train+folds[j]
#         train_loader = data.DataLoader(train, batch_size=batch, num_workers=4)
#         val_loader = data.DataLoader(folds[i], batch_size=batch, num_workers=4)
#         trainer = Trainer(train_loader, val_loader,
#                           learning_rate=learning_rate, focal=focal, verbose=True)
#         print('dataloader and trainer created, start fitting')
#         plot_path = trainer.fit()
#         print('start evaluate')
#         eval_res = trainer.eval()
#         eval_res['plot'] = plot_path
#         result.append(eval_res, ignore_index=True)
#     filename = 'output/'+datetime.now().strftime('%d-%H_%M_%S') + '-repost.csv'
#     result.to_csv(filename)
#     return result
