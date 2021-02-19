# -*- coding: utf-8 -*-
import torch
import os


def get_filelist(dir, Filelist=[]):
    if os.path.isfile(dir):
        Filelist.append(dir)
    return Filelist


class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience
    """

    def __init__(self, verbose: bool = False, patience: int = 16, no_stop: bool = False):
        self.verbose = verbose
        self.patience = patience
        self.best_loss = float('inf')
        self.counter = 0
        self.isToStop = False
        self.enable_stop = not no_stop

    def __call__(self, val_loss, model, optimizer, epoch, filename):
        is_best = bool(val_loss < self.best_loss)
        if is_best:
            self.best_loss = val_loss
            self.__save_checkpoint(model, filename)
            if self.verbose:
                print(filename)
            self.counter = 0
        elif self.enable_stop:
            self.counter += 1
            if self.verbose:
                print(
                    f'=> Early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.isToStop = True

    def __save_checkpoint(self, model, filename):
        torch.save(model.state_dict(), filename)
        if self.verbose:
            print('=> Saving a new best')


# def plot(data, columns_name, x_label, y_label, title, inline=False):
#     from matplotlib.ticker import MaxNLocator
#     import pandas as pd
#     from hashlib import md5
#     import time
#     df = pd.DataFrame(data).T
#     df.columns = columns_name
#     df.index += 1
#     plot = df.plot(linewidth=2, figsize=(15, 8), color=[
#                    'darkgreen', 'orange'], grid=True)
#     train = columns_name[0]
#     val = columns_name[1]
#     # find position of lowest validation loss
#     idx_min_loss = df[val].idxmin()
#     plot.axvline(idx_min_loss, linestyle='--', color='r', label='Best epoch')
#     plot.legend()
#     plot.set_xlim(0, len(df.index)+1)
#     plot.xaxis.set_major_locator(MaxNLocator(integer=True))
#     plot.set_xlabel(x_label, fontsize=12)
#     plot.set_ylabel(y_label, fontsize=12)
#     plot.set_title(title, fontsize=16)
#     if not inline:
#         m = md5()
#         m.update(str(time.time()).encode('utf-8'))
#         filename = 'output/plot/' + m.hexdigest() + '.png'
#         plot.figure.savefig(filename, bbox_inches='tight')
#         return filename
