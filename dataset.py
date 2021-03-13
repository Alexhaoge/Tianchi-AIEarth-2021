import xarray as xr
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset


def NCDataset(Dataset):
    
    def __init__(self, X: xr.Dataset, y: xr.Dataset, index_col: str = 'ym'):
        """
        Label should have 24 months more than features,
        but they have the same start time.
        """
        assert X.sizes[index_col] + 24 == y.sizes[index_col]
        super(NCDataset, self).__init__()
        self.X = X
        self.y = y
        self.index_col = index_col

    def __len__(self):
        return self.X.sizes[self.index_col] - 11

    def __getitem__(self, index):
        arg_map = {self.index_col: list(range(index,index+12))}
        label_arg_map = {self.index_col: list(range(index+12, index+36))}
        return torch.Tensor(self.X.isel(**arg_map).values).permute(3,1,2,0),\
            torch.Tensor(self.y.isel(**arg_map).values).permute(3,1,2,0)


def get_dataset_old(
    name:str, 
    debug_mode: bool = False, 
    small: int = -1,
    fillna: int = True
) -> TensorDataset:
    """
    NOTE:
        The entire dataset input is in shape 
        [number(year), month(36), 4, lat(24), lon(72)],
        label in shape [number(year), month(36)]
        Variable order in last dim:
        0: SST, 1: T300, 2: Ua, 3: Va

    Args:
        name: Dataset name, must be one of 
            ['cmip', 'cmip5', 'cmip6', 'soda']
        debug_model: if debug mode is on, the path for dataset differs
        small: random sample and create a tiny dataset,
            helpful for local debugging. Default -1 means using the whole dataset
    
    Returns:
        torch.utils.data.Dataset
    """
    assert name in ['cmip', 'cmip5', 'cmip6', 'soda']
    if name == 'soda':
        train_path = 'tcdata/enso_round1_train_20210201/SODA_train.nc'
        label_path = 'tcdata/enso_round1_train_20210201/SODA_label.nc'
    else:
        train_path = 'tcdata/enso_round1_train_20210201/CMIP_train.nc'
        label_path = 'tcdata/enso_round1_train_20210201/CMIP_label.nc'
    if not debug_mode:
        train_path = '/' + train_path
        label_path = '/' + label_path
    train = xr.open_dataset(train_path)
    label = xr.open_dataset(label_path)
    if name == 'cmip5':
        train = train[dict(year=slice(0, 2265))]
        label = label[dict(year=slice(0, 2265))]
    elif name == 'cmip6':
        train = train[dict(year=slice(2265, 4645))]
        label = label[dict(year=slice(2265, 4645))]
    if small > 0:
        from random import sample
        sample_list = sample(np.arange(train.sizes['year']).tolist(), small)
        train = train[dict(year=sample_list)]
        label = label[dict(year=sample_list)]
    if fillna:
        train = train.fillna(0)
    return  TensorDataset(
        torch.Tensor(train.to_array().data).permute(1,2,0,3,4),
        torch.Tensor(label.to_array().data).squeeze()
    )


def comb_dropna():
    pass