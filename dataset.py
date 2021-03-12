import xarray as xr
import numpy as np

import torch
from torch.utils.data import TensorDataset

def get_dataset(
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

    train_sst = train['sst'][:, :12].values  # (4645, 12, 24, 72)截取前12项
    train_t300 = train['t300'][:, :12].values
    train_ua = train['ua'][:, :12].values
    train_va = train['va'][:, :12].values
    train_label = label['nino'][:, 12:36].values
    
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
