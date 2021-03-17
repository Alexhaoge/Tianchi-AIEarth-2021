import xarray as xr
import numpy as np
import torch
from torch.utils.data import TensorDataset, Dataset, ConcatDataset


class NCDataset(Dataset):
    
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
        return \
            torch.Tensor(
                self.X.isel(
                    **{self.index_col: list(range(index,index+12))}
                ).values
            ).permute(3,1,2,0),\
            torch.Tensor(
                self.y.isel(
                    **{self.index_col: list(range(index+12, index+36))}
                ).values
            ).permute(3,1,2,0)


def NCDSFactory(
    raw_feature: xr.Dataset,
    raw_label: xr.Dataset,
    index_col: str = 'ym'
    ) -> NCDataset:
    """
    NC Dataset Factory
    """
    from gc import collect
    X = raw_feature.isel(month=list(range(12)))\
                .stack(**{index_col:('year', 'month')})
    del raw_feature
    collect()
    y = raw_label.isel(month=list(range(12)))\
                .stack(**{index_col:('year', 'month')})
    y_end = raw_label.tail(year=1, month=24)\
                .stack(**{index_col:('year', 'month')})
    y = y.merge(y_end)
    del raw_label
    collect()
    return NCDataset(X, y, index_col)


def get_dataset_new(
    name: str,
    debug_mode: bool = False,
    small: int = -1
) -> Dataset:
    """
    Args:
        name: Dataset name, must be one of 
            ['cmip', 'cmip5', 'cmip6', 'soda']
        debug_model: if debug mode is on, the path for dataset differs
        small: get only first nth year from each model,
            helpful for local debugging. Default -1 means using the whole dataset
    
    Returns:
        torch.utils.data.ConcatDataset
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
    if name == 'soda':
        return NCDSFactory(train, label)
    ds_list = []
    if name != 'cmip6':
        start_i = 2265
        for i in range(17):
            len = 140 if small == -1 else small
            i_list = list(range(start_i, start_i+len))
            start_i += 140
            ds_list.append(NCDSFactory(
                train.isel(year=i_list),
                label.isel(year=i_list)
            ))
    if name != 'cmip5':
        start_i = 0
        for i in range(15):
            if i in [6,7,8,9,13]:
                start_i += 151
                continue
            len = 151 if small == -1 else small
            i_list = list(range(start_i, start_i+len))
            start_i += 151
            ds_list.append(NCDSFactory(
                train.isel(year=i_list),
                label.isel(year=i_list)
            ))
    return ConcatDataset(ds_list)


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

    # train_sst = train['sst'][:, :12].values  # (4645, 12, 24, 72)截取前12项
    # train_t300 = train['t300'][:, :12].values
    # train_ua = train['ua'][:, :12].values
    # train_va = train['va'][:, :12].values
    # train_label = label['nino'][:, 12:36].values
    
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
