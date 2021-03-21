import os
import torch
import numpy as np
import xarray as xr
import pandas as pd
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader

#设置种子
def set_seed(seed = 427):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)

#数据导入
def split_month(array,size):#input shape: :,36,24,72
    temp=array[:size,0:12,:,:]
    temp=temp.reshape(size*12,24,72)
    temp2=np.array([temp[i:i+12,:,:] for i in range(size*12-40)])
    return temp2

def split_month_label(array,size):#input shape: :,24
    temp=array[:size,0:12]
    temp=temp.reshape(size*12)
    temp2=np.array([temp[i+12:i+36] for i in range(size*12-40)])
    return temp2
    
def load_data2():
    # CMIP data    
    size1=3000
    train = xr.open_dataset('../tcdata/enso_round1_train_20210201/CMIP_train.nc')
    label = xr.open_dataset('../tcdata/enso_round1_train_20210201/CMIP_label.nc')    
    train_sst = train['sst'].values
    train_sst= np.concatenate((train_sst[:151*5],train_sst[151*9:151*12],train_sst[151*13:]))  
    train_sst=split_month(train_sst,size1)
    train_t300 = train['t300'].values
    train_t300= np.concatenate((train_t300[:151*5],train_t300[151*9:151*12],train_t300[151*13:]))
    train_t300=split_month(train_t300,size1)
    train_ua = train['ua'].values
    train_ua= np.concatenate((train_ua[:151*5],train_ua[151*9:151*12],train_ua[151*13:])) 
    train_ua=split_month(train_ua,size1)
    train_va = train['va'].values
    train_va= np.concatenate((train_va[:151*5],train_va[151*9:151*12],train_va[151*13:]))
    train_va=split_month(train_va,size1)
    train_label = label['nino'].values
    train_label= np.concatenate((train_label[:151*5],train_label[151*9:151*12],train_label[151*13:]))
    train_label=split_month_label(train_label,size1)
    
    #train_ua = np.nan_to_num(train_ua)#缺失值补0
    #train_va = np.nan_to_num(train_va)
    #train_t300 = np.nan_to_num(train_t300)
    #train_sst = np.nan_to_num(train_sst)

    # SODA data  
    size2=100
    train2 = xr.open_dataset('../tcdata/enso_round1_train_20210201/SODA_train.nc')
    label2 = xr.open_dataset('../tcdata/enso_round1_train_20210201/SODA_label.nc')
    
    train_sst2 = train2['sst'].values  # (3890, 12, 24, 72)
    train_sst2=split_month(train_sst2,size2)
    train_t3002 = train2['t300'].values
    train_t3002=split_month(train_t3002,size2)
    train_ua2 = train2['ua'].values
    train_ua2=split_month(train_ua2,size2)
    train_va2 = train2['va'].values
    train_va2=split_month(train_va2,size2)
    train_label2 = label2['nino'].values
    train_label2=split_month_label(train_label2,size2)

    print('Train samples: {}, Valid samples: {}'.format(len(train_label), len(train_label2)))

    dict_train = {
        'sst':train_sst,
        't300':train_t300,
        'ua':train_ua,
        'va': train_va,
        'label': train_label}
    dict_valid = {
        'sst':train_sst2,
        't300':train_t3002,
        'ua':train_ua2,
        'va': train_va2,
        'label': train_label2}
    train_dataset = EarthDataSet(dict_train)
    valid_dataset = EarthDataSet(dict_valid)
    return train_dataset, valid_dataset

class EarthDataSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data['sst'])

    def __getitem__(self, idx):   
        return (self.data['sst'][idx], self.data['t300'][idx], self.data['ua'][idx], self.data['va'][idx]), self.data['label'][idx]

#模型
class simpleSpatailTimeNN(nn.Module):
    def __init__(self, n_cnn_layer:int=1, kernals:list=[3], n_lstm_units:int=64):
        super(simpleSpatailTimeNN, self).__init__()
        self.conv1 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals]) 
        self.conv2 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv3 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.conv4 = nn.ModuleList([nn.Conv2d(in_channels=12, out_channels=12, kernel_size=i) for i in kernals])
        self.pool1 = nn.AdaptiveAvgPool2d((22, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((1, 70))
        self.pool3 = nn.AdaptiveAvgPool2d((1, 128))
        self.batch_norm = nn.BatchNorm1d(12, affine=False)
        self.lstm = nn.LSTM(1540 * 4, n_lstm_units, 2, bidirectional=True)
        self.linear = nn.Linear(128, 24)

    def forward(self, sst, t300, ua, va):
        for conv1 in self.conv1:
            sst = conv1(sst)  # batch * 12 * (24 - 2) * (72 -2)
        for conv2 in self.conv2:
            t300 = conv2(t300)
        for conv3 in self.conv3:
            ua = conv3(ua)
        for conv4 in self.conv4:
            va = conv4(va)

        sst = torch.flatten(sst, start_dim=2)  # batch * 12 * 1540
        t300 = torch.flatten(t300, start_dim=2)
        ua = torch.flatten(ua, start_dim=2)
        va = torch.flatten(va, start_dim=2)  # if flat, lstm input_dims = 1540 * 4              
            
        x = torch.cat([sst, t300, ua, va], dim=-1) #在内层合并 batch*12*1540*4
        x = self.batch_norm(x)
        x, _ = self.lstm(x)#输入1540*4,64hidden,2layer且双向 输出batch*(128=64*2)
        x = self.pool3(x).squeeze(dim=-2)
        x = self.linear(x)#128-->24
        return x

#loss function
def coreff(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    c1 = sum((x - x_mean) * (y - y_mean))
    c2 = sum((x - x_mean)**2) * sum((y - y_mean)**2)
    return c1/np.sqrt(c2)

def rmse(preds, y):
    return np.sqrt(sum((preds - y)**2)/preds.shape[0])

def eval_score(preds, label):
    # preds = preds.cpu().detach().numpy().squeeze()
    # label = label.cpu().detach().numpy().squeeze()
    acskill = 0
    RMSE = 0
    a = 0
    a = [1.5]*4 + [2]*7 + [3]*7 + [4]*6
    for i in range(24):
        RMSE += rmse(label[:, i], preds[:, i])
        cor = coreff(label[:, i], preds[:, i])
    
        acskill += a[i] * np.log(i+1) * cor
    return 2/3 * acskill - RMSE

#train
fit_params = {
    'n_epochs' : 22,
    'learning_rate' : 8e-5,
    'batch_size' : 64,
}

#下面的和原来有更改
def train(model):
    set_seed()
    train_dataset, valid_dataset = load_data2()      
    train_loader = DataLoader(train_dataset, batch_size=fit_params['batch_size'])
    valid_loader = DataLoader(valid_dataset, batch_size=fit_params['batch_size'])
    #model = simpleSpatailTimeNN()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'   
    optimizer = torch.optim.Adam(model.parameters(), lr=fit_params['learning_rate'])
    loss_fn = nn.MSELoss()   
    
    model.to(device)
    loss_fn.to(device)

    for i in range(fit_params['n_epochs']):
        model.train()
        for step, ((sst, t300, ua, va), label) in enumerate(train_loader):                
            sst = sst.to(device).float()
            t300 = t300.to(device).float()
            ua = ua.to(device).float()
            va = va.to(device).float()
            optimizer.zero_grad()
            label = label.to(device).float()
            preds = model(sst, t300, ua, va)
            loss = loss_fn(preds, label)
            loss.backward()
            optimizer.step()
    return model

model= simpleSpatailTimeNN()
model_trained=train(model)
def predict_test(x):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_sst = x[:,:,:,0] 
    test_sst=torch.tensor(test_sst.reshape(1,12,24,72))
    test_t300 = x[:,:,:,1] 
    test_t300=torch.tensor(test_t300.reshape(1,12,24,72))
    test_ua = x[:,:,:,3] 
    test_ua=torch.tensor(test_ua.reshape(1,12,24,72))
    test_va = x[:,:,:,0] 
    test_va=torch.tensor(test_va.reshape(1,12,24,72))
    sst = test_sst.to(device).float()
    t300 = test_t300.to(device).float()
    ua = test_ua.to(device).float()
    va = test_va.to(device).float()
    preds = model_trained(sst, t300, ua, va)
    return preds.view(-1).cuda().data.cpu().numpy()#要写这些把GPU上tensor转到CPU numpy上

test_dir = 'tcdata/enso_round1_test_20210201'
print('Predict from ' + os.getcwd())
for i in os.listdir(test_dir):
    if i.endswith('.npy'):
        x = np.load(test_dir + '/' + i)
        np.save('result/' + i, predict_test(x))