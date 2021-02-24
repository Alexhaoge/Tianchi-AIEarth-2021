import pandas as pd
import numpy as np
import netCDF4 as nc
import os
import xarray as xr
import datetime

class dataset_reshape:
    def __init__(self,path_Cl,path_Ct,path_Sl,path_St):
        self.path_Cl = path_Cl
        self.path_Ct= path_Ct
        self.path_Sl=path_Sl
        self.path_St=path_St
    def trans(x):
        temp=10000*(6-(x>2265))+(x<2266)*((x-1)//151+1)+(x>2265)*(((x-1)-2265)//140+1)#这里返回str报错
        #<2266:CMIP6 >2265:CMIP5
        #编码顺序开始5，6，1表示CMIP5,6,SODA 后面表示模式 从1计数 
        return temp
    def transyear(x):#因为遍历规则，不能对X比较大小 报错ambiguous
        temp=(x<2266)*((x-1)%151+1)+(x>2265)*((x-1-2265)%140+1)#年份的顺序，计数从1开始
        return temp
    def data_set(self):
        A=nc.Dataset(self.path_Cl)
        A= xr.open_dataset(xr.backends.NetCDF4DataStore(A))
        B=nc.Dataset(self.path_Ct)
        B= xr.open_dataset(xr.backends.NetCDF4DataStore(B))
        CMIP=B
        CMIP['nino']=(('year','month'),A['nino'])
        CMIP=CMIP.drop_sel(month=np.arange(13,37))
        CMIPdf=CMIP.to_dataframe()
        CMIPdf=CMIPdf.reset_index()
        CMIPdf['mode']=CMIPdf['year'].apply(dataset_reshape.trans)
        CMIPdf['year']=CMIPdf['year'].apply(dataset_reshape.transyear)
        CMIPdf['time']=CMIPdf.year*12+CMIPdf.month-12
        C=nc.Dataset(self.path_St)
        C= xr.open_dataset(xr.backends.NetCDF4DataStore(C))
        D=nc.Dataset(self.path_Sl)
        D= xr.open_dataset(xr.backends.NetCDF4DataStore(D))
        SODA=C
        SODA['nino']=(('year','month'),D.nino)
        SODA=SODA.drop_sel(month=np.arange(13,37))
        SODAdf=SODA.to_dataframe()
        SODAdf=SODAdf.reset_index()
        SODAdf['mode']=10000
        SODAdf['time']=SODAdf.year*12+SODAdf.month-12
        data=pd.concat([CMIPdf, SODAdf])
        data_reindex=data.set_index(['lat', 'lon','month','year','mode'])
        data_set=xr.Dataset.from_dataframe(data_reindex)
        return(data_set)
    
if __name__ == "__main__":
    train_dir = 'tcdata/enso_round1_train_20210201'
    path_train=[]
    for i in os.listdir(train_dir):
        if i.endswith('.nc'):path_train.append(train_dir + '/' + i)
    data=dataset_reshape(path_train[0],path_train[1],path_train[2],path_train[3])