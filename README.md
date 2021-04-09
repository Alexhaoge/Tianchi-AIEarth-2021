# Tianchi-AIEarth-2021
[Tianchi-AIEarth-2021](https://tianchi.aliyun.com/competition/entrance/531871/information)  
Alibaba Tianchi AIEarth 2021 match submission source repo for team *这是个队名*. 
Tried models: simple CNN, ConvLSTM
Quit the match due the team are otherwise occupied, so honestly speaking the result is far from satisfaction.
## Install locally
1. clone repository
2. create directories as below
```
Repo root
|--model
|--notebook
|--output
|--result
|--tcdata
   |--enso_round1_train_20210201
       |--CMIP_label.nc
       |--CMIP_train.nc
       |--readme.txt
       |--SODA_label.nc
       |--SODA_train.nc
   |--enso_round1_test_20210201
​       |--test_00001_07_06.npy
       |--test_00014_02_01.npy
```
3. Download dataset
https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531871/enso_round1_train_20210201.zip
4. install dependencies
```shell
conda create -n aiearth pytorch=1.4 cudatoolkit=10.1 xarray netCDF4 numpy pandas
conda activate aiearth
# conda deactivate
```
5. debug & run
```
python main.py --debug --small-dataset 500
```
for more arguments, see `python main.py --help`

## Docker Images
This repo is binded with Alicloud, every commit of main branch and release will be built automatically.
Image repo: https://cr.console.aliyun.com/repository/cn-shenzhen/alexhaoge/aiearth/details
