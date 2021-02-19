# Tianchi-AIEarth-2021
Tianchi-AIEarth-2021

## 本地部署
1. clone repository
2. 仓库路径下创建文件夹
```
仓库根目录
|--其他文件
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
3. 下载数据集
https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531871/enso_round1_train_20210201.zip
4. 安装Python库
```shell
conda create -n aiearth pytorch=1.4 cudatoolkit=10.1 xarray netCDF4 numpy pandas
conda activate aiearth
#退出 conda deactivate
```
5. 调试&运行
```
python main.py --debug --small-dataset
```
参数格式参考`python main.py --help`

## 工作
### 特征工程
1. 变量取值范围 Scale
2. 指标变化趋势
#### 把时间拼起来
[Xarray的文档](http://xarray.pydata.org/en/stable/)
#### 研究物理意义
看这几个变量是啥玩意
https://www.yuque.com/eatcosmos/ifggyy/nx6g61
特征运算
#### 数据分析
看周期性
画特征-标签散点图/相关系数，看相关性
SST在陆地上的取值是啥
### 差分

### 模型 
对于这个12->24时间序列，更适合Encoder-Decoder/Transformer
1. baseline
    - 多变量回归
    - Dense
2. RNN
    - CNN+LSTM+Linear
    - convLSTM/trajGRU/predRNN
    - social + convLSTM
    RNN最大的问题是输入timestep < 预测timestep
        - 自欺欺人：把输入复制一遍
        - 把LSTM当成encoder-decoder
3. Encoder-Decoder
    - Seq2Seq / stconvs2s
4. Transformer
    - Axial Transformer
5. GCN
    - 找个合适的GCN基础模型
    - ASTGCN
6. CNN
纯CNN可以做，把不同时间看成多个通道
    - https://doi.org/10.1038/s41586-019-1559-7
    - U-net

### 数据集
官方交流群里（好像是出题人）的站着说话不腰疼的提示:
> 有几点可能大家可以关注下：1、虽然测试条件给的是过去12个月数据，但是模型最佳输入时长是多少？2、测试集中起始月份是随机的，相应训练数据制备需要注意；3，CMIP数据和观测数据分布还是不一样的，如何利用CMIP数据？观测数据量比较少，合理的数据增强策略；4、预测目标是enso3.4指数，但enso3.4指数是有明确物理含义的，在模型训练时是不是要考虑到这一点

1. 看周期性了
2. 按月做起始点 4546*12
3. 训练顺序
    - 最终的模型可以先用CMIP，SODA retrain 学习率不同
    - 问题是真实数据不见得对模型训练有帮助
4. 特征工程