# Tianchi-AIEarth-2021
[Tianchi-AIEarth-2021](https://tianchi.aliyun.com/competition/entrance/531871/information)
## 本地部署
1. clone repository
2. 仓库路径下创建文件夹
```
仓库根目录
|--model 模型代码
|--notebook notebook实验文件夹
|--output 保存的模型checkpoints/中间结果
|--result 预测结果
|--tcdata 数据
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
python main.py --debug --small-dataset 500
```
参数格式参考`python main.py --help`

## 镜像提交
1. github代码仓库通过webhook连接到阿里云镜像仓库，配置好规则之后release和主分支commit都会自动构建  
镜像仓库地址: https://cr.console.aliyun.com/repository/cn-shenzhen/alexhaoge/aiearth/details
2. 给队友开通阿里云RAM子账号，授予个人的镜像仓库管理权限
3. 比赛界面提交
registry.cn-shenzhen.aliyuncs.com/alexhaoge/aiearth:[镜像版本号]

## 工作
### 目前问题
1. 小数据集出了结果是负分，大数据集本地能看到loss，但是提交后得到评分NaN
    - 据说训练集里有数据NaN，需要排查（SODA里没有）
    - 梯度爆炸了？但是本地大数据集训练没有出现NaN，可能看一下验证集的具体结果？
### 特征工程
1. 变量取值范围 Scale
2. 指标变化趋势
#### 把时间拼起来
拼成连续的150年，方便分析，以及之后分割成逐月起始的数据
[Xarray的文档](http://xarray.pydata.org/en/stable/)
#### 研究物理意义
1. 看这几个变量是啥玩意
https://www.yuque.com/eatcosmos/ifggyy/nx6g61
2. 特征运算
3. **Nature上用CNN做Nino3.4预测的论文，估计就是这题原出处，metrics都差不多**
https://doi.org/10.1038/s41586-019-1559-7
#### 数据分析
看周期性
画特征-标签散点图/相关系数，看相关性
SST在陆地上的取值是啥
### 差分

### 模型 
对于这个12->24时间序列，更适合Encoder-Decoder/Transformer
1. baseline
    - 多变量回归？
    - Dense
2. RNN
    - CNN+LSTM+Linear
    - convLSTM/trajGRU/predRNN
    - social + convLSTM
    RNN最大的问题是输入timestep < 预测timestep
        - 自欺欺人：把输入复制一遍
        - **把LSTM当成encoder**
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
**估计模型之间区别不是很大，更大的问题是数据处理和训练方式**
### 数据集
官方交流群里（好像是出题人）的站着说话不腰疼的提示:
> 有几点可能大家可以关注下：
> 1. 虽然测试条件给的是过去12个月数据，但是模型最佳输入时长是多少？
> 2. 测试集中起始月份是随机的，相应训练数据制备需要注意；
> 3. CMIP数据和观测数据分布还是不一样的，如何利用CMIP数据？观测数据量比较少，合理的数据增强策略；
> 4. 预测目标是enso3.4指数，但enso3.4指数是有明确物理含义的，在模型训练时是不是要考虑到这一点

1. 看周期性了
2. 按月做起始点 4546*12
3. 训练顺序
    - 最终的模型可以先用CMIP，SODA retrain 学习率不同
    - 问题是真实数据不见得对模型训练有帮助
4. 特征工程，之后看相关论文瞎搞了