import os
import warnings
warnings.filterwarnings("ignore")  # avoid printing out absolute paths避免打印出绝对路径
import copy
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.data.examples import get_stallion_data
import csv
import pickle
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

#<a href="https://colab.research.google.com/github/pgpanagiotidis/Temporal-Fusion-Transformers-for-stock-price-forecasting/blob/main/Pfizer_stock_closing_price_forecasting_model.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

#read the data
df_Pfizer = pd.read_csv('Pfizer.csv')
#convert column to datetime转变成datatime
df_Pfizer["Date"]= pd.to_datetime(df_Pfizer["Date"])
#print(df_Pfizer["Date"])   #Name: Date, Length: 1520, dtype: datetime64[ns]

#add time index时间索引
time_idx=np.arange(1520)
df_Pfizer["time_idx"] = time_idx

#create the group column
df_Pfizer['group']=[0 for i in range(0,1520)]   #循环这么多个0 print(df_Pfizer['group'])

# keep only the necessary columns  只保持close列
df_Pfizer.drop(['Open', 'High', 'Low','Adj Close', 'Volume'], axis=1, inplace=True)
#print(df_Pfizer)  #只有date close time_idx  group

# split the dataset into train, validation确认 and test set
max_prediction_length =3    #最大预测长度
max_encoder_length = 14     #最大编码长度

training_cutoff = int(df_Pfizer["time_idx"].max() * 0.6)    #1519*0.6=911
print("training_cutoff", training_cutoff)
#一个对象   <pytorch_forecasting.data.timeseries.TimeSeriesDataSet object at 0x000001EBB49070F0>拟合时间序列模型
training = TimeSeriesDataSet(
    df_Pfizer[lambda x: x.time_idx <= training_cutoff],#<=911的是训练集
    group_ids=["group"],
    target='Close',
    time_idx="time_idx",
    min_encoder_length=max_encoder_length,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=['Close'],
    time_varying_known_reals=["time_idx"],
    target_normalizer=GroupNormalizer(
        groups=["group"], transformation="softplus"
    ),
    # use softplus and normalize by group,
    # SoftPlus is a smooth approximation to the ReLU function and can be used to constrain the output of a machine to always be positive.
    add_relative_time_idx=True,
    add_target_scales=True,
    # add_encoder_length=True,
)

validation_cutoff = training_cutoff + int(df_Pfizer["time_idx"].max() * 0.2)  # 1214"YYYY-MM-DD"  # day for cutoff
print("validation_cutoff", validation_cutoff)

# create validation dataset using the same normalization techniques as for the training dataset
validation = TimeSeriesDataSet.from_dataset(training, df_Pfizer[
    lambda x: (x.time_idx > training_cutoff) & (x.time_idx <= validation_cutoff)], stop_randomization=True) #911< <1214

# create test dataset using the same normalization techniques as for the training dataset
test = TimeSeriesDataSet.from_dataset(training, df_Pfizer[lambda x: x.time_idx > validation_cutoff],#>1214
                                      stop_randomization=True)
print(test)
BATCH_SIZE = 128
# convert datasets to dataloaders for training
train_dataloader = training.to_dataloader(train=True, batch_size=BATCH_SIZE, shuffle=False)
val_dataloader = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, shuffle=False)
test_dataloader = test.to_dataloader(train=False, batch_size=BATCH_SIZE, shuffle=False)
# create validation set (predict=True) which means to predict the last max_prediction_length points in time
# for each series
validation = TimeSeriesDataSet.from_dataset(training, df_Pfizer, predict=False, stop_randomization=True)#生成具有不同基础数据但变量编码器和缩放器相同的数据集,如果要停止随机化编码器和解码器长度

x, y = next(iter(val_dataloader))
print("x =", x)
print("\ny =", y)
print("\nsizes of x =")
for key, value in x.items():
    print(f"\t{key} = {value.size()}")

# configure network and trainer配置网络和培训器
pl.seed_everything(42)  #设置伪随机数生成器种子的函数
trainer = pl.Trainer(
    gpus=0,
    # clipping gradients is a hyperparameter and important to prevent divergance of the gradient for recurrent neural networks
    # 削波梯度是一个超参数，对防止分裂很重要
    #递归神经网络的梯度
    gradient_clip_val=0.1,
)


tft = TemporalFusionTransformer.from_dataset(           #Create model from dataset.
    training,
    # not meaningful for finding the learning rate but otherwise very important
    learning_rate=0.03,
    hidden_size=16,  # most important hyperparameter apart from learning rate除了学习率之外最重要的超参数
    # number of attention heads. Set to up to 4 for large datasets ---的数量。对于大型数据集，设置为最多4个
    attention_head_size=4,
    dropout=0.1,  # between 0.1 and 0.3 are good values
    hidden_continuous_size=8,  # set to <= hidden_size
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    # reduce learning rate if no improvement in validation loss after x epochs如果在x个时期后验证没有改善，则降低学习率
    reduce_on_plateau_patience=4,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find optimal最佳的 learning rate
res = trainer.tuner.lr_find(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
    max_lr=10.0,
    min_lr=1e-6,
)

#print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

# configure配置 network and trainer
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
#变体支持同时保存和恢复多个早期停止回调

lr_logger = LearningRateMonitor()  # log the learning rate记录学习率

logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard 将结果记录到tensorboard
print(logger)
trainer = pl.Trainer(
    max_epochs=300,
    gpus=0,
    weights_summary="top",
    gradient_clip_val=0.1,
    limit_train_batches=30,  # coment in for training, running valiation every 30 batches参加培训，每30批次进行一次验证
    # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
    callbacks=[lr_logger, early_stop_callback],
    logger=logger,
)


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.001,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,  # 7 quantiles by default
    loss=QuantileLoss(),
    log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
    reduce_on_plateau_patience=4,
)

# fit network拟合网络
trainer.fit(
    tft,
    train_dataloader=train_dataloader,
    val_dataloaders=val_dataloader,
)

# create study
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=100,
    max_epochs=20,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(16, 128),
    hidden_continuous_size_range=(8, 64),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.0000001, 0.25),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=1.0),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
)

# save study results - also we can resume tuning at a later point in time
with open("test_study.pkl", "wb") as fout:
    #将python项目过程中用到的一些暂时变量、或者需要提取、暂存的字符串、列表、字典等数据保存起来。
    # #3）保存方式就是保存到创建的.pkl文件里面。
    # #4）然后需要使用的时候再 open，load。
    pickle.dump(study, fout)    #将对象obj保存到文件file中去。

# show best hyperparameters
print(study.best_trial.params)

# load the best model according to the validation loss
# (given that we use early stopping, this is not necessarily the last epoch)
best_model_path = trainer.checkpoint_callback.best_model_path
best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

# calcualte mean absolute error on test set   计算测试集的平均绝对误差
actuals = torch.cat([y[0] for x, y in iter(test_dataloader)])
predictions = best_tft.predict(test_dataloader)

(actuals - predictions).abs().mean()

# raw predictions are a dictionary from which all kind of information including quantiles can be extracted
#原始预测是一本字典，可以从中提取包括分位数在内的所有信息
raw_predictions, x = best_tft.predict(test_dataloader, mode="raw", return_x=True)

#display the best predictions
predictions = best_tft.predict(test_dataloader)
mean_losses = SMAPE(reduction="none")(predictions, actuals).mean(1)  #对称平均绝对百分比
indices = mean_losses.argsort(descending=False)  # sort losses

for idx in range(10):  # plot 10 examples
    best_tft.plot_prediction(
        x, raw_predictions, idx=indices[idx], add_loss_to_title=SMAPE(quantiles=best_tft.loss.quantiles)
    )