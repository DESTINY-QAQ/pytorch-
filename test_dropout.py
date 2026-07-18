<<<<<<< HEAD
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ---------- 从原始来源加载波士顿数据集 ----------
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = data.astype(np.float32)
y = target.astype(np.float32)
# ------------------------------------------------

dim = X.shape[1]  # 13个特征

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
num_train = X_train.shape[0]

# 标准化
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train -= mean
X_train /= std
X_test -= mean
X_test /= std

# 转换为 PyTorch 张量（此时还在 CPU）
train_data = torch.from_numpy(X_train).float()
train_target = torch.from_numpy(y_train).float()
test_data = torch.from_numpy(X_test).float()
test_target = torch.from_numpy(y_test).float()

# ---------- 定义设备 ----------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 将数据移动到设备（一次性全部移入，因为数据量不大）
train_data = train_data.to(device)
train_target = train_target.to(device)
test_data = test_data.to(device)
test_target = test_target.to(device)

# 定义模型（保持原来的13维输入）
net1_overfitting = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
).to(device)   # 模型移动到设备

net1_dropped = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
).to(device)

net1_nb = torch.nn.Sequential(
    torch.nn.Linear(13, 8),
    nn.BatchNorm1d(8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 4),
    nn.BatchNorm1d(4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1),
).to(device)

# 损失函数和优化器
loss_func = torch.nn.MSELoss()
optimizer_ofit = torch.optim.Adam(net1_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net1_dropped.parameters(), lr=0.01)
optimizer_nb = torch.optim.Adam(net1_nb.parameters(), lr=0.01)

# TensorBoard 记录器
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='logs')

# 训练循环
for epoch in range(200):
    net1_overfitting.train()
    net1_dropped.train()
    net1_nb.train()

    pred_ofit = net1_overfitting(train_data)
    pred_drop = net1_dropped(train_data)
    pred_nb = net1_nb(train_data)

    loss_ofit = loss_func(pred_ofit, train_target)
    loss_drop = loss_func(pred_drop, train_target)
    loss_nb = loss_func(pred_nb, train_target)

    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    optimizer_nb.zero_grad()

    loss_ofit.backward()
    loss_drop.backward()
    loss_nb.backward()

    optimizer_ofit.step()
    optimizer_drop.step()
    optimizer_nb.step()

    # 记录训练损失（包含三个模型）
    writer.add_scalars('train_group_loss',
                       {'overfitting': loss_ofit.item(),
                        'dropout': loss_drop.item(),
                        'batchnorm': loss_nb.item()},
                       epoch)

    # 切换到评估模式
    net1_overfitting.eval()
    net1_dropped.eval()
    net1_nb.eval()

    test_pred_orig = net1_overfitting(test_data)
    test_pred_drop = net1_dropped(test_data)
    test_pred_nb = net1_nb(test_data)

    orig_loss = loss_func(test_pred_orig, test_target)
    drop_loss = loss_func(test_pred_drop, test_target)
    nb_loss = loss_func(test_pred_nb, test_target)

    writer.add_scalars('test_group_loss',
                       {'overfitting': orig_loss.item(),
                        'dropout': drop_loss.item(),
                        'batchnorm': nb_loss.item()},
                       epoch)

writer.close()
=======
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# ---------- 从原始来源加载波士顿数据集 ----------
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

X = data.astype(np.float32)
y = target.astype(np.float32)
# ------------------------------------------------

dim = X.shape[1]  # 13个特征

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
num_train = X_train.shape[0]

# 标准化
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)
X_train -= mean
X_train /= std
X_test -= mean
X_test /= std

# 转换为 PyTorch 张量（此时还在 CPU）
train_data = torch.from_numpy(X_train).float()
train_target = torch.from_numpy(y_train).float()
test_data = torch.from_numpy(X_test).float()
test_target = torch.from_numpy(y_test).float()

# ---------- 定义设备 ----------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 将数据移动到设备（一次性全部移入，因为数据量不大）
train_data = train_data.to(device)
train_target = train_target.to(device)
test_data = test_data.to(device)
test_target = test_target.to(device)

# 定义模型（保持原来的13维输入）
net1_overfitting = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
).to(device)   # 模型移动到设备

net1_dropped = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
).to(device)

net1_nb = torch.nn.Sequential(
    torch.nn.Linear(13, 8),
    nn.BatchNorm1d(8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 4),
    nn.BatchNorm1d(4),
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1),
).to(device)

# 损失函数和优化器
loss_func = torch.nn.MSELoss()
optimizer_ofit = torch.optim.Adam(net1_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net1_dropped.parameters(), lr=0.01)
optimizer_nb = torch.optim.Adam(net1_nb.parameters(), lr=0.01)

# TensorBoard 记录器
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='logs')

# 训练循环
for epoch in range(200):
    net1_overfitting.train()
    net1_dropped.train()
    net1_nb.train()

    pred_ofit = net1_overfitting(train_data)
    pred_drop = net1_dropped(train_data)
    pred_nb = net1_nb(train_data)

    loss_ofit = loss_func(pred_ofit, train_target)
    loss_drop = loss_func(pred_drop, train_target)
    loss_nb = loss_func(pred_nb, train_target)

    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    optimizer_nb.zero_grad()

    loss_ofit.backward()
    loss_drop.backward()
    loss_nb.backward()

    optimizer_ofit.step()
    optimizer_drop.step()
    optimizer_nb.step()

    # 记录训练损失（包含三个模型）
    writer.add_scalars('train_group_loss',
                       {'overfitting': loss_ofit.item(),
                        'dropout': loss_drop.item(),
                        'batchnorm': loss_nb.item()},
                       epoch)

    # 切换到评估模式
    net1_overfitting.eval()
    net1_dropped.eval()
    net1_nb.eval()

    test_pred_orig = net1_overfitting(test_data)
    test_pred_drop = net1_dropped(test_data)
    test_pred_nb = net1_nb(test_data)

    orig_loss = loss_func(test_pred_orig, test_target)
    drop_loss = loss_func(test_pred_drop, test_target)
    nb_loss = loss_func(test_pred_nb, test_target)

    writer.add_scalars('test_group_loss',
                       {'overfitting': orig_loss.item(),
                        'dropout': drop_loss.item(),
                        'batchnorm': nb_loss.item()},
                       epoch)

writer.close()
>>>>>>> 2a1c479467547321c0d829d3ea5107836c911ba4
print("训练完成，可使用 tensorboard --logdir=logs 查看损失曲线")