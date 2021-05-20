# coding:utf-8
# 神经网络预测股价
# 没完成


import run
import tools
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# 训练神经网络
def DL(code="sh000300", start="2018-01-01", end="2018-12-31", refresh = False):
    # 加载数据
    tools.initOutput()
    data = tools.loadData(code=code, start=start, end=end, refresh=refresh)
    # 筛选特征
    feature_cols = ["open", "close", "high", "low", "volume"]
    target_cols = ["close"]
    features_data = data.loc[:, feature_cols]
    # 特征归一化
    mm = MinMaxScaler(feature_range=(0, 1))
    for col in feature_cols:
        features_data[col] = mm.fit_transform(features_data[col].values.reshape(-1, 1))
    target_data = data.loc[:, target_cols].values
    print(features_data.shape)
    features_data = np.array(features_data)
    features_data = torch.tensor(features_data)
    stock_len = len(features_data)
    tr_val_slip = int(0.8*stock_len)
    print("数据天数:", stock_len)
    print("可预测天数:", tr_val_slip)
    sqe_len = 5
    x = torch.zeros(stock_len - sqe_len, sqe_len, 5)
    y = torch.zeros(stock_len - sqe_len, 1)
    for i in range(0, stock_len - sqe_len - 1):
        x[i] = features_data[i:i+sqe_len]
        y[i] = features_data[i+sqe_len, 1]
        print(x[i])
    
    # 形成训练集和验证集
    train_x = x[0:tr_val_slip]
    train_y = y[0:tr_val_slip]
    vaild_x = x[tr_val_slip:]
    vaild_y = y[tr_val_slip:]
    print(train_x.shape, train_y.shape, vaild_x.shape, vaild_y.shape)
    
    # 形成DataLoader
    class StockDataset(Dataset):
        def __init__(self, x, y):
            self.x = x
            self.y = y
            
        def __len__(self):
            return len(self.x)
            
        def __getitem__(self, index):
            X = self.x[index]
            Y = self.y[index]
            return X, Y
            
    batch_size = 2
    train_set = StockDataset(train_x, train_y)
    vaild_set = StockDataset(vaild_x, vaild_y)
    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = False)
    vaild_loader = DataLoader(vaild_set, batch_size = batch_size, shuffle = False)
    
    #for i in range(5, len(features_data)):
#        temp = []
#        for j in range(5):
#            temp.append(features_data[i][j])
#        x.append(temp)
#        y.append(target_data[i])
#    print(x, len(x), len(y))
#    train_x = x[:200]
#    train_y = y[:200]
#    test_x = x[200:]
#    test_y = y[200:]
    # 定义神经网络
    lr = 0.001
    epochs = 20
    class bp_net(nn.Module):
        def __init__(self, batch_size = batch_size):
            super(bp_net, self).__init__()
            self.layer_input = nn.Linear(5, 200)
            self.layer_hide = nn.Linear(200, 16)
            self.layer_output = nn.Linear(16, 1)
            self.batch_size = batch_size
            
        def forward(self, x):
            x = self.layer_input(x)
            nn.ReLU()
            x = self.layer_hide(x)
            nn.ReLU()
            x = self.layer_output(x)
            return x
            
    net = bp_net(batch_size)
    criterion = nn.MSELoss()
    optim = torch.optim.Adam(net.parameters(), lr = lr, weight_decay=0.0)
    # 训练过程
    for epoch in range(epochs):
        train_loss = []
        for x, y in train_loader:
            # x = x.view(batch_size, -1)
            y_pred = net.forward(x)
            print("测试", y.shape, y_pred.shape)
            print(y_pred)
            loss = criterion(y_pred, y)
            train_loss.append(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
        mean_train_loss = torch.mean(torch.tensor(train_loss))
        print("第%d次迭代，平均损失值:%f" % (i, mean_train_loss))


if __name__ == "__main__":
    DL()