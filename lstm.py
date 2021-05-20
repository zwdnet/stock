# coding:utf-8
# lstm神经网络预测股价


import run
import tools
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


# 读取数据
def readData(code="sh000300", start="2018-01-01", end="2018-12-31", refresh = False):
    # 加载数据
    tools.initOutput()
    data = tools.loadData(code=code, start=start, end=end, refresh=refresh)
    print(data, type(data))
    return data
    
    
# 数据归一化
def data_pre(data):
    dataset = data.astype("double")
    max_value = np.max(dataset)
    min_value = np.min(dataset)
    scalar = max_value - min_value
    dataset = list(map(lambda x: x / scalar, dataset))
    return dataset
    
    
# 创建数据集
def create_dataset(dataset, look_back = 2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:i+look_back]
        dataX.append(a)
        dataY.append(dataset[i+look_back])
    return np.array(dataX), np.array(dataY)
    
    
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layer):
        super(LSTM, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, bias = True, batch_first = False, dropout = 0.1)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x, _ = self.layer1(x)
        s, b1, n = x.size()
        x = x.view(s*b1, n)
        x = self.layer2(x)
        x = x.view(s*b1, -1)
        return x


# lstm预测模型
@run.change_dir
def lstm():
    # 准备数据
    data = readData()
    open = data["open"].values
    open = open[::-1]
    open = data_pre(open)
    plt.figure()
    plt.plot(open)
    plt.savefig("./output/open.png")
    index = int(len(open)*0.8)
    train = open[:index]
    test = open[index:]
    # 生成数据集
    data_X2, data_Y2 = create_dataset(test)
    data_X1, data_Y1 = create_dataset(train)
    train_X = data_X1
    train_Y = data_Y1
    train_X = train_X.reshape(-1, 1, 2)
    train_Y = train_Y.reshape(-1, 1, 1)
    train_x = torch.from_numpy(train_X).float()
    train_y = torch.from_numpy(train_Y).float()
    # 模型搭建
    model = LSTM(2, 4, 2, 2)
    # 损失函数
    criterion = nn.MSELoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # 模型训练
    epoches = 1500
    plt.figure()
    for epoch in range(epoches):
        # 前向传播
        out = model.forward(train_x)
        loss = criterion(out, train_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % int(epoches/10) == 0:
            out1=loss.view(-1).data.numpy()
            plt.plot(out1,'r')
            plt.plot(train_x.view(-1).data.numpy(), 'b')
            print(epoch, loss)
    plt.savefig("./output/loss.png")
    plt.close()
    # 模型测试
    test_X = data_X2
    test_Y = data_Y2
    test_X = test_X.reshape(-1, 1, 2)
    test_Y = test_Y.reshape(-1, 1, 1)
    test_x = torch.from_numpy(test_X).float()
    test_y = torch.from_numpy(test_Y).float()
    pred_test = model(test_x)
    pred_test = pred_test.view(-1).data.numpy()
    plt.figure()
    plt.plot(pred_test, 'r', label='prediction')
    plt.plot(test_x.view(-1).data.numpy(), 'b', label='real')
    plt.legend(loc='best')
    plt.savefig("./output/predict.png")
    plt.close()


if __name__ == "__main__":
    lstm()