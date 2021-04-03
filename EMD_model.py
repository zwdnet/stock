# coding:utf-8
# EMD方法


from PyEMD import EMD, EEMD, Visualisation
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import pearsonr
import run
import matplotlib.pyplot as plt
import tools


# 测试PyEMD
@run.change_dir
def testPyEMD():
    testEMD()
    testEEMD()


# emd
def testEMD():
    # 例子1
    s = np.random.random(100)
    emd = EMD()
    IMFs = emd.emd(s)
    print(len(IMFs), len(IMFs[0]), type(IMFs))
    fig = plt.figure()
    ax = fig.add_subplot(len(IMFs)+1, 1, 1)
    ax.plot(s)
    for i in range(len(IMFs)):
        ax = fig.add_subplot(len(IMFs)+1, 1, i+2)
        ax.plot(IMFs[i])
    plt.savefig("./output/emd.png")
    plt.close()
    
    # 例子2
    t = np.linspace(0, 1, 200)
    s = np.cos(11*2*np.pi*t*t) + 6*t*t
    
    IMF = EMD().emd(s, t)
    N = IMF.shape[0] + 1
    
    # 画结果
    plt.subplot(N, 1, 1)
    plt.plot(t, s, 'r')
    plt.title("Input signal: $S(t)=cos(22\pi t^2) + 6t^2$")
    plt.xlabel("Time [s]")
    
    for n, imf in enumerate(IMF):
        plt.subplot(N, 1, n+2)
        plt.plot(t, imf, 'g')
        plt.title("IMF "+str(n+1))
        plt.xlabel("Time [s]")
        
    plt.tight_layout()
    plt.savefig("./output/emd2.png")
    plt.close()
    
    
# eemd
def testEEMD():
    t = np.linspace(0, 1, 200)
    # 定义信号
    sin = lambda x, p: np.sin(2*np.pi*x*t + p)
    S = 3*sin(18,0.2)*(t-0.2)**2
    S += 5*sin(11,2.7)
    S += 3*sin(14,1.6)
    S += 1*np.sin(4*2*np.pi*(t-0.8)**2)
    S += t**2.1 -t

    eemd = EEMD()
    
    # 设置探测极值的方法
    emd = eemd.EMD
    emd.extrema_detection="parabol"
    
    # 对信号执行eemd
    eIMFs = eemd.eemd(S, t)
    nIMFs = eIMFs.shape[0]
    
    # 画结果
    plt.figure(figsize=(12,9))
    plt.subplot(nIMFs+1, 1, 1)
    plt.plot(t, S, 'r')

    for n in range(nIMFs):
        plt.subplot(nIMFs+1, 1, n+2)
        plt.plot(t, eIMFs[n], 'g')
        plt.ylabel("eIMF %i" %(n+1))
        plt.locator_params(axis='y', nbins=5)

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig('./output/eemd.png', dpi=120)
    plt.close()
    
    
# 对EEMD分解结果进行统计描述
def IMF_statics(rawdata, IMFs):
    results = pd.DataFrame()
    n = IMFs.shape[0]
    for i in range(n):
        temp = pd.Series()
        # 1.计算周期 = 数据个数/极值点个数
        data_num = len(IMFs[i])
        greater = argrelextrema(IMFs[i], np.greater)
        less = argrelextrema(IMFs[i], np.less)
        extrem = len(greater[0]) + len(less[0])
        if extrem == 0:
            extrem = 1
        temp["周期"] = data_num / extrem
        # 2.计算均值
        temp["均值"] = IMFs[i].mean()
        # 3.方差
        temp["方差"] = IMFs[i].var()
        # 4.方差占比
        temp["方差占比"] = temp["方差"] / rawdata.var()
        # 5.pearson相关系数
        temp["pearson相关系数"] = pearsonr(IMFs[i], rawdata)[0]
        
        # 形成数据
        results = results.append(temp, ignore_index = True)
    
    return results
        
    
    
# EEMD模型预测股价
@run.change_dir
def EEMD_model():
    start="2005-01-01"
    end="2020-12-31"
    data = tools.loadData(start=start, end=end)
    print(data.info())
    # 划分训练集和测试集
    train_data = data[start:"2019-01-01"].close
    test_data = data["2019-01-01":end].close
    print(train_data.head(), test_data.head())
    # 画图看看
    plt.figure()
    train_data.plot()
    plt.savefig("./output/price.png")
    # 对数据进行EEMD分解
    eemd = EEMD(trials=100, noise_width=0.2)
    eemd.eemd(train_data.values)
    eIMFs, R = eemd.get_imfs_and_residue()
    plt.figure()
    visual = Visualisation()
    visual.plot_imfs(eIMFs, residue=R,  include_residue=True)
    plt.savefig("./output/eemd_hs300.png")
    plt.close()
    # 对分解数据进行统计描述
    statics = IMF_statics(train_data.values, eIMFs)
    print("分解结果统计指标")
    print(statics)
    
    # 将IMFs合成为三个部分
    H_f = eIMFs[0] + eIMFs[1] + eIMFs[2] + eIMFs[3] + eIMFs[4] + eIMFs[5]
    L_f = eIMFs[6] + eIMFs[7] + eIMFs[8] + eIMFs[9] +  eIMFs[10]
    R_f = R
    # 画出原始数据和三个部分
    plt.figure()
    plt.plot(train_data.values, label = "hs300")
    plt.plot(H_f, label = "High", linestyle="--")
    plt.plot(L_f, label = "Low", linestyle=":")
    plt.plot(R_f, label = "residue", linestyle="-.")
    plt.legend(loc = "best")
    plt.savefig("./output/imfs_hs300.png")
    plt.close()
    
    
# 用SVM模型求解
@run.change_dir
def SVM_model():
    # 加载数据
    start="2018-01-01"
    end="2018-12-31"
    data = tools.loadData(start=start, end=end)
    print(data.info())
    # 计算特征值
    data["diff"] = data["close"] - data["close"].shift(1)
    data["diff"].fillna(0, inplace=True)
    print(data["diff"])
    data["up"] = data["diff"]
    data["up"][data["diff"] > 0] = 1
    data["up"][data["diff"] <= 0] = 0
    # 预测值暂置为0
    data["pred"] = 0
    print(data.head())
    target = data["up"]
    length = len(data)
    trainNum = int(length*0.8)
    predNum = length - trainNum
    # 选择指定列作为特征列
    feature = data[["open", "high", "low", "close", "volume", "nextopen", "amount"]]
    # 标准化处理特征值
    feature = preprocessing.scale(feature)
    print(feature[0:10])
    # 训练集的特征值和目标值
    featureTrain = feature[1:trainNum-1]
    targetTrain = target[1:trainNum-1]
    # 训练模型
    svmTool = svm.SVC(kernel="linear")
    svmTool.fit(featureTrain, targetTrain)
    # 预测
    predIndex = trainNum
    # 逐行预测
    temp = []
    while predIndex < length:
        testFeature = feature[predIndex:predIndex+1]
        predForUp = svmTool.predict(testFeature)
        temp.append(predForUp[0])
        predIndex += 1
        
    print(temp)
    
    # 只包含测试集的数据
    dataWithPred = data[trainNum:length]
    dataWithPred["pred"] = temp
    fig = plt.figure()
    (axClose, axUpOrDown) = fig.subplots(2, sharex=True)
    dataWithPred["close"].plot(ax=axClose)
    # axUpOrDown.plot(temp, color = "red", label = "pred")
    dataWithPred["pred"].plot(ax=axUpOrDown, color = "red", label = "pred")
    dataWithPred["up"].plot(ax=axUpOrDown, color = "green", label = "real")
    plt.legend(loc="best")
#    major_index = dataWithPred.index[dataWithPred.index%2==0]
#    major_xtics = dataWithPred['date'][dataWithPredicted.index%2==0]
#    plt.xticks(major_index, major_xtics)
#    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.title("hs300_pred")
    plt.savefig("./output/svm_pred.png")
    plt.close()
    
    # 计算预测正确率
    correct = len(dataWithPred[dataWithPred["pred"] == dataWithPred["up"]])
    print("预测准确率:", correct/predNum)
    
    return svmTool
    
    
# SVM预处理函数
def preprocess(data):
    feature = data[["open", "high", "low", "close", "volume", "nextopen", "amount"]]
    # 标准化处理特征值
    feature = preprocessing.scale(feature)
    return feature
    
    
# EEMD_SVM预处理函数
def EEMD_preprocess(data):
    print("测试2", data.info(), len(data))
    feature = data[["open", "high", "low", "close", "volume", "nextopen", "amount"]]
    # 进行EEMD分解
    feature = do_eemd(feature)
    # 标准化处理特征值
    feature = preprocessing.scale(feature)
    return feature
    
    
# EEMD_SVM预测模型
@run.change_dir
def eemd_svm():
    # 加载数据
    start="2018-01-01"
    end="2018-12-31"
    data = tools.loadData(start=start, end=end)
    # 计算特征值
    data["diff"] = data["close"] - data["close"].shift(1)
    data["diff"].fillna(0, inplace=True)
    print(data["diff"])
    data["up"] = data["diff"]
    data["up"][data["diff"] > 0] = 1
    data["up"][data["diff"] <= 0] = 0
    # 预测值暂置为0
    data["pred"] = 0
    print(data.head())
    target = data["up"]
    length = len(data)
    trainNum = int(length*0.8)
    predNum = length - trainNum
    # 选择指定列作为特征列
    feature = data[["open", "high", "low", "close", "volume", "nextopen", "amount"]]
    # 进行eemd分解
    feature = do_eemd(feature)
    # 标准化处理特征值
    feature = preprocessing.scale(feature)
    print(feature[0])
    
   # 训练集的特征值和目标值
    featureTrain = feature[1:trainNum-1]
    targetTrain = target[1:trainNum-1]
    # 训练模型
    svmTool = svm.SVC(kernel="linear")
    svmTool.fit(featureTrain, targetTrain)
    # 预测
    predIndex = trainNum
    # 逐行预测
    temp = []
    while predIndex < length:
        testFeature = feature[predIndex:predIndex+1]
        predForUp = svmTool.predict(testFeature)
        temp.append(predForUp[0])
        predIndex += 1
        
    print(temp)
    
    # 只包含测试集的数据
    dataWithPred = data[trainNum:length]
    dataWithPred["pred"] = temp
    fig = plt.figure()
    (axClose, axUpOrDown) = fig.subplots(2, sharex=True)
    dataWithPred["close"].plot(ax=axClose)
    # axUpOrDown.plot(temp, color = "red", label = "pred")
    dataWithPred["pred"].plot(ax=axUpOrDown, color = "red", label = "pred")
    dataWithPred["up"].plot(ax=axUpOrDown, color = "green", label = "real")
    plt.legend(loc="best")
#    major_index = dataWithPred.index[dataWithPred.index%2==0]
#    major_xtics = dataWithPred['date'][dataWithPredicted.index%2==0]
#    plt.xticks(major_index, major_xtics)
#    plt.setp(plt.gca().get_xticklabels(), rotation=30)
    plt.title("eemd_svm_hs300_pred")
    plt.savefig("./output/eemd_svm_pred.png")
    plt.close()
    
    # 计算预测正确率
    correct = len(dataWithPred[dataWithPred["pred"] == dataWithPred["up"]])
    print("预测准确率:", correct/predNum)
    
    return svmTool
    
    
# 对特征进行EEMD分解
@run.change_dir
def do_eemd(feature):
    print(feature.columns)
    return_feature = pd.DataFrame()
    eemd = EEMD(trials=100, noise_width=0.2)
    # 设置探测极值的方法
    emd = eemd.EMD
    emd.extrema_detection="parabol"
    for col in feature.columns:
        eemd.eemd(feature[col].values)
        eIMFs, R = eemd.get_imfs_and_residue()
        plt.figure()
        visual = Visualisation()
        visual.plot_imfs(eIMFs, residue=R,  include_residue=True)
        plt.savefig("./output/" + col + ".png")
        plt.close()
        # 对分解数据进行统计描述
        statics = IMF_statics(feature[col].values, eIMFs)
        print(col + "分解结果统计指标")
        print(statics)
        # 将分解的数据合并成高频、低频和剩余项
        n = eIMFs.shape[0]
        high = eIMFs[0] + eIMFs[1] + eIMFs[2]
        low = eIMFs[3]
        for i in range(4, n):
            low += eIMFs[i]
        return_feature[col+"high"] = high
        return_feature[col+"low"] = low
        return_feature[col+"res"] = R
        #elif n == 6:
#            low = eIMFs[4] + eIMFs[5]
#            return_feature[col+"high"] = high
#            return_feature[col+"low"] = low
#            return_feature[col+"res"] = R
        plt.figure()
        plt.plot(return_feature[col+"high"].values, color = "red", label = "high")
        plt.plot(return_feature[col+"low"].values, color = "green", label = "low")
        plt.plot(return_feature[col+"res"].values, color = "blue", label = "res")
        plt.plot(feature[col].values, color = "black", label = "real")
        plt.title(col)
        plt.legend(loc = "best")
        plt.savefig("./output/"+col+"_all.png")
        plt.close()
    return return_feature
    

if __name__ == "__main__":
    # testPyEMD()
    # EEMD_model()
    # 初始化输出格式
    tools.initOutput()
    # model = SVM_model()
    # bt = tools.BackTest(model, type="classify", preprocess=preprocess)
    # results = bt.run()
    # print(results)
    model = eemd_svm()
    bt = tools.BackTest(model, type="classify", preprocess=EEMD_preprocess)
    results = bt.run()
    print(results)
    