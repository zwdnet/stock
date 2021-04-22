# coding:utf-8
# 时间序列模型ARMA预测股价


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tools
import run
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import _arma_predict_out_of_sample as out_predict
import warnings
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA', FutureWarning)


# 准备数据
def readData(code="sh000300", start="2018-01-01", end="2018-12-31", refresh = False):
    # 加载数据
    tools.initOutput()
    data = tools.loadData(code=code, start=start, end=end, refresh=refresh)
    # 筛选特征
    feature_cols = ["close"]
    features_data = data.loc[:, feature_cols]
    scaler = MinMaxScaler(feature_range=(1, 2))
    features = scaler.fit_transform(features_data.values)
    x = []
    for i in range(len(features)):
        x.append(features[i][0])
    new_data = pd.DataFrame({"close":x})
    new_data["date"] = data.index
    new_data.set_index("date", inplace = True)
    print(new_data.head())
    return new_data
    
    
# 画数据
@run.change_dir
def draw(data):
    plt.figure()
    plt.plot(data.close)
    plt.savefig("./output/rawdata.png")
    # 画自相关函数
    plot_acf(data.close, lags=25, title="acf")
    plt.savefig("./output/acf.png")
    # 画自相关函数
    plot_pacf(data.close, lags=25, title="pacf")
    plt.savefig("./output/pacf.png")
    plt.close()
    
    
# 将数据差分
def diff_data(data):
    diff = data.diff(1).dropna()
    return diff
    
    
# 寻找参数p和q
def findPQ(data):
    # N = len(data)
    best = float("inf")
    bestp, bestq = -1, -1
    N = 20
    for p in range(1, N):
        for q in range(1, N):
            try:
                arma = ARMA(data, (p, q)).fit(disp = -1)
                aic = arma.aic
                print(p, q, aic)
                if aic < best:
                    best = aic
                    bestp = p
                    bestq = q
            except:
                print("出现异常")
    return p, q
    
    
# 用ARMA模型建模并预测
@run.change_dir
@run.timethis
def arma_model(train_data, p, q):
    data = readData(start = "2019-01-01", end = "2019-12-31")
    data = diff_data(data)
    model = ARMA(train_data, (p, q)).fit()
    print(model.summary)
    print(model.conf_int())
    history_p = train_data
    pred_p = []
    print("测试", history_p)
    for i in range(len(data)):
        model_p = ARMA(train_data, p, q)
        model_fit_p = model_p.fit(disp = -1)
        yhat_p = model_fit_p.predict(start = len(history_p), end = len(history_p))[0]
        pred_p.append(yhat_p)
        history_p.append(yhat_p)
        print("测试", yhat_p)
    results = pd.DataFrame()
    results["pred"] = pred_p
    results["real"] = data
    results["date"] = data.index
    results.set_index("date", inplace = True)
    print("预测结果")
    print(results.head())
#    # predict = model.forecast()
#    print("预测")
#    plt.figure()
#    model.plot_predict(start = 10, end = 200)
#    plt.savefig("./output/predict_arma.png")
#    plt.close()
#    # print(model.forecast(5))
#    pred = model.predict()
#    # 提取参数
#    params = model.params
#    residuals = model.resid
#    p = model.k_ar
#    q = model.k_ma
#    k_exog = model.k_exog
#    k_trend = model.k_trend
#    steps = 1

#    # 样本内数据验证
#    in_data = train_data.iloc[:11, :]
#    in_resid = residuals.iloc[0:11]
#    a = out_predict(params, steps, in_resid, p, q, k_trend, k_exog, endog = in_data["close"], exog = None, start = len(in_data))
#    test_pred = pred[8:13]
#    print(test_pred, a[0])
#    # 样本外预测
#    new_resid =  residuals.tail(9).copy()
#    new_data = train_data.tail(9).copy()
#    for  i in range(len(data)):
#        print(i)
#        a = out_predict(params, steps, in_resid, p, q, k_trend, k_exog, endog = new_data["close"], exog = None, start = len(new_data))
#        tomorrow_index = data.index[i]
#        temp_resid = data.loc[tomorrow_index,'close'] - a[0]
#        new_resid[tomorrow_index] = temp_resid
#        new_resid.drop(new_resid.index[0],inplace=True)
#        new_data = new_data.append(data.iloc[i,:])
#        # new_data.drop(data.index[0], inplace=True)
#        pred[tomorrow_index] = a[0]


if __name__ == "__main__":
    data = readData()
    draw(data)
    data = diff_data(data)
    draw(data)
    # p, q = findPQ(data)
    arma_model(data, 9, 9)
    