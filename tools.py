# coding:utf-8
# 股票价格预测，工具函数


import tushare as ts
import akshare as ak
import pandas as pd
import os
import math
import run
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import empyrical as ey


# 初始化输出
def initOutput():
    pd.set_option('display.max_columns', None)
    #显示所有行
    pd.set_option('display.max_rows', None)
    #设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', 100)


# 下载历史数据
@run.change_dir
def downloadData_old(code="hs300", start="2018-01-01", end="2018-12-31"):
    pro = ts.pro_api()
    data = ts.get_k_data(code=code, start=start, end=end, ktype="D")
    data.index = data.date
    data = data.loc[:, ["open", "close", "high", "low", "volume", "amount"]]
    data.to_csv("data.csv")
    print("下载数据完毕!")
    
    
# 下载历史数据，用akshare
@run.change_dir
def downloadData(code="sh000300"):
    result = ak.stock_zh_index_daily_em(symbol=code)
    result.index = result.date
    result = result.loc[:, ["open", "close", "high", "low", "volume", "amount"]]
    # print(result)
    result.to_csv("./result.csv")
    
    
# 从文件读取数据
@run.change_dir
def loadData(code="sh000300", start="2018-01-01", end="2018-12-31", refresh = False):
    datafile = "./result.csv"
    if os.path.exists(datafile) == False or refresh == True:
        downloadData(code)
    data = pd.read_csv("./result.csv", index_col="date")
    data = data[start : end]
    # print(data.describe())
    data = preProcess(data)
    return data
    
    
# 数据预处理
def preProcess(data):
    data["nextclose"] = data["close"].shift(-1)
    data["nextopen"] = data["open"].shift(-1)
    data = data.iloc[:-1, :]
    # print(len(result))
    # 对成交量成交额进行标准化
    ss = StandardScaler()
    volume = data["volume"].values.reshape(-1, 1)
    # print(len(volume), volume[0])
    data["volume"] = 3500.0 + 500*ss.fit_transform(volume)
    ss = StandardScaler()
    amount = data["amount"].values.reshape(-1, 1)
    # print(len(amount), amount[0])
    data["amount"] = 3500.0 + 500*ss.fit_transform(amount)
    return data
    
    
# 划分特征和目标
def splitData(data, features=["open", "high", "low", "close", "volume", "nextopen", "amount"], target = ["nextclose"]):
    # 划分特征和目标
    X = data.loc[:, features]
    Y = data.loc[:, target]
    return X, Y
    
    
# 策略回测类
class BackTest:
    def __init__(self, model, type="regress", preprocess=None, code="sh000300", start="2019-01-01", end="2019-12-31"):
        self.data = loadData(code=code, start=start, end=end)
        self.X, self.Y = splitData(self.data)
        self.model = model             # 模型
        self.model_type = type       # 模型类型
        if preprocess != None:
            self.preprocess = preprocess # 数据预处理器
        self.stock = [0]                     # 持仓
        self.cash = [100000000]    # 现金
        self.value = []                        # 资产总额
        self.cost = [0.0]                    # 交易成本
        self.fee_rate = 1e-4              # 手续费率
        self.modelname = str(model)[:-2] # 模型名称
        self.bk_results = pd.DataFrame()
        
    # 回归模型回测
    def __regress_run(self):
        for i in range(len(self.data)):
            today_X = self.X.iloc[i, :]
            pred_Y = self.model.predict(today_X.values.reshape(1, -1))
            if i == 0:
                # print("第0天")
                amount = 0
            elif pred_Y[0][0] > today_X.open: # 全仓买入
                # print("买")
                money = self.cash[i - 1]
                price = today_X.open
                amount = math.floor(0.9*money/price)
                # 买入操作
                self.stock.append(self.stock[i-1] + amount)
                self.cash.append(money - price*amount*(1.0 + self.fee_rate))
                self.cost.append(self.cost[i-1] + price*amount*self.fee_rate)
            elif pred_Y[0][0] <= today_X.open: # 清仓
                # print("卖")
                amount = self.stock[i-1]
                price = today_X.open
                self.stock.append(0)
                money = amount*price
                self.cash.append(money*(1.0 - self.fee_rate) + self.cash[i-1])
                self.cost.append(self.cost[i-1] + money*self.fee_rate)
            self.value.append(self.cash[i] + self.stock[i]*today_X.close)
            
        # 生成收益率数据
        self.genReturn()
        
        # 计算回测指标
        self.evaluation()
        
        return self.bk_results
        
    # 分类模型回测
    def __classify_run(self):
        for i in range(10, len(self.data)):
            today_X = self.X.iloc[i-10:i]
            print("测试1", type(today_X), len)
            feature = self.preprocess(today_X)
            # pred_Y = self.model.predict(today_X.values.reshape(1, -1))
            pred_Y = self.model.predict(feature.reshape(10, -1))
            if i == 10:
                # print("第0天")
                amount = 0
            elif pred_Y[0] == 1: # 全仓买入
                # print("买")
                money = self.cash[i - 1]
                price = today_X.open
                amount = math.floor(0.9*money/price)
                # 买入操作
                self.stock.append(self.stock[i-1] + amount)
                self.cash.append(money - price*amount*(1.0 + self.fee_rate))
                self.cost.append(self.cost[i-1] + price*amount*self.fee_rate)
            elif pred_Y[0] == 0: # 清仓
                # print("卖")
                amount = self.stock[i-1]
                price = today_X.open
                self.stock.append(0)
                money = amount*price
                self.cash.append(money*(1.0 - self.fee_rate) + self.cash[i-1])
                self.cost.append(self.cost[i-1] + money*self.fee_rate)
            self.value.append(self.cash[i] + self.stock[i]*today_X.close)
            
        # 生成收益率数据
        self.genReturn()
        
        # 计算回测指标
        self.evaluation()
        
        return self.bk_results
        
    # 进行回测
    def run(self):
        if self.model_type == "regress":
            return self.__regress_run()
        elif self.model_type == "classify":
            return self.__classify_run()
        
            
    # 生成收益率数据
    def genReturn(self):
        # 生成收益率数据
        self.return_value = pd.DataFrame(self.value)
        self.return_value["value"] = self.value
        self.return_value["returns"] = self.return_value["value"].pct_change()
        self.return_value["benchmark_returns"] = self.data["close"].pct_change().values
        self.return_value["date"] = self.data.index[:len(self.value)]
        self.return_value.index = self.return_value["date"]
            
    # 画结果
    def draw(self):
        oldpath = os.getcwd()
        newpath = "/home/code/"
        os.chdir(newpath)
        plt.figure()
        plt.plot(self.value)
        plt.savefig("./output/" + modelname + "_backtest_value.png")
        plt.close()
        # 画每日收益率图
        plt.figure()
        plt.plot(self.return_value["returns"])
        plt.savefig("./output/" + modelname + "_backtest_returns.png")
        plt.close()
        os.chdir(oldpath)
        
    # 计算并返回回测评估结果
    def evaluation(self):
        returns = self.return_value.returns
        benchmark = self.return_value.benchmark_returns
        excess_return = returns - benchmark
    
        # 用empyrical计算回测指标
        # 年化收益率
        self.bk_results["年化收益率"] = [ey.annual_return(returns)]
        # 累计收益率
        self.bk_results["累计收益率"] = [ey.cum_returns(returns)]
        # 最大回撤
        self.bk_results["最大回撤"] = [ey.max_drawdown(returns)]
        # 夏普比率
        self.bk_results["夏普比率"] = [ey.sharpe_ratio(excess_return)]
        # 索提比率
        self.bk_results["索提比率"] = [ey.sortino_ratio(returns)]
        # αβ值
        ab = ey.alpha_beta(returns, benchmark, risk_free = 0.02)
        self.bk_results["α"] = ab[0]
        self.bk_results["β"] = ab[1]


if __name__ == "__main__":
    data = loadData()
    print(data.head(), data.describe())
    data = preProcess(data)
    print(data.head(), data.describe())
    