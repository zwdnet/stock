# coding:utf-8
# 多元线性回归
# 参考王培冬.基于多元线性回归的股价分析及预测.科技经济市场，2020(1):84-85.


import run
import tools
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns


# 训练模型的工具
class TrainTool:
    def __init__(self, Model, code="sh000300", start="2018-01-01", end="2018-12-31", refresh = False):
        # 初始化输出格式
        tools.initOutput()
        self.data = tools.loadData(code=code, start=start, end=end, refresh=refresh)
        self.model = Model()
        self.features = ["open", "high", "low", "close", "volume", "nextopen", "amount"]
        self.target = ["nextclose"]
        
    # 数据探索
    @run.change_dir
    def EDA(self):
        print("数据探索")
        print(self.data.head())
        print(self.data.describe())
        print(self.data[self.data.isnull() == True].count())
        # 画图看看
        plt.figure()
        self.data.boxplot()
        plt.savefig("./output/boxview.png")
        plt.close()

        # 计算特征之间的相关系数
        corr = self.data.corr()
        print(corr)
        plt.figure()
        sns.pairplot(self.data, x_vars=self.features, y_vars=self.target)
        plt.savefig("./output/paircorr.png")
        plt.close()
    
    # 划分特征和目标
    def splitData(self, data):
        # 划分特征和目标
        return tools.splitData(data, features=self.features, target=self.target)
    
    # 用模型Model训练
    def train(self, train_size=0.8):
        X, Y = self.splitData(self.data)
        # 分割训练集和测试集
        X_train,X_test,Y_train,Y_test = train_test_split(X, Y,train_size=.80)
        # 训练
        self.model.fit(X_train, Y_train)
        # 模型评价
        score = self.model.score(X_test, Y_test)
        print("模型评分:", score)
        return self.model
    
    
    # 用新数据验证模型
    @run.change_dir
    def check(self, start="2019-01-01", end="2019-12-31"):
        name = str(self.model)[:-2]
        val_data = tools.loadData(start=start, end=end)
        X, Y= self.splitData(val_data)
        # 模型评价
        score = self.model.score(X, Y)
        print("模型验证评分:", score)
        # 用模型进行预测
        pred_Y = self.model.predict(X)
        res = pred_Y - Y
        # 预测值与实际值曲线
        plt.figure()
        plt.plot(Y, label="real")
        plt.plot(pred_Y, label="prediction")
        plt.legend(loc='best')
        plt.savefig("./output/" + name + "_val_result.png")
        # 计算预测值与真实值的误差率
        error = (pred_Y - Y)/Y
        plt.figure()
        plt.plot(error, ".")
        plt.savefig("./output/" + name + "_error.png")
        plt.close()
        plt.figure()
        plt.hist(error, bins=20)
        plt.savefig("./output/" + name + "_error_hist.png")
        plt.close()
        print("平均误差率:", error.mean())
    

# 进行多元线性回归
@run.change_dir
def LR():
    lr_trainer = TrainTool(LinearRegression)
    # 数据探索
    # lr_trainer.EDA()
    model = lr_trainer.train()
    a = model.intercept_
    b = model.coef_
    print("截距:", a)
    print("回归系数:", b)
    # 验证模型
    lr_trainer.check()
    return model
    
    

    
    
# 用模型进行回测
@run.change_dir
def backTest(model, code="sh000300", start="2019-01-01", end="2019-12-31"):
    data = tools.loadData(code=code, start=start, end=end)
    X, Y = tools.splitData(data)
    stock = [0]                     # 持仓
    cash = [100000000]    # 现金
    value = []                        # 资产总额
    cost = [0.0]                          # 交易成本
    fee_rate = 1e-4              # 手续费率
    for i in range(len(data)):
    # for i in range(4):
        today_X = X.iloc[i, :]
        pred_Y = model.predict(today_X.values.reshape(1, -1))
        if i == 0:
            # print("第0天")
            amount = 0
        elif pred_Y[0][0] > today_X.open: # 全仓买入
            # print("买")
            money = cash[i - 1]
            price = today_X.open
            amount = math.floor(0.9*money/price)
            # 买入操作
            stock.append(stock[i-1] + amount)
            cash.append(money - price*amount*(1.0 + fee_rate))
            cost.append(cost[i-1] + price*amount*fee_rate)
        elif pred_Y[0][0] <= today_X.open: # 清仓
            # print("卖")
            amount = stock[i-1]
            price = today_X.open
            stock.append(0)
            money = amount*price
            cash.append(money*(1.0 - fee_rate) + cash[i-1])
            cost.append(cost[i-1] + money*fee_rate)
        value.append(cash[i] + stock[i]*today_X.close)
    modelname = str(model)[:-2]
    plt.figure()
    plt.plot(value)
    plt.savefig("./output/" + modelname + "_backtest_value.png")
    plt.close()
    # 生成收益率数据
    return_value = pd.DataFrame(value)
    return_value["value"] = value
    return_value["returns"] = return_value["value"].pct_change()
    return_value["benchmark_returns"] = data["close"].pct_change().values
    return_value["date"] = data.index[:len(value)]
    return_value.index = return_value["date"]
    # 画每日收益率图
    plt.figure()
    plt.plot(return_value["returns"])
    plt.savefig("./output/" + modelname + "_backtest_returns.png")
    plt.close()
    return return_value
    
    
# 计算回测指标
def evaluation(value):
    print(value.head())

    returns = value.returns
    benchmark = value.benchmark_returns
    excess_return = returns - benchmark
    
    # 用empyrical计算回测指标
    bk_results = pd.DataFrame()
    # 年化收益率
    bk_results["年化收益率"] = [ey.annual_return(returns)]
    # 累计收益率
    bk_results["累计收益率"] = [ey.cum_returns(returns)]
    # 最大回撤
    bk_results["最大回撤"] = [ey.max_drawdown(returns)]
    # 夏普比率
    bk_results["夏普比率"] = [ey.sharpe_ratio(excess_return)]
    # 索提比率
    bk_results["索提比率"] = [ey.sortino_ratio(returns)]
    # αβ值
    ab = ey.alpha_beta(returns, benchmark, risk_free = 0.02)
    bk_results["α"] = ab[0]
    bk_results["β"] = ab[1]
    return bk_results
    
    
# 测试夏普比率，α、β值
@run.change_dir
def test():
    stock_data = pd.read_csv("stock_data.csv",
            parse_dates=['Date'], 
            index_col = ['Date']).dropna()
    benchmark_data = pd.read_csv("benchmark_data.csv",
            parse_dates=['Date'], 
            index_col = ['Date']).dropna()
            
    print("Stocks\n")
    print(stock_data.info())
    print(stock_data.head())
    print("Benchmarks")
    print(benchmark_data.info())
    print(benchmark_data.head())
    
    stocks_ret = stock_data.pct_change()
    benchmark_ret = benchmark_data.pct_change()
    print(stocks_ret.head())
    
    excess_returns = pd.DataFrame()
    excess_returns["Amazon"] = stocks_ret["Amazon"] - benchmark_ret["S&P 500"]
    excess_returns["Facebook"] = stocks_ret["Facebook"] - benchmark_ret["S&P 500"]
    print(excess_returns.describe())
    
    avg_excess_return = excess_returns.mean()
    print(avg_excess_return)
    
    sd_excess_return = excess_returns.std() 
    print(sd_excess_return)
    
    # 计算夏普比率
    # 日夏普比率
    daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)
    # 年化夏普比率
    annual_factor = np.sqrt(252)
    annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)
    print("夏普比率:")
    print("日", daily_sharpe_ratio)
    print("年", annual_sharpe_ratio)
    
    # 用empyrical计算夏普比率
    sharpe = pd.DataFrame()
    sharpe["Amazon"] = [ey.sharpe_ratio(excess_returns["Amazon"])]
    sharpe["Facebook"] = [ey.sharpe_ratio(excess_returns["Facebook"])]
    print("empyrical结果:\n", sharpe)
    
    


if __name__ == "__main__":
    model = LR()
#    value = backTest(model)
#    result = evaluation(value)
#    print(result)
    # test()
    bt = tools.BackTest(model)
    results = bt.run()
    print(results)
    