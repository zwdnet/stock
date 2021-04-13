# coding:utf-8
# 机器学习模型


import run
import tools
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import random
import time
import math
from sko.GA import GA
import empyrical as ey
import os


# 准备数据
def readData(code="sh000300", start="2018-01-01", end="2018-12-31", refresh = False):
    # 加载数据
    tools.initOutput()
    data = tools.loadData(code=code, start=start, end=end, refresh=refresh)
    # 筛选特征
    feature_cols = ["open", "close", "high", "low", "volume", "amount"]
    target_cols = ["close"]
    features_data = data.loc[:, feature_cols]
    target_data = data.loc[:, target_cols]
    # target_data.fillna(method = "ffill", inplace = True)
#    print(features_data.info())
#    print(target_data.head())
    scaler = MinMaxScaler(feature_range=(1, 2))
    features = scaler.fit_transform(features_data.values)
    target = scaler.fit_transform(target_data.values)

    # 将特征数据扩展到5天的数据
    new_features = []
    new_target = []
    for i in range(len(features) - 5):
        # x = features[i:i+5].reshape(1, -1)[0].tolist()
        x = features[i:i+5]
        x = np.mean(x, axis = 1).tolist()
        new_features.append(x)
        new_target.append(target[i+5].tolist()[0])
#        print(x)
#        print(new_features)
#        print(new_target)
#        input("按任意键继续")
    return (new_features, new_target, features_data, target_data)
    # return (features, target)
    
    
# 将时间序列特征数据划分为训练集和测试集
def splitTimeSeries(features, target, train=0.8):
    n = len(features)
    train_size = int(n*train)
    print(len(features), len(target))
    X_train = features[:train_size][:]
    X_test = features[train_size:][:]
    y_train = target[:train_size]
    y_test = target[train_size:]
    return X_train, X_test, y_train, y_test


# 支持向量机进行预测
@run.change_dir
def SVM():
    data = readData()
    features, target = data[0], data[1]
    # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(features, target, train_size = 0.8)
    X_train, X_test, y_train, y_test = splitTimeSeries(features, target)
    #print(len(X_train), len(y_train), len(X_test), len(y_test))
    #print("y_test", y_test)
    # 训练模型
    clf = svm.SVR(kernel = "linear")
    clf.fit(X_train, y_train)
    # 评估模型
    score = clf.score(X_test, y_test)
    print("模型评分:", score)
    # 返回模型
    return clf
    
    
# 测试模型
@run.change_dir
def testModel(model):
    data = readData(start="2019-01-01", end="2019-12-31")
    features, target = data[0], data[1]
    score = model.score(features, target)
    print("验证数据模型评分:", score)
    pred = model.predict(features)
    plt.figure()
    plt.plot(target, label="real")
    plt.plot(pred, label="pred")
    plt.legend(loc="best")
    plt.savefig("./output/"+str(model)[:3]+"_test.png")
    plt.close()
    print("模型预测涨跌准确率:%f" % (testHighLow(target, pred)))
    
    
# 测试模型预测股价涨跌的能力
@run.change_dir
def testHighLow(real, pred):
    real_hl = []
    pred_hl = []
    n = len(real)
    acc = 0
    for i in range(1, n):
        if real[i] > real[i-1]:
            real_hl.append(1)
        elif real[i] <= real[i-1]:
            real_hl.append(0)
        if pred[i] > pred[i-1]:
            pred_hl.append(1)
        elif pred[i] <= pred[i-1]:
            pred_hl.append(0)
    # print(len(real_hl), len(pred_hl))
    for i in range(len(real_hl)):
        if real_hl[i] == pred_hl[i]:
            acc += 1
    accuracy = acc/n
    return accuracy
    
    
# 用收盘价均值作为预测值
@run.change_dir
def experiment():
    data = readData(start="2019-01-01", end="2019-12-31")
    features, target = data[0], data[1]
    # score = model.score(features, target)
    # print("验证数据模型评分:", score)
    pred = []
    for i in range(len(features)):
        pred.append(features[i][1])
    # print("试验值:", pred)
    plt.figure()
    plt.plot(target, label="real")
    plt.plot(pred, label="pred")
    plt.legend(loc="best")
    plt.savefig("./output/experiment.png")
    plt.close()
    print("模型预测涨跌准确率:%f" % (testHighLow(target, pred)))    
    
    
# 网格搜索进行支持向量机调参
@run.change_dir
def GridSVM():
    features, target = readData()[0:1]
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = splitTimeSeries(features, target)
    #print(len(X_train), len(y_train), len(X_test), len(y_test))
    #print("y_test", y_test)
    # 训练模型
    model = svm.SVR()
    params = [
        {'C': range(1, 1000, 10), 'epsilon':np.arange(1e-4, 1.0, 1e-2), 'kernel': ['linear']},
        {'C': range(1, 1000, 10), 'epsilon':np.arange(1e-4, 1.0, 1e-2), 'kernel': ['rbf']},
        {'C': range(1, 1000, 10), 'epsilon':np.arange(1e-4, 1.0, 1e-2), 'kernel': ['sigmoid']}
    ]
    clf = GridSearchCV(model, params, cv = 5, verbose = 10)
    clf.fit(X_train, y_train)
    best_model = clf.best_estimator_
    # 输出模型参数
    print("模型参数:", clf.best_params_)
    # 评估模型
    y_pred = best_model.predict(X_test)
    print('最佳模型准确率评分', best_model.score(X_test, y_test))
    print("模型预测涨跌准确率:%f" % (testHighLow(y_test, y_pred)))   
    # 返回模型
    return best_model
    
    
# 粒子群算法进行SVR调参
class PSO:
    # max_value和min_value分别为参数的最大/最小值
    def __init__(self, particle_num, particle_dim, iter_num, c1, c2, w, max_value, min_value):
        self.particle_num = particle_num
        self.particle_dim = particle_dim
        self.iter_num = iter_num
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.max_value = max_value
        self.min_value = min_value
        # 读取数据
        features, target = readData()[0:1]
        self.X_train, self.X_test, self.y_train, self.y_test = splitTimeSeries(features, target)
        
    # 粒子群初始化
    def swarm_origin(self):
        # 初始化随机数种子
        random.seed(time.time())
        particle_loc = []
        particle_dir = []
        for i in range(self.particle_num):
            tmp1 = []
            tmp2 = []
            for j in range(self.particle_dim):
                a = random.random()
                b = random.random()
                tmp1.append(a * (self.max_value[j] - self.min_value[j]) + self.min_value[j])
                tmp2.append(b)
            particle_loc.append(tmp1)
            particle_dir.append(tmp2)
            
        return particle_loc, particle_dir
        
    # 计算适应度列表，更新pbest,gbest
    def fitness(self, particle_loc):
        fitness_value = []
        # 适应度函数为模型预测正确率
        for i in range(self.particle_num):
            clf = svm.SVR(kernel = "linear", C = particle_loc[i][0], epsilon = particle_loc[i][1])
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_test)
            fitness_value.append(testHighLow(self.y_test, y_pred))
        # 当前粒子群最优适应度函数值和对应参数
        current_fitness = 0.0
        current_parameter = []
        for i in range(self.particle_num):
            if current_fitness < fitness_value[i]:
                current_fitness = fitness_value[i]
                current_parameter = particle_loc[i]
                
        return fitness_value, current_fitness, current_parameter
        
    # 粒子位置更新
    def update(self, particle_loc, particle_dir, gbest_parameter, pbest_parameters):
        # 计算新的粒子群方向和位置
        for i in range(self.particle_num):
            a1 = [x*self.w for x in particle_dir[i]]
            a2 = [y*self.c1*random.random() for y in list(np.array(pbest_parameters[i]) - np.array(particle_loc[i]))]
            a3 = [z*self.c2*random.random() for z in list(np.array(gbest_parameter) - np.array(particle_dir[i]))]
            particle_dir[i] = list(np.array(a1) + np.array(a2) + np.array(a3))
            particle_loc[i] = list(np.array(particle_loc[i]) + np.array(particle_dir[i]))
        # 将更新后的粒子位置参数固定
        parameter_list = []
        for i in range(self.particle_dim):
            tmp1 = []
            for j in range(self.particle_num):
                tmp1.append(particle_loc[j][i])
            parameter_list.append(tmp1)
        value = []
        for i in range(self.particle_dim):
            tmp2 = []
            tmp2.append(max(parameter_list[i]))
            tmp2.append(min(parameter_list[i]))
            value.append(tmp2)
            
        for i in range(self.particle_num):
            for j in range(self.particle_dim):
                particle_loc[i][j] = (particle_loc[i][j] - value[j][1])/(value[j][0] - value[j][1]) * (self.max_value[j] - self.min_value[j]) + self.min_value[j]
                
        return particle_loc, particle_dir
        
    # 画出适应度函数值变化图
    @run.change_dir
    def plot(self, results):
        x = []
        y = []
        for i in range(self.iter_num):
            x.append(i + 1)
            y.append(results[i])
        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Number of iteration")
        plt.ylabel("Value of fitness")
        plt.title("PSO_SVM")
        plt.savefig("./output/PSO_svm.png")
        plt.close()
        
    # 主函数
    def main(self):
        results = []
        best_fitness = 0.0
        # 粒子群初始化
        particle_loc, particle_dir = self.swarm_origin()
        # 初始化参数
        gbest_parameter = []
        for i in range(self.particle_dim):
            gbest_parameter.append(0.0)
        pbest_parameters = []
        for i in range(self.particle_num):
            tmp1 = []
            for j in range(self.particle_dim):
                tmp1.append(0.0)
            pbest_parameters.append(tmp1)
        fitness_value = []
        for i in range(self.particle_num):
            fitness_value.append(0.0)
            
        # 迭代
        for i in range(self.iter_num):
            # 计算当前适应度函数值列表
            current_fitness_value, current_best_fitness, current_best_parameter = self.fitness(particle_loc)
            # 求当前最佳参数
            for j in range(self.particle_num):
                if current_fitness_value[j] > fitness_value[j]:
                    pbest_parameters[j] = particle_loc[j]
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                gbest_parameter = current_best_parameter
                
            print("迭代次数:", i+1, " 最佳参数:", gbest_parameter, " 最佳适应度:", best_fitness)
            results.append(best_fitness)
            # 更新适应度值
            fitness_value = current_fitness_value
            # 更新粒子群
            particle_loc, particle_dir = self.update(particle_loc, particle_dir, gbest_parameter, pbest_parameters)
        
        # 结果展示
        results.sort()
        self.plot(results)
        print("最终参数:", gbest_parameter)
        return gbest_parameter
        
        
# 粒子群算法SVM调参
@run.timethis
def PSO_SVM():
    particle_num = 200
    particle_dim = 2
    iter_num = 500
    c1 = 0.5
    c2 = 0.5
    w = 2.0
    max_value = [100, 1.0]
    min_value = [1, 0.0001]
    pso = PSO(particle_num,particle_dim,iter_num,c1,c2,w,max_value,min_value)
    best_params = pso.main()
    # 用新数据验证模型
    model = svm.SVR(kernel = "linear", C = best_params[0], epsilon = best_params[1])
    features, target = readData()[0:1]
    X_train, X_test, y_train, y_test = splitTimeSeries(features, target)
    model.fit(X_train, y_train)
    testModel(model)
    
    
"""
# 没调通，放弃
# 遗传算法调参
class GA:
    # 1.初始化
    def __init__(self, population_size, chromosome_num, chromosome_length, max_value, iter_num, pc, pm):
#        
#        初始化参数
#        population_size(int):种群数
#        chromosome_num(int):染色体数，对应需要寻优的参数个数
#        chromosome_length:染色体的基因长度
#        max_value(float):作用于二进制基因转化为染色体十进制数值
#        iter_num(int):迭代次数
#        pc(float):交叉概率阈值(0<pc<1)
#        pm(float):变异概率阈值(0<pm<1)
#        
        self.population_size = population_size
        self.chromosome_num = chromosome_num
        self.chromosome_length = chromosome_length
        self.max_value = max_value
        self.iter_num = iter_num
        self.pc = pc
        self.pm = pm
        # 读取数据
        features, target = readData()
        self.X_train, self.X_test, self.y_train, self.y_test = splitTimeSeries(features, target)
        
    # 初始化种群
    def species_origin(self):
        random.seed(time.time())
        population = []
        # 分别初始化两个染色体
        for i in range(self.chromosome_num):
            tmp1 = []
            for j in range(self.population_size):
                tmp2 = []
                for l in range(self.chromosome_length):
                    tmp2.append(random.randint(0, 1))
                tmp1.append(tmp2)
            population.append(tmp1)
        return population
        
    # 2.计算适应度函数值
    # 将染色体二进制基因转换为十进制取值
    def translation(self, population):
        population_decimalism = []
        for i in range(len(population)):
            tmp = []
            for j in range(len(population[0])):
                total = 0.0
                for l in range(len(population[0][0])):
                    total += population[i][j][l] * (math.pow(2, l))
                tmp.append(total)
            population_decimalism.append(tmp)
        return population_decimalism
        
    # 计算每一条染色体对应的适应度
    def fitness(self, population):
        fitness = []
        population_decimalism = self.translation(population)
        for i in range(len(population[0])):
            tmp = []
            for j in range(len(population)):
                value = population_decimalism[j][i] * self.max_value / (math.pow(2, self.chromosome_length) - 10)
                tmp.append(value)
            # 防止参数值为0
            if tmp[0] == 0.0:
                tmp[0] = 0.5
            if tmp[1] == 0.0:
                tmp[1] = 0.5
            clf = svm.SVR(kernel = "linear", C = abs(tmp[0]), epsilon = abs(tmp[1]))
            clf.fit(self.X_train, self.y_train)
            y_pred = clf.predict(self.X_train)
            score = testHighLow(self.y_test, y_pred)
            fitness.append(score)
            
        # 将适应值中为负数的数值排除
        fitness_value = []
        num = len(fitness)
        for l in range(num):
            if (fitness[l] > 0):
                tmp1 = fitness[l]
            else:
                tmp1 = 0.0
            fitness_value.append(tmp1)
        return fitness_value
        
    # 3.选择操作
    # 适应度求和
    def sum_value(self, fitness_value):
        total = 0.0
        for i in range(len(fitness_value)):
            total += fitness_value[i]
        return total
        
    # 计算适应度累加列表
    def cumsum(self, fitness):
        for i in range(len(fitness)-1, -1, -1):
            total = 0.0
            j = 0
            while (j <= i):
                total += fitness[j]
                j += 1
            fitness[i] = total
            
    # 选择操作
    def selection(self, population, fitness_value):
        new_fitness = []
        total_fitness = self.sum_value(fitness_value)
        for i in range(len(fitness_value)):
            new_fitness.append(fitness_value[i] / total_fitness)
        self.cumsum(new_fitness)
        
        ms = []
        pop_len = len(population[0]) # 种群数
        
        for i in range(pop_len):
            ms.append(random.randint(0, 1))
        ms.sort()
        
        # 存储每个染色体的取值指针
        fitin = 0
        newin = 0
        
        new_population = population
        
        # 轮盘赌方式选择染色体
        while newin < pop_len & fitin < pop_len:
            if (ms[newin] < new_fitness[fitin]):
                for j in range(len(population)):
                    new_population[j][newin] = population[j][fitin]
                newin += 1
            else:
                fitin += 1
                
        population = new_population
        
    # 交叉操作
    def crossover(self, population):
        pop_len = len(population[0])
        
        for i in range(len(population)):
            for j in range(pop_len - 1):
                if (random.random() < self.pc):
                    cpoint = random.randint(0, len(population[i][j]))
                    tmp1 = []
                    tmp2 = []
                    tmp1.extend(population[i][j][0:cpoint])
                    tmp1.extend(population[i][j+1][cpoint:len(population[i][j])])
                    tmp2.extend(population[i][j+1][0:cpoint])
                    tmp1.extend(population[i][j][cpoint:len(population[i][j])])
                    # 将交叉后的染色体取值放入新的种群中
                    population[i][j] = tmp1
                    population[i][j+1] = tmp2
                    
    # 变异操作
    def mutation(self, population):
        print(len(population))
        print(len(population[1]))
        # print(len(population[1][57]))
        # print(population[1][57])
        # print(population[1][57][255])
        pop_len = len(population[0]) # 种群数
        Gene_len = len(population[0][0]) # 基因长度
        print("测试", len(population), pop_len, Gene_len)
        for i in range(len(population)):
            for j in range(pop_len):
                if (random.random() < self.pm):
                    mpoint = random.randint(0, Gene_len - 1)
                    print("循环内", i, j, mpoint)
                    # print(len(population[i][j][mpoint]))
                    print(len(population[i])) 
                    print(len(population[i][j]))
                    if (population[i][j][mpoint] == 1):
                        population[i][j][mpoint] = 0
                    else:
                        population[i][j][mpoint] = 1
                        
    # 找出当前种群中最好的适应度和对应的参数值
    def best(self, population_decimalism, fitness_value):
        pop_len = len(population_decimalism[0])
        bestparameters = []
        bestfitness = 0.0
        
        for i in range(0, pop_len):
            tmp = []
            if (fitness_value[i] > bestfitness):
                bestfitness = fitness_value[i]
                for j in range(len(population_decimalism)):
                    tmp.append(abs(population_decimalism[j][i] * self.max_value / (math.pow(2, self.chromosome_length) - 10)))
                    bestparameters = tmp
                    
        return bestparameters, bestfitness
        
    # 画适应度变化图
    @run.change_dir
    def plot(self, results):
        x = []
        y = []
        for i in range(self.iter_num):
            x.append(i + 1)
            y.append(results[i])
        plt.figure()
        plt.plot(x, y)
        plt.xlabel("Number of iteration")
        plt.ylabel("Value of fitness")
        plt.title("GA_SVM")
        plt.savefig("./output/GA_svm.png")
        plt.close()
        
    # 主函数
    def main(self):
        results = []
        parameters = []
        best_fitness = 0.0
        best_parameters = []
        # 初始化种群
        population = self.species_origin()
        print("测试2", population)
        print("测试3", len(population), len(population[0]), len(population[0][0]))
        # 迭代参数寻优
        for i in range(self.iter_num):
            # 计算适应值列表
            fitness_value = self.fitness(population)
            # 计算每个染色体的十进制取值
            population_decimalism = self.translation(population)
            # 寻找当前种群最优参数
            current_parameters, current_fitness = self.best(population_decimalism, fitness_value)
            # 与之前最优值对比
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_parameters = current_parameters
            print('迭代次数:',i,';最优参数:',best_parameters,';最优适应值:',best_fitness)
            results.append(best_fitness)
            parameters.append(best_parameters)
            
            ## 种群更新
            ## 选择
            self.selection(population,fitness_value)
            ## 交叉
            self.crossover(population)
            ## 变异
            self.mutation(population)
            
        results.sort()
        self.plot(results)
        print('最终参数值 :',parameters[-1])
        return parameters[-1]
"""

        
# 遗传算法SVM调参
@run.timethis
@run.change_dir
def GA_SVM():
    """
    population_size=2    
    chromosome_num = 2
    max_value=500
    chromosome_length=10
    iter_num = 100
    pc=0.6
    pm=0.01
    ga = GA(population_size, chromosome_num, chromosome_length, max_value, iter_num, pc, pm)
    best_params = ga.main()
    # 用新数据验证模型
    model = svm.SVR(kernel = "linear", C = best_params[0], epsilon = best_params[1])
    features, target = readData()
    X_train, X_test, y_train, y_test = splitTimeSeries(features, target)
    model.fit(X_train, y_train)
    testModel(model)
    """
    data = readData()
    features, target = data[0], data[1]
    X_train, X_test, y_train, y_test = splitTimeSeries(features, target)
    def object(p):
        C, epsilon = p
        clf = svm.SVR(kernel = "linear", C = C, epsilon = epsilon)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_train)
        score = testHighLow(y_test, y_pred)
        return 1.0-score
        
    ga = GA(func = object, size_pop = 50, n_dim = 2, lb = [1, 0.00001], ub = [100, 1.0], max_iter = 500, prob_mut = 0.01)
    best_x, best_y = ga.run()
    print("最佳参数及适应值")
    print(best_x, 1.0-best_y)
    # 画图
    plt.figure()
    Y_history = pd.DataFrame(ga.all_history_Y)
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
    Y_history.min(axis=1).cummin().plot(kind='line')
    plt.savefig("./output/ga_results.png")
    # 用新数据验证模型
    model = svm.SVR(kernel = "linear", C = best_x[0], epsilon = best_x[1])
    data = readData(start="2018-01-01", end="2018-12-31")
    features, target = data[0], data[1]
    X_train, X_test, y_train, y_test = splitTimeSeries(features, target)
    model.fit(X_train, y_train)
    testModel(model)
    # 用模型进行策略回测
    data = readData(start="2019-01-01", end="2019-12-31")
    print("策略回测结果:")
    price = data[2]
    bechk = data[2]
    feature = data[0]
    print(len(feature))
    pred = model.predict(feature)
    pred = np.insert(pred, 0, 0.0)
    scaler = MinMaxScaler(feature_range=(1, 2))
    print(price.head())
    date = price.index
    price = scaler.fit_transform(price.values)
    # print(len(pred), len(price), price[0][1], pred[0])
    close = [price[i][1] for i in range(len(price))]
    open = [price[i][0] for i in range(len(price))]
    # print(close[0], len(close))
    inputdata = pd.DataFrame({"close":close, "open":open, "date":date})
    inputdata = inputdata.drop(index = [0, 1, 2, 3])
    inputdata.set_index("date", inplace = True)
    inputdata["pred"] = pred
    # print(inputdata.info(), inputdata.head())
    # 进行交易回测
    cash = [1000000] # 初始资金
    stock = []             # 持股数量
    value = []             # 市值
    fee_rate = 1e-4   # 手续费率
    for i in range(0, len(inputdata)-1):
        # print(i, inputdata.pred[i+1], inputdata.close[i])
        if i == 0:
            # print("a")
            value.append(cash[i])
            stock.append(0)
        # 预测收盘价高于前一日，全仓买入
        elif inputdata.pred[i+1] > inputdata.close[i]:
            # print("b")
            price = inputdata.open[i+1]
            close = inputdata.close[i+1]
            money = cash[i-1]
            # print(money, price, cash)
            amount = math.floor(0.9*money/price)
            spend = amount*price*(1+fee_rate)
            cash.append(money - spend)
            stock.append(stock[i-1]+amount)
            value.append(cash[i] + stock[i]*close)
            # print(amount, spend, cash[i], stock[i], value[i])
            # print(value)
        elif inputdata.pred[i+1] <= inputdata.close[i]: # 否则全部卖出
            # print("c", i)
            # print(stock)
            if stock[i-1] == 0: # 持仓为0，啥也不干
                # print("c1")
                stock.append(0)
                cash.append(cash[i-1])
                value.append(cash[i])
            else: # 将持仓卖出
                # print("c2")
                price = inputdata.open[i+1]
                close = inputdata.close[i+1]
                amount = stock[i-1]
                income = amount*price*(1-fee_rate)
                cash.append(cash[i-1] + income)
                stock.append(0.0)
                value.append(cash[i])
                # print(amount, income, cash[i], stock[i], value[i])
                # print(value)
#        if i >= 35:
#            input("按任意键继续")
    # print(value, len(value))
    value.append(value[-1])
    inputdata["value"] = value
#    plt.figure()
#    plt.plot(value)
#    plt.savefig("./output/test_bt.png")
#    plt.close()
    # 计算回测值
    bt = BackTest(inputdata)
    results = bt.run()
    bt.draw()
    print(results)
            
            
#    bt = BackTest(value, bechk)
#    results = bt.run()
#    print(results)
    
    
# 测试scikit-opt
def testOpt():
    def demo_func(x):
        x1, x2, x3 = x
        return x1**2 + (x2-0.05)**2 + x3**2
        
    ga = GA(func = demo_func, size_pop = 500, n_dim = 3, lb = [-1, -10, -5], ub = [2, 10, 2], max_iter = 100)
    best_x, best_y = ga.run()
    print(best_x, best_y)
    
    
# 回测指标计算类
# 输入每日资产净值序列，计算各种回测指标
class BackTest:
    def __init__(self, data):
        self.data = data
        self.cost = [0.0]                         # 交易成本
        self.bk_results = pd.DataFrame()
        

    # 进行回测
    def run(self):
        # 生成收益率数据
        self.genReturn()
        
        # 计算回测指标
        self.evaluation()
        
        return self.bk_results
        
            
    # 生成收益率数据
    def genReturn(self):
        # 生成收益率数据
        self.return_value = pd.DataFrame()
        self.return_value["value"] = self.data["value"]
        self.return_value["returns"] = self.return_value["value"].pct_change()
        self.return_value["benchmark_returns"] = self.data["close"].pct_change().values
        # print(type(self.value), type(self.value.index))
        self.return_value["date"] = self.data.index
        self.return_value.index = self.return_value["date"]
            
    # 画结果
    def draw(self):
        oldpath = os.getcwd()
        newpath = "/home/code/"
        os.chdir(newpath)
        plt.figure()
        plt.plot(self.return_value["value"])
        plt.savefig("./output/backtest_value.png")
        plt.close()
        # 画每日收益率图
        plt.figure()
        plt.plot(self.return_value["returns"])
        plt.savefig("./output/backtest_returns.png")
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
    model = SVM()
    testModel(model)
    experiment()
    # GridSVM()
    # PSO_SVM()
    GA_SVM()
    # testOpt()
    