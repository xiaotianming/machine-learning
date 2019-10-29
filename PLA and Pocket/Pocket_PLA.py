import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt


# 1、导入数据集
def loaddata():
    data1 = pd.read_csv("data2.csv",header = None)
    samples = data1.iloc[:,:2].values
    labels = data1.iloc[:,2].values
    return samples, labels


# 训练感知机模型
class Pocket_PLA:
    def __init__(self, x, y, a=1):
        self.x = x
        self.y = y
        self.w = np.zeros((x.shape[1], 1))  # 初始化权重，w1,w2均为0
        self.b = 0
        self.best_w = np.zeros((x.shape[1], 1))  # 最好
        self.best_b = 0
        self.a = 1  # 学习率
        self.numsamples = self.x.shape[0]
        self.numfeatures = self.x.shape[1]

    def sign(self, w, b, x):
        y = np.dot(x, w) + b
        return int(y)

    def update(self, label_i, data_i):
        tmp = label_i * self.a * data_i
        tmp = tmp.reshape(self.w.shape)
        # 更新w和b
        tmpw = tmp + self.w
        tmpb = self.b + label_i * self.a
        # print(len(self.classify(self.w,self.b)),',',len(self.classify(tmpw,tmpb)))
        if(len(self.classify(self.w,self.b))>=(len(self.classify(tmpw,tmpb)))):
            self.best_w = tmp + self.w
            self.best_b = self.b + label_i * self.a
        self.w = tmp + self.w
        self.b = self.b + label_i * self.a

    def classify(self,w1,b1):
        mistkaes = []
        for i in range(self.numsamples):
            tmpY = self.sign(w1, b1, self.x[i, :])
            if tmpY * self.y[i] <= 0:  # 如果是一个误分类实例点
                mistkaes.append(i)
        return mistkaes

    def train(self,num):
        count = 0
        isFind = False
        while not isFind:
            mistakes = self.classify(self.w,self.b)
            if(len(mistakes) == 0):
                print('最终训练得到的w和b为：', self.best_w, self.best_b)
                break
            n = mistakes[random.randint(0,len(mistakes)-1)]
            self.update(self.y[n], self.x[n, :])
            print('第',count,'次迭代误分类点为：', self.x[n, :], '此时的w和b为：', self.w, self.b)
            count += 1
            if count == num:
                print( '最终训练得到的w和b为：', self.best_w, self.best_b)
                isFind = True
        return self.best_w, self.best_b

# 画图描绘
class Picture:
    def __init__(self, data, w, b,l):
        self.b = b
        self.w = w
        plt.figure(1)
        plt.title('Pocket_PLA Learning Algorithm', size=14)
        plt.xlabel('x0-axis', size=14)
        plt.ylabel('x1-axis', size=14)

        xData = np.linspace(0,10, 100)
        yData = self.expression(xData)
        plt.plot(xData, yData, color='r', label='sample data')
        for i in range(0,len(l)):
            if (l[i]>0):
                plt.scatter(data[i][0], data[i][1],color='b', s=50)
            else:
                plt.scatter(data[i][0], data[i][1],color='g', s=50, marker='x')
        plt.savefig('Pocket2.png', dpi=75)

    def expression(self, x):
        y = (-self.b - self.w[0] * x) / self.w[1]  # 注意在此，把x0，x1当做两个坐标轴，把x1当做自变量，x2为因变量
        return y

    def Show(self):
        plt.show()


if __name__ == '__main__':
    start = time.time()
    samples, labels = loaddata()
    # u = np.mean(samples,axis=0)
    # o = np.std(samples, axis=0)
    # samples = (samples - u)/o
    myPocket_PLA = Pocket_PLA(x=samples, y=labels)
    weights, bias = myPocket_PLA.train(50000)
    Picture = Picture(samples, weights, bias,labels)
    Picture.Show()
    costtime = time.time() - start
    print("Time used:",costtime)