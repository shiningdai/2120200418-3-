# -*- coding: utf-8 -*-
###########################################################################
## pocket_perception.py
## 基于感知器的pocket算法实现
## 姓名：代素蓉
## 学号：2120200418
###########################################################################
import numpy as np
from datasets.data_loader import DataLoader
from copy import deepcopy

def fun_decide(XX, W):
    # 判别式函数
    g = np.matmul(W, XX)
    if g>0:
        return 0
    else:
        return 1
def fun_delta(X, W, label):
    g = fun_decide(X, W)
    if g == label:
        # 决策正确， 代价为0
        return 0
    elif g == 1 :
        # 决策错误， x为w1类, 判别为w2类
        return -1
    else:
        # 决策错误， 且为w2类
        return 1

def fun_loss(W, X, Y):
    losses = 0
    w_gradients = np.zeros(len(W))
    WT = W.T
    for i in range(len(X)):
        delta_x = fun_delta(X[i], W, Y[i])
        losses += delta_x * np.matmul(WT,X[i])
        w_gradients += delta_x*X[i]
    return losses, w_gradients

def accuracy(W, X, Y):
    n_correct = 0.
    x_size = X.shape[0]
    for i in range(x_size):
        predict = fun_decide(X[i], W)
        if predict == Y[i]:
            n_correct += 1
    return n_correct/x_size, n_correct

def perception(W, X, Y, learning_rate=0.5):
    losses, w_gradients = fun_loss(W, X, Y)
    print("loss: ", losses)
    print("learning_rate: ", learning_rate)
    lr = learning_rate
    return W - lr*w_gradients

def pockets(x_train, y_train, iter_times=10, learning_rate=0.1):
    # 随机初始化权重向量
    n_features = x_train.shape[1]
    weights = np.random.rand(n_features)
    print(weights.shape)  # (52,)
    # print(weights)
    # 暂存权重 ws
    ws = deepcopy(weights)
    # 暂存变量 hs = 0
    hs = 0
    lr = learning_rate
    acc, n_correct = accuracy(weights, x_train, y_train)
    print("before training: " )
    print("training accuracy:", acc)
    print("correct decisions:", n_correct)
    print("-------------------------------------------")
    for i in range(iter_times):
        weights = perception(weights, x_train, y_train, lr)
        acc, n_correct = accuracy(weights, x_train, y_train)
        print("<====== % d-th training =======> "%(i+1))
        print("training accuracy:", acc)
        print("correct decisions:", n_correct)
        print("-------------------------------------------")
        if n_correct > hs:
            hs = n_correct
            ws = deepcopy(weights)
        if lr>0.00001:
            lr /= 1.5
        if n_correct==len(x_train):
            break
    return ws


if __name__=="__main__":
    data_path = r'D:\Daisurong\NKICS_DSR\Projects\Class_Projects\PatternRecognition\ThirdPro\src\datasets\FaceImages.mat'
    datas = DataLoader()
    x_test, y_test, x_train, y_train = datas(data_path)
    # 将 d 维的特征向量变为d+1维，便于采用线性判别式方程的齐次坐标表示 ，模式特征的维度+1 （+1作为偏置项）
    train_bias = np.ones(len(x_train))
    test_bias = np.ones(len(x_test))
    x_train = np.c_[x_train, train_bias.T]
    x_test = np.c_[x_test, test_bias.T]

    # print(x_train.shape) # (51, 52)
    # print(y_train.shape) # (51,)
    # print(x_test.shape) # (27, 52)
    # print(y_test.shape) # (27,)

    # 首先打乱训练数据集
    state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)

    # Pocket算法的迭代次数 iter_times
    iter_times = 20
    weights = pockets(x_train, y_train, iter_times=iter_times)
    test_acc, test_correc = accuracy(weights, x_test, y_test)
    print("test accuracy:", test_acc)
    print("correct decisions:", test_correc)

    # 通过调整迭代次数iter_times， 初始学习率， 终止学习率 if lr>0.00001: lr /= 2  学习率变化率lr /= 1.2   改变结果
    # train 准确率最高的时候， test准确率不是最高的









