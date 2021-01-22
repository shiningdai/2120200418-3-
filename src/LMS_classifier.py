# -*- coding: utf-8 -*-
###########################################################################
## LMS_classifier.py
## 基于LMS的分类器实现
## 姓名：代素蓉
## 学号：2120200418
###########################################################################
import numpy as np
from datasets.data_loader import DataLoader
from copy import deepcopy

def least_mean_square(x_train, y_train, iter_times=10, learning_rate=0.5):
    # 随机初始化权重向量
    n_features = x_train.shape[1]
    weights = np.random.rand(n_features)
    print(weights.shape)  # (52,)
    lr = learning_rate
    for i in range(iter_times):
        # mloss = fun_mloss(weights, x_train, y_train)
        acc, n_correct = accuracy(weights, x_train, y_train)
        print("===========================================")
        print("learning rate:", lr)
        print("train accuracy:", acc)
        print("train n_correct: ", n_correct)
        if n_correct == len(x_train):
            print(" break at %d-th iteration"%(i))
            break
        losses = []
        for i in range(len(x_train)):
            predict = sign(np.matmul(x_train[i].T, weights))
            _delta = x_train[i] * (y_train[i]-predict)
            weights += lr*_delta
            losses.append((y_train[i] - predict) ** 2)
        # lr /= 2
        loss = np.mean(losses)
        print("loss:", loss)
        if loss < 1e-7:
            print(" break at %d-th iteration" % (i))
            break
    return weights

# def fun_mloss(W, X, Y):
#     losses = []
#     for i in range(len(X)):
#         losses.append((Y[i] - np.matmul(W.T, Y[i])) ** 2)
#     return np.mean(losses)

def sign(value):
    if value>0:
        return 1
    else:
        return 0
def accuracy(W, X, Y):
    n_correct = 0.
    x_size = X.shape[0]
    for i in range(x_size):
        predict = sign(np.matmul(X[i], W))
        if predict == Y[i]:
            n_correct += 1
    return n_correct/x_size, n_correct

if __name__=="__main__":
    data_path = r'D:\Daisurong\NKICS_DSR\Projects\Class_Projects\PatternRecognition\ThirdPro\src\datasets\FaceImages.mat'
    datas = DataLoader()
    x_test, y_test, x_train, y_train = datas(data_path)

    # 将 d 维的特征向量变为d+1维，便于采用线性判别式方程的齐次坐标表示 ，模式特征的维度+1 （+1作为偏置项）
    train_bias = np.ones(len(x_train))
    test_bias = np.ones(len(x_test))
    x_train = np.c_[x_train, train_bias.T]
    x_test = np.c_[x_test, test_bias.T]

    # 首先打乱训练数据集
    state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)


    weights = least_mean_square(x_train, y_train, iter_times=10)
    test_acc, test_correc = accuracy(weights, x_test, y_test)
    print("----------------------------------------------------")
    print("test accuracy:", test_acc)
    print("correct decisions:", test_correc)

    # 通过调整迭代次数iter_times(10, 30 , 50, 60, 70, 80)， 学习率 0.1， 0.05， 0.01
    # iter_times=80，90, lr=0.01 在训练集上略好一点（约60%）， 但在测试集上很烂