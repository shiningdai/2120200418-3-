# -*- coding: utf-8 -*-
###########################################################################
## data_loader.py
## 数据预处理
## 姓名：代素蓉
## 学号：2120200418
###########################################################################
import scipy.io as scio
import numpy as np
from PIL import Image
from scipy import misc
import matplotlib.pyplot as plt
from .spca import SimplePCA as SPCA

class DataLoader(object):
    def __call__(self, path=r'D:\Daisurong\NKICS_DSR\Projects\Class_Projects\PatternRecognition\ThirdPro\src\datasets\FaceImages.mat'):
        self.data_path = path
        return self.call()

    def call(self):
        datas = scio.loadmat(self.data_path)
        self.datas = datas
        test_female = datas['test_female']
        train_female = datas['train_female']
        test_male = datas['test_male']
        train_male = datas['train_male']
        self.test_nums = len(test_female) + len(test_male)  # 27
        self.train_nums = len(train_female) + len(train_male)  # 51
        x_train = np.concatenate((train_male, train_female), axis=0)
        x_test = np.concatenate((test_male, test_female), axis=0)

        # 首先将像素值处理为[0,1]范围内的值
        x_train = x_train/255.
        x_test = x_test/255.

        # 降维最少不能小于数据集的最小的一个维度
        self.n_components = min(x_train.shape)
        self.spac = SPCA(self.n_components)
        x_train = self.spac(x_train)

        # 测试集减去训练集降维时的均值， 并乘以训练集降维时的转换矩阵

        x_test = x_test - self.spac.means
        x_test = np.matmul(x_test, self.spac.transform_mat)

        self.x_train = x_train
        self.x_test = x_test
        y_train, y_test = self.set_label()
        self.y_train = np.array(y_train)
        self.y_test = np.array(y_test)
        return self.x_test, self.y_test, self.x_train, self.y_train


    def set_label(self):
        # 为数据集添加标签，设male对应的标签为-1， female对应的标签为1
        y_train = []
        y_test = []
        for i in range(len(self.datas["train_male"])):
            y_train.append(0)
        for i in range(len(self.datas["train_female"])):
            y_train.append(1)
        for i in range(len(self.datas["test_male"])):
            y_test.append(0)
        for i in range(len(self.datas["test_female"])):
            y_test.append(1)

        return y_train, y_test







