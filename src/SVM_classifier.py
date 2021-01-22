# -*- coding: utf-8 -*-
###########################################################################
## SVM_classifier.py
## 基于SVM的分类器实现
## 姓名：代素蓉
## 学号：2120200418
###########################################################################
from sklearn import svm
import numpy as np
from datasets.data_loader import DataLoader

def accuracy(predict, true_labels):
    x_size = float(len(predict))
    n_correct = np.sum(predict==true_labels)
    acc = n_correct/x_size
    return n_correct, acc

if __name__=="__main__":
    data_path = r'D:\Daisurong\NKICS_DSR\Projects\Class_Projects\PatternRecognition\ThirdPro\src\datasets\FaceImages.mat'
    datas = DataLoader()
    x_test, y_test, x_train, y_train = datas(data_path)

    # 首先打乱训练数据集
    state = np.random.get_state()
    np.random.shuffle(x_train)
    np.random.set_state(state)
    np.random.shuffle(y_train)

    c = 0.75
    # g = "auto"
    g = 0.01
    '''
    SVM分类器， 采用多项式核函数,kernel='poly'
    '''
    poly_model = svm.SVC(kernel='poly', C=c, gamma=g)
    poly_model.fit(x_train, y_train)
    poly_score = poly_model.score(x_train, y_train)
    print("poly score:", poly_score)
    poly_predict = poly_model.predict(x_test)
    # print("poly_predict:", poly_predict)
    test_acc, test_correc = accuracy(poly_predict, y_test)
    print("poly test accuracy:", test_acc)
    print("correct decisions:", test_correc)

    '''
        SVM分类器， 采用多径向基核函数,kernel='rbf'
    '''
    rbf_model = svm.SVC(kernel='rbf', C=c, gamma=g)

    rbf_model.fit(x_train, y_train)
    rbf_score = rbf_model.score(x_train, y_train)
    print("rbf score:", rbf_score)
    rbf_predict = rbf_model.predict(x_test)
    # print("rbf_predict:", rbf_predict)
    test_acc, test_correc = accuracy(rbf_predict, y_test)
    print("rbf test accuracy:", test_acc)
    print("correct decisions:", test_correc)

    print("poly_predict:", poly_predict)
    print("true label:", y_test)
    print("rbf_predict:", rbf_predict)
