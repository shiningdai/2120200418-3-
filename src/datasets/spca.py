# -*- coding: utf-8 -*-
###########################################################################
## spca.py
## 简单的PCA算法实现
## 姓名：代素蓉
## 学号：2120200418
###########################################################################
import scipy.io as scio
import numpy as np
from sklearn.decomposition import PCA

class SimplePCA():
    def __init__(self,componets = 51, axis=0, ):
        self.reduce_axis = axis  # 降维的矩阵维度，默认对列降维
        self.saved_components = componets # 降维后保留的特征个数
        self.means = None # 样本数据降维所在维度的均值
        self.transform_mat = None # 由样本数据协方差矩阵最大的N个特征值对应的特征向量构成的转换矩阵

    def __call__(self, datas):
        return self.call(datas)

    def call(self, datas):
        '''
        进行PCA降维处理
        :param datas: 待降维的样本数据
        :return: 降维后的数据
        '''
        means = np.mean(datas, axis=self.reduce_axis)
        self.means = means
        # 样本数据中心化,去平均值,每一列减去其平均值
        mean_removed = datas - means

        self.mean_removed = mean_removed
        # 求中心化后样本数据的协方差矩阵
        covariance_mat = np.cov(mean_removed, rowvar=False)
        self.covariance_mat = covariance_mat
        # 协方差矩阵的特征值和特征向量
        # print("covariance_mat.shape",covariance_mat.shape)
        # print(np.mat(covariance_mat))
        eig_values, eig_vectors = np.linalg.eigh(covariance_mat)
        self.eig_values = eig_values
        self.eig_vectors = eig_vectors
        # print("----------------------------------")
        # print(eig_values)
        # print(eig_vectors)
        # 取最大的 saved_components 个特征值
        # 特征值的索引降序排列
        _index = np.argsort(eig_values)[::-1]
        max_index = _index[ : self.saved_components]

        # 取最大的 saved_components 个特征值对应的saved_components个特征向量构成转换矩阵
        transform_mat = eig_vectors[:,max_index ]
        self.transform_mat = transform_mat

        # 用转换矩阵将原始数据转换到降维后的空间， 并进一步得到最终的降维后的数据集
        transformed_datas = np.matmul(mean_removed , transform_mat)

        self.transformed_datas = transformed_datas
        # retransformed_datas = self.low_dimension_space * self.transform_mat + self.means

        return transformed_datas


if __name__ == "__main__":
    # 测试一下
    X = np.array([[-1., -1., 3, 7], [-2., -1., 3, 2], [-3., -2., 3, 8], [1., 1., 1, 9], [2., 1., 5, 7], [3., 2., 2, 9]])
    # 调用sklearn库中的PCA测试
    pca = PCA(n_components=2)
    # pca.fit(X)
    pca_datas = pca.fit_transform(X)
    spca = SimplePCA(componets=2)
    spca_datas = spca(X)
    print(X.shape)
    print(spca.means.shape)
    print(spca.transform_mat.shape)
    print(spca_datas.shape)

    print("--------------------------------------")
    print(pca_datas)
    print("--------------------------------------")
    print(spca_datas)

    XP = np.array([[-1., -1., 3, 7],  [1., 1., 1, 9], [2., 1., 5, 7], [3., 2., 2, 9]])
    print(XP.shape)
    XP -= spca.means
    XP = np.matmul(XP , spca.transform_mat)
    print(XP.shape)





