#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   PCA.py
@Time    :   2021/06/28 18:53:37
@Author  :   devis dong 
@Version :   1.0.0
@Contact :   devis.dong@gmail.com
@License :   (C)Copyright 2020-2021, ZJU
@Desc    :   None
'''

# here put the import lib
import numpy as np
import matplotlib.pyplot as plt


def readData(file_name):
    datas = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if '3' == line[-1]:
                datas.append([int(i) for i in line.split(',')[:-1]])
    return np.array(datas)

def computeBase(Y, p=2):
    # K = np.dot(Y.T, Y)
    # eigenval, eigenvec = np.linalg.eig(K)
    # idx = np.argsort(eigenval)[::-1][0:p]
    # baseE = eigenvec[:, idx].T
    baseE = np.linalg.svd(Y.T)[0][:, 0:p].T
    return baseE

def PCA(X, baseE=None, p=2):
    if baseE is None:
        baseE = computeBase(X, p)
    return np.dot(X, baseE.T)

def computeW(K, p=2):
    # eigenval_K, eigenvec_K = np.linalg.eig(K)
    # eigenvec_W = eigenvec_K / np.abs(eigenval_K)
    # cols = np.argsort(eigenval_K)[::-1][0:p]
    # W = eigenvec_W[:, cols].T
    U, S, _ = np.linalg.svd(K)
    W = (U[:, 0:p]/((S[0:p]))).T
    return W

def kernel0(Y, d=1):
    return np.dot(Y.T, Y)**d

def KPCA(X, K=None, p=2):
    if K is None:
        K = kernel0(X)
    return np.dot(np.dot(X, K), computeW(K, p).T)

def concatenateData(Y:np.ndarray, h, w):
    n, d = Y.shape
    assert d == h*w
    rows = int(np.floor(n**0.5))
    cols = int(np.ceil(n/rows))
    if rows*cols > n:
        Y = np.concatenate((Y, np.zeros((rows*cols-n, d))), axis=0)
    X = Y.reshape(rows, cols, h, w)
    img = np.empty((0, 0))
    for i in range(rows):
        tmp = X[i, 0]
        for j in range(1, cols):
            tmp = np.concatenate((tmp, X[i, j]), axis=1)
        img = tmp if 0 == i else np.concatenate((img, tmp), axis=0)
    return img

def showImg(Y:np.ndarray, h=8, w=8, title=''):
    img = concatenateData(Y, h, w)
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    # plt.show()

def showPoints(Y:np.ndarray, title=''):
    dim0, dim1 = Y[:, 0], Y[:, 1]
    plt.figure()
    plt.title(title)
    plt.xlabel('first principle component')
    plt.ylabel('second principle component')
    plt.scatter(dim0, dim1)
    # plt.show()

def testPCA(p=2):
    Ytra, Ytes = readData('data\optdigits.tra'), readData('data\optdigits.tes')
    baseE = computeBase(Ytra, p)
    Ytra_reduced, Ytes_reduced = PCA(Ytra, baseE), PCA(Ytes, baseE)
    Ytra_hat, Ytes_hat = np.dot(Ytra_reduced, baseE), np.dot(Ytes_reduced, baseE)
    showImg(Ytra, 8, 8, title='train origin')
    showImg(Ytra_hat, 8, 8, title='train pca')
    showImg(Ytes, 8, 8, title='test origin')
    showImg(Ytes_hat, 8, 8, title='test pca')
    showPoints(Ytra_reduced, title='PCA reduced train points')
    showPoints(Ytes_reduced, title='PCA reduced test points')
    # plt.show()
    return Ytra_reduced, Ytes_reduced

def testKPCA(p=2):
    Ytra, Ytes = readData('data\optdigits.tra'), readData('data\optdigits.tes')
    K = kernel0(Ytra, d=0.5)
    Ytra_reduced, Ytes_reduced = KPCA(Ytra, K, p), KPCA(Ytes, K, p)
    showPoints(Ytra_reduced, title='KPCA reduced train points')
    showPoints(Ytes_reduced, title='KPCA reduced test points')
    # plt.show()
    return Ytra_reduced, Ytes_reduced

def distance(X, Y):
    assert X.shape == Y.shape
    dis = np.mean(np.sum((X-Y)**2, axis=1)**0.5)
    return dis

def main():
    print('running...')
    p = 2
    Ytra_PCA, Ytes_PCA = testPCA(p)
    Ytra_KPCA, Ytes_KPCA = testKPCA(p)
    plt.show()
    print('distance:\n', 'train points:', distance(Ytra_PCA, Ytra_KPCA), 'test points', distance(Ytes_PCA, Ytes_KPCA))
    print('done!')

if __name__ == '__main__':
    main()


    

    



