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

def PCA(Y:np.ndarray, p=2):
    U, S, UT = np.linalg.svd(Y.T)
    baseX, W = U[:, 0:p].T, np.matmul(np.diag(S[0:p]), UT[0:p, :]).T
    Y_hat = np.matmul(W, baseX)
    return W, baseX, Y_hat

def reduceDim(Y:np.ndarray, baseX):
    return np.dot(Y, baseX.T)

def train(file_name, p=2):
    Y = readData(file_name)
    W, baseX, Y_hat = PCA(Y, p)
    return W, baseX, Y, Y_hat

def test(file_name, baseX):
    Y = readData(file_name)
    W = reduceDim(Y, baseX)
    Y_hat = np.dot(W, baseX)
    return W, Y, Y_hat

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

def main():
    print('running...')
    Ytra_reduced, baseX, Ytra, Ytra_hat = train('data\optdigits.tra', p=2)
    showImg(Ytra, 8, 8, title='train origin')
    showImg(Ytra_hat, 8, 8, title='train pca')
    Ytes_reduced, Ytes, Ytes_hat = test('data\optdigits.tes', baseX)
    showImg(Ytes, 8, 8, title='test origin')
    showImg(Ytes_hat, 8, 8, title='test pca')
    showPoints(Ytra_reduced, title='train points')
    showPoints(Ytes_reduced, title='test points')
    plt.show()
    print('done!')

if __name__ == '__main__':
    main()


    

    



