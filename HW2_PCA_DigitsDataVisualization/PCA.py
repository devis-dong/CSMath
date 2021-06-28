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


def PCA(Y:np.ndarray, p=2):
    U, S, UT = np.linalg.svd(Y.T)
    baseX, W = U[:, 0:p].T, np.matmul(np.diag(S[0:p]), UT[0:p, :])
    Y_hat = np.matmul(W.T, baseX)
    return baseX, W, Y_hat

def decompose(Y:np.ndarray, baseX):
    return np.dot(baseX, Y.T)

def readData(file_name):
    datas = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if '3' == line[-1]:
                datas.append([int(i) for i in line.split(',')[:-1]])
    return np.array(datas)

def train(file_name):
    Y = readData(file_name)
    baseX, W, Y_hat = PCA(Y, 2)
    return baseX, W, Y, Y_hat

def test(file_name, baseX):
    Y = readData(file_name)
    W = decompose(Y, baseX)
    Y_hat = np.dot(W.T, baseX)
    return Y, Y_hat

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
        if 0 == i:
            img = tmp
        else:
            img = np.concatenate((img, tmp), axis=0)
    return img

def showImg(Y:np.ndarray, h=8, w=8, title=''):
    img = concatenateData(Y, h, w)
    plt.figure()
    plt.title(title)
    plt.imshow(img)
    # plt.show()

def main():
    print('running...')
    baseX, W, Ytra, Ytra_hat = train('optdigits.tra')
    showImg(Ytra, 8, 8, title='train origin')
    showImg(Ytra_hat, 8, 8, title='train pca')
    Ytes, Ytes_hat = test('optdigits.tes', baseX)
    showImg(Ytes, 8, 8, title='test origin')
    showImg(Ytes_hat, 8, 8, title='test pca')
    plt.show()
    print('done!')

if __name__ == '__main__':
    main()


    

    


