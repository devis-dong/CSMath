from matplotlib import legend
import numpy as np
import matplotlib.pyplot as plt

def func(x):
    return np.sin(x)

def sampleWithNoise(func, n, loc=0.0, scale=0.2):
    x = np.linspace(0, 2*np.pi, n)
    y = func(x) + np.random.normal(loc=loc, scale=scale, size=n)
    return x, y

def sampleFunc(func, n):
    x = np.linspace(0, 2*np.pi, n)
    y = func(x)
    return x, y

def generateX(x, m):
    x = x.reshape((-1, 1))
    n = x.shape[0]
    e = np.tile(np.arange(m+1).reshape((1, -1)), (n, 1))
    X = np.power(np.tile(x, (1, m+1)), e)
    return X

def polynomialWeigts(x, y, m, lamda=0):
    X = generateX(x, m)
    Y = y.reshape((-1, 1))
    W = np.dot(np.linalg.inv(np.dot(X.T, X)+lamda*np.eye(m+1)), np.dot(X.T, Y))
    return W

def fitFunc(x, W):
    W = W.reshape((-1, 1))
    X = generateX(x, W.shape[0]-1)
    Y = np.dot(X, W)
    return Y

def demo(x, y, m, lamda=0, func=func):
    plt.figure()
    plt.title('n=%s, m=%s, lamda=%s' % (len(x), m, lamda))
    plt.scatter(x, y, c='b', marker='x', label='samples')
    #optimize the weights
    W = polynomialWeigts(x, y, m, lamda)
    #plot ground truth line
    x_gt, y_gt = sampleFunc(func, 1000)
    plt.plot(x_gt, y_gt, 'g', label='groung truth')
    #plot the fitting line
    y_pre = fitFunc(x_gt, W).reshape(y_gt.shape)
    plt.plot(x_gt, y_pre, 'r', label='fitting line')
    plt.legend()
    # plt.show()

def run():
    x, y = sampleWithNoise(func, 10)
    demo(x, y, 3, 0, func)
    demo(x, y, 9, 0, func)
    x, y = sampleWithNoise(func, 100)
    demo(x, y, 3, 0, func)
    demo(x, y, 9, 0, func)
    demo(x, y, 9, np.exp(-18), func)
    # demo(x, y, 9, np.exp(3), func)
    plt.show()

def main():
    run()

if __name__ == '__main__':
    main()