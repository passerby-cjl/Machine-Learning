import copy
import time


import numpy as np
import cupy as cp
#cp for gpu compile, np for cpu compile
import random


class DNNModel:
    lamdas = [0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1,2,4,8,10]
    epsolum = 0.1
    def __init__(self, S_l:list, learning_rate:float, lamda:int=0, theta=None):
        self.K = S_l[-1][0] #分成几类
        self.S_l = S_l #每层的神经元个数
        self.L = len(self.S_l)-1 #神经网络共几层（包括输出层不包括输入层）
        self.m = S_l[0][0]
        self.learning_rate = learning_rate #学习率
        self.lamda = self.lamdas[lamda] if lamda >= 0 else -lamda #正则化项
        self.theta=theta
        self.active={"ReLU":activeReLU, "Sig":activeSig, "Tanh":activeTanh, "Liner":activeLiner}
        self.derivative={"ReLU":derivativeReLU, "Sig":derivativeSig, "Tanh":derivativeTanh, "Liner":derivativeLiner}
        if theta is None:
            self._random_init()
        return
    def _random_init(self): #初始化所有theta
        self.theta=[]
        for l in range(0, self.L):
            self.theta.append([])
            for x in range(0, self.S_l[l+1][0]):
                self.theta[l].append([])
                for y in range(0, self.S_l[l][0]+1):
                    self.theta[l][x].append(self.epsolum * (random.random() - 0.5))
            self.theta[l] = np.array(self.theta[l], dtype='float32').T
        return

    def hypothesis(self, input_layor): #num = 训练样本编号 #预测函数
        next_level = np.array(input_layor.copy(), dtype='float32', ndmin=2)
        next_level=np.append(np.ones((next_level.shape[0],1)), next_level, axis=1)
        hypomatrix = [next_level]
        Z=[]
        for l in range(0, self.L):
            next_level=self.active[self.S_l[l+1][1]](np.dot(next_level, self.theta[l]))
            if l < self.L-1:
                next_level=np.append(np.ones((next_level.shape[0],1)), next_level, axis=1)
            hypomatrix.append(next_level)
        return hypomatrix

    def train(self, input_layor, output_layor): #对代价函数求偏导
        hypomatrix = self.hypothesis(input_layor)
        output_layor=np.array(output_layor, dtype='float32')
        delta = [np.multiply(np.subtract(hypomatrix[-1] , output_layor).T, self.derivative[self.S_l[-1][1]](hypomatrix[-1]))]
        D = [np.dot(delta[0].copy(), hypomatrix[self.L - 1])]
        delta[0] = np.append(np.zeros((1, delta[0].shape[1])), delta[0], axis=0)
        for l in range(self.L - 1, 0, -1):
            delta.insert(0, np.multiply(np.dot(self.theta[l], delta[0][1:]), self.derivative[self.S_l[l][1]](hypomatrix[l])))
            D.insert(0, np.dot(delta[0][1:], hypomatrix[l-1]))
        for l in range(0, self.L):
            tempmtx = self.theta[l].copy()
            if l:
                tempmtx[:, 0] = 0
            self.theta[l] -= self.learning_rate * (D[l].T + self.lamda * tempmtx) / len(input_layor)


train_set = [[[1,1],[2,1],[4,1],[1,3],[2,3],[3,3],[2,2],[1,2],[4,2]],[[1,0,0,0],[0,1,0,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]]
cv_set = [[[3,4],[2,6],[1,5]],[[0,0,1,0],[0,1,0,0],[1,0,0,0]]]
test_set=[[[3,5]],[[0,0,1,0]]]
minerror=1e9
final_model=None
def activeSig(x):
    return 1.0 / (1 + np.exp(-x))
def derivativeSig(x):
    return np.multiply(x,(1-x)).T
def activeReLU(x):
    return np.maximum(x,0)
def derivativeReLU(x):
    return np.greater(x,np.zeros(x.shape)).T.astype(float)
def activeTanh(x):
    return np.tanh(x)
def derivativeTanh(x):
    t=np.tanh(x)
    return (np.ones(x.shape)-np.multiply(t,t)).T
def activeLiner(x):
    return x
def derivativeLiner(x):
    return np.ones(x.shape).T

if __name__ == "__main__":
    time0=time.time()
    for mod in range(13):
        temp_model = DNNModel([(2, ""),(64, "ReLU"), (64, "ReLU"), (4, "Sig")], 0.01, lamda=mod)
        for times in range(0,5000):
            temp_model.train(train_set[0], train_set[1])
        cv_model=DNNModel(temp_model.S_l, 0, theta=temp_model.theta)
        errors = np.sum(np.square(cv_model.hypothesis(cv_set[0])[-1]-np.array(cv_set[1])))
        print(errors)
        if minerror>errors:
            final_model=copy.deepcopy(temp_model)
            minerror=errors
    print(time.time()-time0)
    print(final_model.hypothesis(test_set[0][0])[-1])


