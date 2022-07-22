import copy
import time

import numpy as np
import cupy as cp
#cp for gpu compile, np for cpu compile
import random


class DNNModel:
    lamdas = [0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1,2,4,8,10]
    epsolum = 0.1
    type='DNN'
    def __init__(self, S_l:list, learning_rate:float, lamda:int=0, theta=None):
        self.K = S_l[-1][0] #分成几类
        self.S_l = S_l #每层的神经元个数
        self.L = len(self.S_l)-1 #神经网络共几层（包括输出层不包括输入层）
        self.learning_rate = learning_rate #学习率
        self.lamda = self.lamdas[lamda] if lamda >= 0 else -lamda #正则化项
        self.theta=theta
        self.active={"ReLU":activeReLU, "Sig":activeSig, "Tanh":activeTanh, "Liner":activeLiner, "Softmax":activeSoftmax}
        self.derivative={"ReLU":derivativeReLU, "Sig":derivativeSig, "Tanh":derivativeTanh, "Liner":derivativeLiner, "Softmax":derivativeSoftmax}
        if theta is None:
            self._random_init()
        return
    def _random_init(self): #初始化所有theta
        self.theta=[]
        for l in range(self.L):
            self.theta.append([])
            for x in range(self.S_l[l+1][0]):
                self.theta[l].append([])
                nargs=self.S_l[l][0]
                if self.type=='CNN':
                    nargs *= self.S_l[l+1][1]**2
                for y in range(nargs+1):
                    self.theta[l][x].append(self.epsolum * (random.random() - 0.5))
            self.theta[l] = np.array(self.theta[l], dtype='float32').T
        return

    def _preproceed_ims(self, x, layourinfo):
        return np.append(np.ones((*x.shape[0:-1], 1)), x, axis=-1) # 增加偏置

    def _preproceed_delta(self, delta, layourinfo, inputshape):
        return delta

    def _preproceed_theta(self, theta, layourinfo):
        return theta[1:].T

    def hypothesis(self, input_layour): #num = 训练样本编号 #预测函数
        next_level = np.array(input_layour, dtype='float32', ndmin=2)
        hypomtx=[next_level]
        colhypomtx=[]
        for l in range(self.L):
            next_level=self._preproceed_ims(next_level, self.S_l[l+1])
            colhypomtx.append(next_level)
            next_level=self.active[self.S_l[l+1][-1]](np.dot(next_level, self.theta[l]))
            hypomtx.append(next_level)
        return hypomtx, colhypomtx

    def train(self, input_layour, output_layour): #对代价函数求偏导
        hypomtx, colhypomtx = self.hypothesis(input_layour)
        output_layour=np.array(output_layour, dtype='float32')
        delta = np.subtract(hypomtx[-1], output_layour)
        grad = np.dot(delta.reshape(-1,delta.shape[-1]).T, colhypomtx[self.L - 1].reshape(-1, colhypomtx[self.L-1].shape[-1]))
        tempmtx = self.theta[-1].copy()
        tempmtx[:, 0] = 0
        self.theta[-1] -= self.learning_rate * (grad.T + self.lamda * tempmtx) / len(input_layour)
        for l in range(self.L - 1, 0, -1):
            pdelta=self._preproceed_delta(delta, self.S_l[l+1], hypomtx[l].shape)
            ptheta=self._preproceed_theta(self.theta[l], self.S_l[l])
            delta = np.multiply(np.dot(pdelta, ptheta), self.derivative[self.S_l[l][-1]](hypomtx[l]))
            grad=np.dot(delta.reshape(-1,delta.shape[-1]).T, colhypomtx[l-1].reshape(-1, colhypomtx[l-1].shape[-1]))
            tempmtx = self.theta[l-1].copy()
            tempmtx[0, :] = 0
            self.theta[l-1] -= self.learning_rate * (grad.T + self.lamda * tempmtx) / len(input_layour)

    def save(self, filename='mymodel/DNNMODEL.txt'):
        with open(filename, "w") as f:
            f.write("TYPE:%s\n"%self.type)
            f.write("STRUCTURE:%s\n"%str(self.S_l))
            f.write("LAMDA:%s\n"%str(-self.lamda))
            for l in range(self.L):
                f.write("%s\n"%str(self.theta[l].tolist()))

def load(filename='mymodel/DNNMODEL.txt'):
    with open(filename, "r") as f:
        if f.readline()!="TYPE:DNN\n":
            raise Exception("Wrong type of model to read")
        S_l=eval(f.readline().split(":")[-1])
        lamda=eval(f.readline().split(":")[-1])
        theta=[]
        while True:
            layour = f.readline()
            if not layour:
                break
            theta.append(np.array(eval(layour)))
            pass
        return DNNModel(S_l, 0, lamda, theta=theta)


train_set = [[[1,1],[2,1],[4,1],[1,3],[2,3],[3,3],[2,2],[1,2],[4,2]],[[1,0,0,0],[0,1,0,0],[0,0,0,1],[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]]
cv_set = [[[3,4],[2,6],[1,5]],[[0,0,1,0],[0,1,0,0],[1,0,0,0]]]
test_set=[[[3,6]],[[0,0,1,0]]]
minerror=1e9
final_model=None
def activeSig(x):
    return 1.0 / (1 + np.exp(-x))
def derivativeSig(x):
    return np.multiply(x,(1-x))
def activeReLU(x):
    return np.maximum(x,0)
def derivativeReLU(x):
    return np.greater(x,np.zeros(x.shape)).astype(float)
def activeTanh(x):
    return np.tanh(x)
def derivativeTanh(x):
    t=np.tanh(x)
    return np.ones(x.shape)-np.multiply(t,t)
def activeLiner(x):
    return x
def derivativeLiner(x):
    return np.ones(x.shape)
def activeSoftmax(x):
    maxx=np.pad(np.max(x, axis=-1).reshape(-1,1), pad_width=((0,0),(0,x.shape[-1]-1)), mode='edge').reshape(x.shape)
    return np.exp(x-maxx)/np.pad(np.sum(np.exp(x-maxx), axis=-1).reshape(-1,1), pad_width=((0,0),(0,x.shape[-1]-1)), mode='edge').reshape(x.shape)
def derivativeSoftmax(x):
    raise Exception("softmax should be output layer")

if __name__ == "__main__":
    time0=time.time()
    for mod in range(13):
        temp_model = DNNModel([(2, ""),(10, "ReLU"), (10, "ReLU"), (4, "Softmax")], 0.1, lamda=mod)
        for times in range(5000):
            temp_model.train(train_set[0], train_set[1])
        errors = np.sum(np.square(temp_model.hypothesis(cv_set[0])[0][-1]-np.array(cv_set[1])))
        print(errors)
        if minerror>errors:
            final_model=copy.deepcopy(temp_model)
            minerror=errors
    print(time.time()-time0)
    final_model.save()
    newmodel=load()
    print(newmodel.hypothesis(test_set[0][0])[0][-1])



