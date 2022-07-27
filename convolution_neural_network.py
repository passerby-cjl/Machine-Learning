import struct

import numpy as np
import deep_neural_network as DNN

class CNNModel(DNN.DNNModel):
    lamdas = [0,0.01,0.02,0.04,0.08,0.16,0.32,0.64,1,2,4,8,10]
    epsolum = 0.1
    type='CNN'
    def __init__(self, S_l: list, learning_rate, lamda=0, theta=None):
        #each layour in S_l consists of(channel,k_size,stride,padding,activate)
        #except the layour0 consists of(channel,dim,padding,activate) representing for input layour
        super().__init__(S_l, learning_rate, lamda, theta)

    def _pad(self, x, leftup, rightdown):
        return np.pad(x,((0, 0), *eval('(leftup, rightdown),'*(x.ndim-2)), (0, 0)), constant_values=0)

    def _im2col(self, ims, k_size, stride=1):
        #N, H, W, C = ims.shape
        dim=ims.ndim-2
        # set shape as (N,*(outputshape),*(k_size,k_size,...),C)
        shape=np.array(list(ims.shape[:-1])+[k_size]*dim+[ims.shape[-1]])
        shape[1:dim+1]=(shape[1:dim+1]-k_size)//stride+1
        #oh = (H - k_size) // stride + 1
        #ow = (W - k_size) // stride + 1
        #print(N,oh,ow,k_size,k_size,C)
        strides=[ims.strides[0]]
        for axis in range(dim):
            strides.append(ims.strides[axis+1]*stride)
        strides += list(ims.strides[1:])
        #strides = (*ims.strides[0], ims.strides[-3]*stride, ims.strides[-2]*stride, *ims.strides[1:])
        #col = np.lib.stride_tricks.as_strided(ims, shape=(N,oh,ow,k_size,k_size,C), strides=strides)
        col = np.lib.stride_tricks.as_strided(ims, shape=shape, strides=strides)
        return col.reshape(*col.shape[:dim+1], -1)

    def _preproceed_ims(self, ims, layourinfo):
        if layourinfo[-2]:
            ims = self._pad(ims, (layourinfo[1]-1)//2, layourinfo[1]//2)
        col = self._im2col(ims, layourinfo[1], layourinfo[2])
        col = np.append(np.ones((*col.shape[0:-1], 1)), col, axis=-1)  # 增加偏置
        return col

    def _preproceed_delta(self, delta, layourinfo, inputshape):
        shape=(np.array(delta.shape[1:-1]))*layourinfo[2]
        tmp=np.zeros((delta.shape[0],*tuple(shape), delta.shape[-1]))
        cmd='tmp[:'+',::layourinfo[2]'*(tmp.ndim-2)+',:]=delta'
        exec(cmd)
        if layourinfo[-2]:
            pad_width=[(0,0),(0,0)]
            for dim in range(1, tmp.ndim-1):
                pad_width.insert(-1, (layourinfo[1]//2, inputshape[dim]+layourinfo[1]-1-layourinfo[1]//2-tmp.shape[dim]))
            delta=np.pad(tmp, pad_width, constant_values=0)
            #delta=self._pad(delta, layourinfo[1]//2, (layourinfo[1]-1)//2)
        else:
            pad_width=[(0,0),(0,0)]
            for dim in range(1, tmp.ndim-1):
                pad_width.insert(-1, (layourinfo[1]-1, inputshape[dim]-tmp.shape[dim]))
            delta=np.pad(tmp, pad_width, constant_values=0)
            #delta=self._pad(delta, layourinfo[1]-1, layourinfo[1]-1)
        #tmp[:,::layourinfo[2],::layourinfo[2],:]=delta
        # if layourinfo[-2]:
        #     delta=self._pad(tmp, layourinfo[1]//2, (layourinfo[1]-1)//2)
        # else:
        #     delta=self._pad(tmp, layourinfo[1]-1, layourinfo[1]-1)
        col_delta=self._im2col(delta, layourinfo[1], 1)
        return col_delta

    def _preproceed_theta(self, theta, layourinfo):
        return np.flipud(theta[1:].reshape(layourinfo[0], -1, order='F').T)

    def save(self, filename='mymodel/CNNMODEL.txt'):
        with open(filename, "w") as f:
            f.write("TYPE:%s\n"%self.type)
            f.write("STRUCTURE:%s\n"%str(self.S_l))
            f.write("LAMDA:%s\n"%str(-self.lamda))
            for l in range(self.L):
                f.write("%s\n"%str(self.theta[l].tolist()))

def load(filename='mymodel/CNNMODEL.txt'):
    with open(filename, "r") as f:
        if f.readline()!="TYPE:CNN\n":
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
        return CNNModel(S_l, 0, lamda=lamda, theta=theta)

def getMNIST():
    with open("MNIST/t10k-images.idx3-ubyte","rb") as cvim, open("MNIST/t10k-labels.idx1-ubyte","rb") as cvla,\
        open("MNIST/train-images.idx3-ubyte","rb") as imgf, open("MNIST/train-labels.idx1-ubyte","rb") as labf:
        imgs1=[]
        labs1=[]
        imbuf=imgf.read()
        labuf=labf.read()
        imindex=struct.calcsize('>IIII')
        laindex=struct.calcsize('>II')
        lab = struct.unpack_from('>60000B', labuf, laindex)
        for i in lab:
            temp=np.zeros((1,1,10))
            temp[0][0][i]=1
            labs1.append(temp)
        for i in range(60000):
            img = struct.unpack_from('>784B', imbuf, imindex)
            imindex += struct.calcsize('>784B')
            img = np.reshape(img, (28,28,1))
            imgs1.append(img)
        imgs2=[]
        labs2=[]
        labuf=cvla.read()
        imbuf=cvim.read()
        imindex=struct.calcsize('>IIII')
        laindex=struct.calcsize('>II')
        cvlab=struct.unpack_from('>10000B', labuf, laindex)
        for i in cvlab:
            temp=np.zeros((1,1,10))
            temp[0][0][i]=1
            labs2.append(temp)
        for i in range(10000):
            img=struct.unpack_from('>784B', imbuf, imindex)
            imindex+=struct.calcsize('>784B')
            img=np.reshape(img,(28,28,1))
            imgs2.append(img)
    return imgs1, labs1, imgs2, labs2

if __name__ == "__main__":
    # easy examples for training
    ## of two dimensions
    X = np.array([[[[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
                   [[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7], [8, 8, 8]],
                   [[7, 7, 7], [8, 8, 8], [9, 9, 9], [10, 10, 10], [11, 11, 11]],
                   [[10, 10, 10], [11, 11, 11], [12, 12, 12], [13, 13, 13], [14, 14, 14]],
                   [[13, 13, 13], [14, 14, 14], [15, 15, 15], [16, 16, 16], [17, 17, 17]]],
                  [[[12, 12, 12], [11, 11, 11], [10, 10, 10], [9, 9, 9], [8, 8, 8]],
                   [[9, 9, 9], [8, 8, 8], [7, 7, 7], [6, 6, 6], [5, 5, 5]],
                   [[6, 6, 6], [5, 5, 5], [4, 4, 4], [3, 3, 3], [2, 2, 2]],
                   [[3, 3, 3], [2, 2, 2], [1, 1, 1], [0, 0, 0], [-1, -1, -1]],
                   [[0, 0, 0], [-1, -1, -1], [-2, -2, -2], [-3, -3, -3], [-4, -4, -4]]]])
    Y = np.array([[[[1]]], [[[0]]]])
    ## of one dimension
    D1 = np.array([[[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]],[[5,5,5],[4,4,4],[3,3,3],[2,2,2],[1,1,1]]])
    D2 = np.array([[[1]],[[0]]])

    cnn=CNNModel([(3,2,True), (2,3,2,True,'ReLU'),(1,3,2,False,'ReLU')],0.01)
    for episode in range(100):
        cnn.train(X,Y)
    print(cnn.hypothesis(X)[0][-1])
    ################################ train with MNIST
    #trainimgs, trainlabs, cvimgs, cvlabs=getMNIST()
    #ocrcnn=CNNModel([(1, True, ''), (3, 3, 2, True, 'ReLU'), (5, 3, 2, True, 'ReLU'), (7, 3, 2, True, 'ReLU'), (10, 4, 1, False, 'Sig')], 0.001)
    # ocrcnn=load()
    # ocrcnn.learning_rate=0.0001
    # try:
    #     for i in range(5000):
    #         for j in range(0,60000, 100):
    #             ocrcnn.train(trainimgs[j:j+100], trainlabs[j:j+100])
    #         errors = np.sum(np.square(ocrcnn.hypothesis(cvimgs)[0][-1] - np.array(cvlabs)))
    #         print(errors)
    # finally:
    #     ocrcnn.save()
    ########################
    #print(ocrcnn.hypothesis([cvimgs[2]])[0][-1], cvlabs[2])