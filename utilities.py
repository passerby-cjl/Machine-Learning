import numpy as np
def get_feature(X,k):
    m_samples , n_features = X.shape
    #中心化  去均值  均值为0
    mean = np.mean(X,axis=0)
    normX = X - mean  #去均值，中心化
    cov_mat = np.dot(normX.T,normX)/n_features #协方差矩阵
    vals , vecs = np.linalg.eig(cov_mat) #得到特征向量和特征值
    print('特征值',vals)
    print('特征向量',vecs)

    eig_pairs = [[np.abs(vals[i]),*vecs[:,i]] for i in range(n_features)]
    print('-------------')
    #将特征值由大到小排列
    eig_pairs.sort(key=lambda x:x[0], reverse=True)
    eig_pairs=np.array(eig_pairs)
    print(eig_pairs)
    feature = np.array(eig_pairs[0:k,1:])
    #将数据进行还原操作 normX 中心化后的数据 和 特征向量相乘
    return mean,feature

def pca(X, ndim, readfile=''):#
    m_samples , n_features = X.shape
    X=normalize(X, recordfile=readfile)
    #中心化  去均值  均值为0
    #mean = np.mean(X,axis=0)
    #normX = X - mean  #去均值，中心化
    cov_mat = np.dot(X.T,X)/n_features #协方差矩阵
    vals , vecs = np.linalg.eig(cov_mat) #得到特征向量和特征值
    print('特征值',vals)
    print('特征向量',vecs)

    eig_pairs = [[np.abs(vals[i]),*vecs[:,i]] for i in range(n_features)]
    print('-------------')
    #将特征值由大到小排列
    eig_pairs.sort(key=lambda x:x[0], reverse=True)
    eig_pairs=np.array(eig_pairs)
    print(eig_pairs)
    feature = np.array(eig_pairs[0:ndim, 1:])
    if readfile:
        with open(readfile, 'a') as f:
            f.write("FEATURE:%s\n"%str(feature.tolist()))
    data = np.dot(X, feature.T)
    return data

def normalize(X, mode='Zscore', recordfile=''):
    data=X
    if mode=='Zscore':
        mean=np.mean(X,axis=0)
        std=np.std(X,axis=0)
        if recordfile:
            with open(recordfile, 'w') as f:
                f.write("MEAN:%s\n"%str(mean.tolist()))
                f.write("STD:%s\n"%str(std.tolist()))
        mean=np.pad(mean.reshape(1,-1),pad_width=((0,X.shape[0]-1),(0,0)),mode='edge')
        std=np.pad(std.reshape(1,-1),pad_width=((0,X.shape[0]-1),(0,0)),mode='edge')
        data=(X-mean)/std
        data=np.nan_to_num(data)
    elif mode=='Minmax':
        maxX=np.max(X, axis=0)
        minX=np.min(X, axis=0)
        if recordfile:
            with open(recordfile, 'w') as f:
                f.write("MIN:%s\n"%str(minX.tolist()))
                f.write("RANGE:%s\n"%str((maxX-minX).tolist()))
        maxX=np.pad(maxX.reshape(1,-1),pad_width=((0,X.shape[0]-1),(0,0)),mode='edge')
        minX=np.pad(minX.reshape(1,-1),pad_width=((0,X.shape[0]-1),(0,0)),mode='edge')
        data=(X-minX)/(maxX-minX)
        data=np.nan_to_num(data)
    return data

#print(pca(np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]),2))
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