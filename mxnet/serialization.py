from mxnet import nd

'''---------------------------------------------'''
# save & load(NDArray, NDArray list, dict)
x = nd.ones(3)
y = nd.zeros(4)
filename = '../data/test1.params'
nd.save(filename, [x, y])
a, b = nd.load(filename)
print(a, b)

'''----------------------------------------------'''
#Gluon模型 save_params & load_params
filename = '../data/mlp.params'
net.save_params(filename)

import mxnet as mx
net2.load_params(filename, mx.cpu())