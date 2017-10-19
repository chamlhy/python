from mxnet import nd
from mxnet.gluon import nn

'''---------------------------------------'''
#简单例子（去除Flatten层的多层感知机）
def get_net():
    net = nn.Sequential() 
    with net.name_scope():
        net.add(nn.Dense(256,activation="relu"))
        net.add(nn.Dense(10))
    return net

x = nd.random_uniform(shape=(3,5))

#错误尝试
import sys
try:
    net = get_net()
    net(x)
except RuntimeError as err:
    sys.stderr.write(str(err))

'''如果不initialize,直接使用，会报错:
    Parameter sequential0_dense0_weight has not been initialized. Note that you should initialize parameters and create Trainer with Block.collect_params() instead of Block.params because the later does not include Parameters of nested child Blocks
    '''
    
#正确方法
print(net)
net.initialize()
net(x)
    
'''output
    Sequential(
        (0): Dense(4, Activation(relu))
        (1): Dense(2, linear)
    )
    '''


'''----------------------------------------------'''
#访问模型参数
#weight & bias 获得Parameter类
w = net[0].weight
b = net[0].bias
print('name: ', net[0].name, '\nweight: ', w, '\nbias: ', b)
'''output
    name:  sequential0_dense0 
    weight:  Parameter sequential0_dense0_weight (shape=(4, 5), dtype=<class 'numpy.float32'>) 
    bias:  Parameter sequential0_dense0_bias (shape=(4,), dtype=<class 'numpy.float32'>)
    '''

# data & grad 获取类中的参数和梯度
w.data()
w.grad()
b.data()
b.grad()

#collect_params访问Block所有的参数，返回一个名字到对应Parameter的dict
params = net.collect_params()
params['sequential0_dense0_weight'].data()
params['sequential0_dense0_bias'].data()
params.get(dense1.weight).data()
params.get(dense1.bias).data()

'''---------------------------------------------------'''
#定制初始化方法
from mxnet import init
#（weight正态分布）
params.initialize(init=init.Normal(sigma=0.02), force_reinit=True)
#or（会将weight初始化为1）
params.initialize(init=init.One(), force_reinit=True)

#自定义初始化
class MyInit(init.Initializer):
    def __init__(self):
        super(MyInit,self).__init__()
        self._verbose = True
    def _init_weight(self, _, arr):
        #初始化权重，使用out=arr后不需指定形状
        nd.random_uniform(low=5, high=10, out=arr)
    def _init_bais(self, _, arr):
        arr[:] = 2

#未初始化bias是gluon的bug
params.initialize(init=MyInit(), force_reinit=True)

'''------------------------------------------------'''
'''延后初始化问题
    初始化时因为不知道输入的形状，所以未初始化weight和bias
    在输入数据后，才开始初始化工作
    '''

#避免延后初始化（指定输入大小）
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, in_units=5, activation="relu"))
    net.add(nn.Dense(2, in_units=4))
    
net.initialize(MyInit())

'''------------------------------------------------------'''
#共享模型参数
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(4, in_units=4, activation="relu"))
    net.add(nn.Dense(4, in_units=4, activation="relu", params=net[-1].params)
    net.add(nn.Dense(2), in_units=4)

net.initialize(MyInit())
'''output
    初始化一次
    且 net[0].weight.data() = net[1].weight.data()
    '''

'''------------------------------------------------------'''
'''question
    net.collect_params() 返回一个mxnet.gluon.parameter.ParameterDict，包含每层网络的weight和bias
    net.params 也返回一个ParameterDict，但是是空的。用net[0].params可以返回一个包含此层weight和bias的ParameterDict
    每层使用不同的初始化参数： 在定义一个层时，传入weight_initializer参数，指定初始化函数。比如：net.add(nn.Dense(4, activation=“relu”, weight_initializer = init.One()))
    两个层共用一个参数，那么求梯度的时候会发生什么？ 不知道，可能会求导两次，然后更新两次？