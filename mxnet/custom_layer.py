from mxnet import nd
from mxnet.gluon import nn

'''-----------------------------------------------------'''
#定义一个简单的层并使用
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    
    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()
layer(nd.array([1,2,3,4,5]))

#or 
net = nn.Sequential()
with net.name_scope():
    net.add(nn.Dense(128))
    net.add(nn.Dnse(10))
    net.add(CenteredLayer())

net.initialize()
y = net(nd.random.uniform(shape=(4,8)))
y.mean()

'''-------------------------------------------------------'''
#带模型参数的自定义层
from mxnet import gluon

#创建一个3*3大小的参数，取名为exciting_parameter_yay。并初始化
my_params = gluon.Parameter('exciting_parameter_yay', shape=(3,3))
my_params.initialize()

#or 使用Block自带的ParamterDict类型的成员变量params。以下获得的参数name为block1_exciting_parameter_yay。初始化
pd = gluon.ParameterDict(profix='block1_')
pd.get('exciting_parameter_yay', shape=(3,3))
pd.get('exciting_parameter_yay').initialize()
pd['block1_exciting_parameter_yay'].data()


#自定义Dense层
def MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get('weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))
    
    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)

dense = MyDense(5, in_units=10, prefix='my_dense_')
dense.params
'''output
    my_dense_ (
        Parameter my_dense_weight (shape=(10, 5), dtype=<class 'numpy.float32'>)
        Parameter my_dense_bias (shape=(5,), dtype=<class 'numpy.float32'>)
        )
    '''

#使用
dense.initialize()
dense(nd.random_uniform(shape=(2,10)))

#or
net = nn.Sequential()
with net.name_scope():
    net.add(MyDense(32, in_units=64))
    net.add(MyDense(2, in_units=32))
net.initialize()
net(nd.random_uniform(shape=(2,64)))

'''question
    修改默认初始化函数：self.weight = self.params.get('weight', shape=(in_units, units), init=nd.One())
        或者可以将init作为参数传入，实现可以改变的初始化函数
    nn.Dense如何支持延迟初始化
        in_units默认传入0，且设置params时，将allow_deferred_init置为True


