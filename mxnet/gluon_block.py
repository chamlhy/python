from mxnet import nd
from mxnet.gluon import nn

'''---------------------------------------'''
#简单例子（去除Flatten层的多层感知机）
net = nn.Sequential() 
with net.name_scope():
    net.add(nn.Dense(256,activation="relu"))
    net.add(nn.Dense(10))

print(net)
'''结果
    Sequential(
    (0): Dense(256, Activation(relu))
    (1): Dense(10, linear)
    )
    '''


'''---------------------------------------'''
#nn.Block(一个一般化的部件)定义上述例子
class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense0 = nn.Dense(256)
            self.dense1 = nn.Dense(10)
    
    def forward(self, x):
            return self.dense1(nd.relu(self.dense0(x)))

'''定义规则
    创建nn.Block的子类，重写以下函数：
    __init__ 创建参数，上面例子我们使用了包含了参数的dense层
    forward() 定义网络的计算
    '''
'''语句解释
    super(MLP, self).__init__(**kwargs): 调用nn.Block的__init__函数，它提供了prefix（指定名字）和params（指定模型参数）两个参数
    with self.name_scope(): 调用nn.Block提供的name_scope()函数。nn.Dense的定义放在这个scope里面。它的作用是给里面的所有层和参数的名字加上前缀（prefix）使得他们在系统里面独一无二。默认自动会自动生成前缀（比如dense0的前缀为mlp0_,name为mlp0_dense0），我们也可以在创建的时候手动指定（MLP(prefix='another_mlp_')）。
    nn.Dense() mxnet.gluon.nn.basic_layers.Dense
    forward() 系统会使用autograd对forward()自动生成对应的backward()函数。
    '''

'''----------------------------------------'''
#使用MLP
net2 = MLP()
print(net2)
#初始化（net的使用从此步开始）
net2.initialize()
#生成服从均匀分布的样本
x = nd.random.uniform(shape=(4,20))
y = net2(x)
y

'''-------------------------------------------'''
#简单的自定义实现和使用（nn.）Sequential
class Sequential(nn.Block):
    def __init__(self, **kwargs):
        super(Sequential, self).__init__(**kwargs)
    def add(self, block):
        self._children.append(block)
    def forward(self, x):
        for block in self._children:
            x = block(x)
        return x

net4 = Sequential()
with net4.name_scope():
    net4.add(nn.Dense(256, activation="relu"))
    net4.add(nn.Dense(10))

net4.initialize()
y = net4(x)
y

'''----------------------------------------------------'''
#灵活使用nn.Block的例子(自定义初始化权重,重复使用dense层)
class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        with self.name_scope():
            self.dense = nn.Dense(256)
            self.weight = nd.random_uniform(shape=(256,20))
    
    def forward(self, x):
        x = nd.relu(self.dense(x))
        x = nd.relu(nd.dot(x, self.weight)+1)
        x = nd.relu(self.dense(x))
        return x

fancy_mlp = FancyMLP()
fancy_mlp.initialize()
y = fancy_mlp(x)
print(y.shape)

'''---------------------------------------------------'''
#嵌套使用nn.Block和nn.Sequential
class RecMLP(nn.Block):
    def __init__(self, **kwargs):
        super(RecMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        with self.name_scope():
            self.net.add(nn.Dense(256, activation="relu"))
            self.net.add(nn.Dense(128, activation="relu"))
            self.dense = nn.Dense(64)
    
    def forward(self, x):
        return nd.relu(self.dense(self.net(x)))

rec_mlp = nn.Sequential()
rec_mlp.add(RecMLP())
rec_mlp.add(nn.Dense(10))
print(rec_mlp)
'''output
    Sequential(
        (0): RecMLP(
            (net): Sequential(
                (0): Dense(256, Activation(relu))
                (1): Dense(128, Activation(relu))
            )
            (dense): Dense(64, linear)
        )
        (1): Dense(10, linear)
    )
    '''
'''question
    如果把RecMLP改成self.denses = [nn.Dense(256), nn.Dense(128), nn.Dense(64)]，forward就用for loop来实现，会有什么问题吗？
    出现问题：print的网络结构如下：
        Sequential(
            (0): RecMLP(
  
            )
            (1): Dense(10, linear)
        )
        y = rec_mlp(x) 操作会报错误：
            RuntimeError: Parameter recmlp4_dense0_weight has not been initialized. Note that you should initialize parameters and create Trainer with Block.collect_params() instead of Block.params because the later does not include Parameters of nested child Blocks
    问题原因：根据报错可以追溯到源码中:(来自论坛root)
        看源码后发现原因为: [nn.Dense(256), nn.Dense(128), nn.Dense(64)] 的 type 是 list, 而不是 Block, 这样就不会被自动注册到 Block 类的 self._children 属性, 导致 initialize 时在 self._children 找不到神经元, 无法初始化参数.

        当执行 self.xxx = yyy 时, __setattr__ 方法会检测 yyy 是否为 Block 类型, 如果是则添加到 self._children 列表中.
        当执行 initialize() 时, 会从 self._children 中找神经元.
        详情见源码 Block 类的 __setattr__ 和 initialize 方法:    https://github.com/apache/incubator-mxnet/blob/master/python/mxnet/gluon/block.py
    
    '''