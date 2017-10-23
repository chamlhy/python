from mxnet import nd

'''--------------------------------------'''
'''卷积层 & 池化层
	每次我们采样一个跟权重一样大小的窗口，让它跟权重做按元素的乘法然后相加。通常我们也是用卷积的术语把这个窗口叫kernel或者filter。
	stride移动窗口步伐大小
	pad在边缘填充窗口大小
	当输入数据有多个通道的时候，每个通道会有对应的权重，然后会对每个通道做卷积之后在通道之间求和
	当输入需要多通道时，每个输出通道有对应权重，然后每个通道上做卷积。
	池化层能够很好的缓解卷积层位置敏感问题。它跟卷积类似每次看一个小窗口，然后选出窗口里面最大的元素，或者平均元素作为输出
	'''

#卷积层
# 输入输出数据格式是 batch x channel x height x width，这里batch和channel都是1
# 权重格式是 output_channels x in_channels x height x width，这里input_filter和output_filter都是1。
w = nd.arange(4).reshape((1,1,2,2))
b = nd.array([1])
data = nd.arange(9).reshape((1,1,3,3))
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])

#移动窗口
out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0],
                     stride=(2,2), pad=(1,1))

#输入多通道（注意weight，data大小变了）
w = nd.arange(8).reshape((1,2,2,2))
data = nd.arange(18).reshape((1,2,3,3))

out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])


#输出多通道（注意weight，bias，data，out大小都变了）
w = nd.arange(16).reshape((2,2,2,2))
data = nd.arange(18).reshape((1,2,3,3))
b = nd.array([1,2])

out = nd.Convolution(data, w, b, kernel=w.shape[2:], num_filter=w.shape[0])

#池化层
data = nd.arange(18).reshape((1,2,3,3))

max_pool = nd.Pooling(data=data, pool_type="max", kernel=(2,2))
avg_pool = nd.Pooling(data=data, pool_type="avg", kernel=(2,2))


'''-----------------------------------------------------'''
#从零开始构建模型
#获取数据
import sys
sys.path.append('..')
from utils import load_data_from_mnist

batch_size = 256
train_data, test_data = load_data_from_mnist(batch_size)

#定义模型
import mxnet as mx
try:
	ctx = mx.gpu()
	_ = nd.zeros((1,), ctx=ctx)
except:
	ctx = mx.cpu()

#定义参数(LeNet)
weight_scale = .01


#output channels = 20, kernel = (5,5)
W1 = nd.random_normal(shape=(20,1,5,5), scale=weight_scale, ctx=ctx)
b1 = nd.zeros(W1.shape[0], ctx=ctx)

#output channels = 50, kernel = (3,3)
W2 = nd.random_normal(shape=(50,20,3,3), scale=weight_scale, ctx=ctx)
b1 = nd,zeros(W2.shape[0], ctx=ctx)

#output dim = 128
W3 = nd.random_normal(shape=(1250,128), scale=weight_scale, ctx=ctx)
b3 = nd.zeros(W3.shape[1], ctx=ctx)

#output dim = 10
W4 = nd.random_normal(shape=(W3.shape[1],10), scale=weight_scale, ctx=ctx)
b4 = nd.zeros(W4.shape[1], ctx=ctx)

#附上梯度
params = [W1, b1, W2, b2, W3, b3, W4, b4]
for param in params:
	param.attach_grad()

#定义网络(卷积模块通常是“卷积层-激活层-池化层”。然后转成2D矩阵输出给后面的全连接层。)
def net(X, verbose=False):
	X = X.as_in_context(W1.context) #将X转到和W1一样的ctx上
	#第一层卷积
	h1_cov = nd.Convolution(data=X, weight=W1, bias=b1, kernel=W1.shape[2:], num_filter=W1.shape[0])
	h1_activation = nd.relu(h1_cov)
	h1 = h1.Pooling(data=h1_activation, pool_type="max", kernel=(2,2), stride=(2,2))
	#第二层卷积
	h2_cov = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=W2.shape[2:], num_filter=W2.shape[0])
	h2_activation = nd.relu(h2_cov)
	h2 = nd.Pooling(data=h2_activation, pool_type="max", kernel=(2,2), stride=(2,2))
	#拉平，变为全连接
	h2 = nd.flatten(h2)
	#第一层全连接
	h3_linear = nd.dot(h2, W3) + b3
	h3 = nd.relu(h3_linear)
	#第四层全连接
	h4_linear = nd.dot(h3, W4) + b4
	#用于打印网络规模
	if verbose:
		print('1st conv block:', h1.shape)
        print('2nd conv block:', h2.shape)
        print('1st dense:', h3.shape)
        print('2nd dense:', h4_linear.shape)
        print('output:', h4_linear)
    return h4_linear

#进行训练
from mxnet import autograd as ag
from utils import SDG, accuracy, evaluate_accuracy
from mxnet import gluon

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

learning_rate = .2

for epoch in range(5):
	train_loss = 0.
	train_acc = 0.
	for data, label in train_data:
		label = label.as_in_context(ctx)
		with ag.record():
			output = net(data)
			loss = softmax_cross_entropy(output, label)
		loss.backward()
		SDG(params, learning_rate/batch_size)

		train_loss += nd.mean(loss).asscalar()
		train_acc += accuracy(output, label)

	test_acc = evaluate_accuracy(test_data, net, ctx)
	print("Epoch %d. Loss: %f, Train acc: %f, Test acc: %f" % (
		epoch, train_loss/len(train_data),
		train_acc/len(train_data), test_acc))


'''------------------------------------------------------------'''
#使用gluon构建模型
#定义模型
from mxnet.gluon import nn

net = nn.Sequential()
with net.name_scope():
	net.add(
		nn.Conv2D(channels=20, kernel_size=5, activation='relu'),
		nn.MaxPool2D(pool_size=2, stride=2),
		nn.Conv2D(channels=50, kernel_size=3, activation='relu'),
		nn.MaxPool2D(pool_size=2, stride=2),
		nn.Flatten(),
		nn.Dense(128, activation='relu'),
		nn.Dense(10)
	)

#获取数据并训练
from mxnet import gluon
import sys
sys.path.append('..')
import utils

#初始化
ctx = utils.try_gpu()
net.initialize(ctx=ctx)

#获取数据
batch_size = 256
train_data, test_data = utils.load_data_from_mnist(batch_size)

#训练
loss = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.5})
utils.train(train_data, test_data, net, loss, trainer, ctx, num_epochs=5)

'''-----------------------------------------------------------'''
#utils.train()内部代码
import mxnet as mx
def train(train_data, test_data, net, loss, trainer, ctx, num_epochs, print_batches=None):
	for epoch in range(num_epoch):
		train_loss = 0.
		train_acc = 0.
		#判断是否为标准数据迭代器
		if isinstance(train_data, mx.io.MXDdataIter):
			train_data.reset()
		#获得索引和值
		for i, batch in enumerate(train_data):
			data, label = _get_batch(batch, ctx)
			with ag.record():
				output = net(data)
				L = loss(output, label)
			L.backward()

			#更新权重
			trainer.step(data.shape[0])

			train_loss += nd.mean(L).asscalar()
			train_acc += accuracy(output, label)

			n = i + 1
			#打印出指定样本的训练损失和准确率
			if print_batches and n % print_batches == 0:
				print('Batch %d. Loss: %f, Train acc: %f' % (
					n, train_loss/n, train_acc/n
					))

		test_acc = evaluate_accuracy(test_data, net, ctx)
		print("Epoch %d. Loss: %f, Train acc: %f, Test acc: %f" % (
			epoch, train_loss/n, train_acc/n, test_acc
			))
