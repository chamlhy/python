#my cifar10 train & test code. score is 0.9516

import os
import shutil
from mxnet import autograd
from mxnet import gluon
from mxnet import image
from mxnet import init
from mxnet import nd
from mxnet.gluon import nn
from mxnet.gluon.data import vision
import numpy as np
import pandas as pd
import datetime
import sys
import utils

#unzip dataset
demo = False
#unzip little dataset
if demo:
	import zipfile
	for fin in ['train_tiny.zip', 'test_tiny.zip', 'trainLables.csv.zip']:
		with zipfile.Zipfile('../data/kaggle_cifar10/' + fin, 'r') as zin:
			zin.extractall('../data/kaggle_cifar10')

#if dataset is .7z, use '7z x filename.7z'

#reorganization dataset
def reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio):
	#read label
	with open(os.path.join(data_dir, label_file), 'r') as f:
		#skip first line
		lines = f.readlines()[1:]
		tokens = [l.rstrip()split(',') for l in lines]
		idx_label = dict(((int(idx), label) for idx, label in tokens))
	labels = set(idx_label.values())

	num_train = len(os.listdir(os.path.join(data_dir, train_dir)))
	num_train_tuning = int(num_train * (1 - valid_ratio))
	assert 0 < num_train_tuning < num_train
	num_train_tuning_per_label = num_train_tuning // len(labels)
	label_count = dict()

	def mkdir_if_not_exist(path):
		if not os.path.exists(os.path.join(*path)):
			os.makedirs(os.path.join(*path))

		#reorganization train & valid dataset
	for train_file in os.listdir(os.path.join(data_dir, train_dir)):
		idx = int(train_file.split('.')[0])
		label = idx_label[idx]
		mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
		shutil.copy(os.path.join(data_dir, train_dir, train_file), 
			os.path.join(data_dir, input_dir, 'train_valid', label))
		if label not in label_count or label_count[label] < num_train_tuning_per_label:
			mkdir_if_not_exist([data_dir, input_dir, 'train', label])
			shutil.copy(os.path.join(data_dir, train_dir, train_file), 
				os.path.join(data_dir, input_dir, 'train', label))
			label_count[label] = label_count.get(label, 0) + 1
		else:
			mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
			shutil.copy(os.path.join(data_dir, train_dir, train_file), 
				os.path.join(data_dir, input_dir, 'valid', label))

	#reorganization test dataset
	mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
	for test_file in os.listdir(os.path.join(data_dir, test_dir)):
		shutil.copy(os.path.join(data_dir, test_dir, test_file), 
			os.path.join(data_dir, input_dir, 'test', 'unknown'))

if demo:
	train_dir = 'train_tiny'
	test_dir = 'test_tiny'
	batch_size = 1
else:
	train_dir = 'train'
	test_dir = 'test'
	batch_size = 128

data_dir = '../data/kaggle_cifar10'
label_file = 'trainLables.csv'
input_dir = 'train_valid_test'
valid_ratio = 0.1
reorg_cifar10_data(data_dir, label_file, train_dir, test_dir, input_dir, valid_ratio)

#transform functions
def transform_train(data, label):
	im = data.astype('float32') / 255
	auglist = image.CreateAugmenter(data_shape(3, 32, 32), resize=0,
		rand_crop=True, rand_resize=True, rand_mirror=True,
		mean=np.array([0.4914, 0.4822, 0.4465]),
		std=np.array([0.2023, 0.1994, 0.2010]),
		brightness=0, contrast=0,
		saturation=0, hue=0,
		pca_noise=0, rand_gray=0, inter_method=2)
	for aug in auglist:
		im = aug(im)
	# change data from height*width*channel to channel*height*width
	im = nd.transpose(im, (2,0,1))
	return (im, nd.array([label]).asscalar().astype('float32'))

def transform_test(data, label):
	im = data.astype('float32') / 255
	auglist = image.CreateAugmenter(data_shape(3, 32, 32), resize=0,
		mean=np.array([0.4914, 0.4822, 0.4465]),
		std=np.array([0.2023, 0.1994, 0.2010]))
	for aug in auglist:
		im = aug(im)
	# change data from height*width*channel to channel*height*width
	im = nd.transpose(im, (2,0,1))
	return (im, nd.array([label]).asscalar().astype('float32'))

#read dataset
input_str = data_dir + '/' + input_dir + '/'

#read origin image file
train_ds = vision.ImageFolderDataset(input_str+'train', flag=1, transform=transform_train)
valid_ds = vision.ImageFolderDataset(input_str+'valid', flag=1, transform=transform_test)
train_valid_ds = vision.ImageFolderDataset(input_str+'train_valid', flag=1, transform=transform_train)
test_ds = vision.ImageFolderDataset(input_str+'test', flag=1, transform=transform_test)

loader = gluon.data.DataLoader
train_data = loader(train_ds, batch_size, shuffle=True, last_batch='keep')
valid_data = loader(valid_ds, batch_size, shuffle=True, last_batch='keep')
train_valid_data = loader(train_valid_ds, batch_size, shuffle=True, last_batch='keep')
test_data = loader(test_ds, batch_size, shuffle=True, last_batch='keep')

#loss function
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


#define model(use Resnet-164_v2 with bottleneck)
class Residual(nn.HybridBlock):
	"""docstring for Residual"""
	def __init__(self, channels, same_shape=True, **kwargs):
		super(Residual, self).__init__(**kwargs)
		self.same_shape = same_shape
		with self.name_scope():
			strides = 1 if same_shape else 2
			self.bn1 = nn.BatchNorm()
			self.conv1 = nn.Conv2D(channels=channels//4, kernel_size=1)
			self.bn2 = nn,BatchNorm()
			self.conv2 = nn.Conv2D(channels=channels//4, kernel_size=3, padding=1, strides=strides)
			self.bn3 = nn.BatchNorm()
			self.conv3 = nn.Conv2D(channels=channels, kernel_size=1)
			self.bn4 = nn.BatchNorm()
			if not same_shape:
				self.conv4 = nn.Conv2D(channels, kernel_size=1, strides=strides)

	def hybrid_forward(self, F, x):
		out = self.conv1(self.bn1(x))
		out = self.conv2(F.relu(self.bn2(out)))
		out = self.conv3(F.relu(self.bn3(out)))
		out = self.bn4(out)
		if not self.same_shape:
			x = self.conv4(x)
		return out + x

class ResNet164_v2(nn.HybridBlock):
	"""docstring for ResNet164_v2"""
	def __init__(self, num_classes, verbose=False, **kwargs):
		super(ResNet164_v2, self).__init__(**kwargs)
		self.verbose = verbose
		with self.name_scope():
			net = self.net = nn.HybridSequential()
			#block1
			net.add(nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1))
			#block2
			for _ in range(27):
				net.add(Residual(channels=64))
			#block3
			net.add(Residual(channels=128, same_shape=False))
			for _ in range(26):
				net.add(Residual(channels=128))
			#block4
			net.add(Residual(channels=256, same_shape=False))
			for _ in range(26):
				net.add(Residual(channels=256))
			#block5
			net.add(nn.BatchNorm())
			net.add(nn.Activation(activation='relu'))
			net.add(nn.AvgPool2D(pool_size=8))
			net.add(nn.Flatten())
			net.add(nn.Dense(num_classes))

	def hybrid_forward(self, F, x):
		out = x
		for i,b in enumerate(self.net):
			out = b(out)
			if self.verbose:
				print('Block %d output: %s' %(i+1, out.shape))
		return out

def get_net(ctx):
	num_outputs = 10
	net = ResNet164_v2(num_outputs)
	net.initialize(ctx=ctx, init=init.Xavier())
	return net

#train function
sys.path.append('..')

def train(net, train_data, valid_data, num_epochs, lr, wd, ctx, lr_period, lr_decay):
	trainer = gluon.Trainer(net.collect_params(), 'sgd',
	 {'learning_rate':lr, 'momentum':0.9, 'wd':wd})

	prev_time = datetime.datetime.now()
	for epoch in range(num_epochs):
		train_loss = 0.0
		train_acc = 0.0
		if epoch > 0 and epoch == lr_period:
			trainer.set_learning_rate(trainer.learning_rate * lr_decay)
			lr_period -= 10
		for data, label in train_data:
			label = lable.as_in_context(ctx)
			with autograd.record():
				output = net(data.as_in_context(ctx))
				loss = softmax_cross_entropy(output, label)
			loss.backward()
			trainer.step(batch_size)
			train_loss += nd.mean(loss).asscalar()
			train_acc += utils.accuracy(output, label)
		cur_time = datetime.datetime.now()
		h, remainder = divmod((cur_time - prev_time).seconds, 3600)
		m, s = divmod(remainder, 60)
		time_str = "Time %02d:%02d:%02d" % (h, m, s)

		if valid_data is not None:
			valid_acc = utils.evaluate_accuracy(valid_data, net, ctx)
			epoch_str = ("Epoch %d. Loss: %f, Train acc %f, Valid acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data), valid_acc))
		else:
			epoch_str = ("Epoch %d. Loss: %f, Train acc %f, "
                         % (epoch, train_loss / len(train_data),
                            train_acc / len(train_data)))
		prev_time = cur_time
		print(epoch_str + time_str + ', lr' + str(trainer.learning_rate))


# train model
ctx = utils.try_gpu()
num_epochs = 250
learning_rate = 0.1
weight_decay = 5e-4
lr_period = 100
lr_decay = 0.1

net = get_net(ctx)
net.hybridize()
#train with valid
train(net, train_data, valid_data, num_epochs, learning_rate, weight_decay, ctx, lr_period, lr_decay)
#train all train_data
train(net, train_valid_data, None, num_epochs, learning_rate, weight_decay, ctx, lr_period, lr_decay)

#get submission.csv
preds = []
for data, label in test_data:
	output = net(data.as_in_context(ctx))
	preds.extend(output.argmax(axis=1).astype(int).asnumpy())

sorted_ids = list(range(1, len(test_ds) + 1))
sorted_ids.sort(key = lambda x : str(x))

df = pd.DataFrame({'id':sorted_ids, 'lable':preds})
df['lable'] = df['lable'].apply(lambda x : train_valid_ds.synsets[x])
df.to_csv('submission.csv', index=False)
		
