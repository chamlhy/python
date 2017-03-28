#-*- coding:utf-8 -*-

import math
import numpy as np
#from matplotlib import pyplot
from collections import Counter
import warnings
import pandas as pd
import random

#KNN算法
def k_nearest_neighbors(data, predict, k=5):
	
	if len(data) >= k:
		warnings.warn("k is too small")
		
	#predict到各点距离
	distances = []
	for group in data:
		for features in data[group]:
			#欧拉距离
			euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
			distances.append([euclidean_distance, group])
	
	sorted_distances = [i[1] for i in sorted(distances)]
	top_nearest = sorted_distances[:k]
	
	group_res = Counter(top_nearest).most_common(1)[0][0]
	confidence = Counter(top_nearest).most_common(1)[0][1]*1.0/k
	
	return group_res, confidence
	
if __name__=='__main__':
 
	#local test
    #dataset = {'black':[ [1,2], [2,3], [3,1] ], 'red':[ [6,5], [7,7], [8,6] ]}
    #new_features = [6,7]  # 判断这个样本属于哪个组
 
    #for i in dataset:
        #for ii in dataset[i]:
            #pyplot.scatter(ii[0], ii[1], s=50, color=i)
 
    #which_group,confidence = k_nearest_neighbors(dataset, new_features, k=3)
    #print(which_group, confidence)
 
    #pyplot.scatter(new_features[0], new_features[1], s=100, color=which_group)
 
    #pyplot.show()
	
	#teat use breast-cancer-wisconsin.data
	df = pd.read_csv('breast-cancer-wisconsin.data')
	df.replace('?', np.nan, inplace=True)
	df.dropna(inplace=True)
	df.drop(['id'], 1, inplace=True)
	
	# 把数据分成两部分，训练数据和测试数据
	full_data = df.astype(float).values.tolist()
 
	#随机排序
	random.shuffle(full_data)
 
	test_size= 0.2   # 测试数据占20%
	train_data = full_data[:-int(test_size*len(full_data))]
	test_data = full_data[-int(test_size*len(full_data)):]
 
	#数据分类/打标签
	train_set = {2:[], 4:[]}
	test_set = {2:[], 4:[]}
	for i in train_data:
		train_set[i[-1]].append(i[:-1])
	for i in test_data:
		test_set[i[-1]].append(i[:-1])
 
	correct = 0
	total = 0
 
	for group in test_set:
		for data in test_set[group]:
			res,confidence = k_nearest_neighbors(train_set, data, k = 5) # 你可以调整这个k看看准确率的变化，你也可以使用matplotlib画出k对应的准确率，找到最好的k值
			if group == res:
				correct += 1
			else:
				print(confidence)
			total += 1
 
	correct *= 1.0
	print(correct/total)  # 准确率
 
	print(k_nearest_neighbors(train_set, [4,2,1,1,1,2,3,2,1], k = 5)) # 预测一条记录
	
	
	
	

