import os
import numpy as np
import scipy.io as sio
import pandas as pd

####################################################

def matrix2vector(np_m):
    vec = np.triu(np_m, 1).flatten()
    vec_data = pd.DataFrame(vec)

    to_drop = []
    for i in np.arange(68):
	    for j in np.arange(0,i+1):
		    to_drop.append(i*68+j)
    vec_data.drop(index=to_drop, inplace=True)
    vec = np.array(vec_data).reshape([1,2278])
    
    return vec

####################################################
#将正常数据读入内存,记为normal_data
path1 = 'C:/Users/llbcc/Downloads/正常数据/正常数据/'
file_list1 = os.listdir(path1)
print('***********************正常数据文件列表************************')
print(file_list1)
print(len(file_list1))

normal_data = np.empty(shape=[57, 15946], dtype=np.complex128)
for count, file in enumerate(file_list1):
    data = sio.loadmat(path1 + file)
    data.pop('__globals__'); data.pop('__header__');  data.pop('__version__')
    sorted(data.keys())

    for i, fi in enumerate(data):
       normal_data[count, i*2278:i*2278+2278] = matrix2vector(data[fi])

print(normal_data[:,15945])
np.save('normal_data.npy', normal_data)

#############################################
# 方法同上，将异常数据读入内存,记为abnormal_data
path2 = 'C:/Users/llbcc/Downloads/棘波数据/棘波数据/'
file_list2 = os.listdir(path2)
print('***********************棘波数据文件列表************************')
print(file_list1)
print(len(file_list2))

abnormal_data = np.empty(shape=[60, 15946], dtype=np.complex128)
for count, file in enumerate(file_list2):
    data = sio.loadmat(path2 + file)
    data.pop('__globals__'); data.pop('__header__');  data.pop('__version__')
    sorted(data.keys())

    for i, fi in enumerate(data):
        abnormal_data[count, i*2278:i*2278+2278] = matrix2vector(data[fi])

np.save('abnormal_data.npy', abnormal_data)

# ############################################

x = np.ndarray([117, 15946], dtype=np.complex128)
x[:57, :] = normal_data
x[57:, :] = abnormal_data

x_real  = np.real(x)
x_imag  = np.imag(x)
x_angle = np.angle(x)
x_abs   = np.abs(x)

y = np.ndarray([117,1])
y[:57] = np.ones([57,1])
y[57:] = np.ones([60,1]) - 2


# ######################使用sigmoid函数进行分类######################

import sklearn
from sklearn.cross_validation import train_test_split
input_data = x_abs
x_train, x_test, y_train, y_test = train_test_split(input_data, y, test_size=0.3, random_state=1)

# 对训练数据进行标准化
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(x_train_std, y_train)

y_pred = lr.predict(x_test_std)
y_pred = y_pred.reshape([36,1])
acc = np.count_nonzero(y_pred - y_test)

print(1- acc/36)