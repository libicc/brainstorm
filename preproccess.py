import os
import numpy as np
import scipy.io as sio
import pandas as pd

path1 = 'C:/Users/llbcc/Downloads/正常数据/正常数据/'
file_list1 = os.listdir(path1)
print(file_list1)

for file in file_list1:
    data = sio.loadmat(path1 + file)
    data.pop('__globals__'); data.pop('__header__');  data.pop('__version__')

    # data_to_save = np.ndarray([68,68,7], dtype=np.complex128)
    # for i, fi in enumerate(data):
    #     data_to_save[:,:,i] = data[fi]
    # #np.save(file +'.npy', data_to_save)

    normal_data = np.ndarray([57,2278], dtype=np.complex128)
    for i, fi in enumerate(data):
        temp = np.ndarray(2278,dtype=np.complex128)
        for j in range(68):            
            temp.append(data[j,j:])  #这里的语法肯定是有问题的，先把算法整理好再调整:ndarray中提取上三角矩阵
        normal_data[i] = temp
        # normal_data[:,-1] = np.ones([57,1])
    np.save('normal_data.npy',data_to_save)

#############################################
# 方法同上，将异常数据读入内存,记为abnormal_data
# ###########################################
# 

x = np.ndarray([114,2278], dtype=np.complex128)
x[:57, :] = normal_data
x[57:, :] = abnormal_data

y = np.ndarray([114,1])
y[:57] = np.ones([57,1])
y[57:] = np.ones([57,1]) - 2

X = pd.DataFrame(x)
Y = pd.DataFrame(y)
X.label = Y

shuffle

#提取验证集数据 0.4

############################################
# 模型搭建
##############################################



