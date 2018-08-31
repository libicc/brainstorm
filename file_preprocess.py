import os
import numpy as np
import scipy.io as sio


path1 = 'C:/Users/llbcc/Downloads/正常数据/正常数据/'
file_list1 = os.listdir(path1)
print('***********************正常数据文件列表************************')
print(file_list1)
print('***********************正常数据文件列表************************')
print(len(file_list1))

for file in file_list1:
    data = sio.loadmat(path1 + file)
    data.pop('__globals__'); data.pop('__header__');  data.pop('__version__')

    data_to_save = np.ndarray([68,68,7], dtype=np.complex128)
    for i, fi in enumerate(data):
        data_to_save[:,:,i] = data[fi]
    #np.save(file +'.npy', data_to_save)
    


