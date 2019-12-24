import subprocess
import os
import numpy as np
import cv2
from . import pse_py
BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if subprocess.call(['make', '-C', BASE_DIR]) != 0:  # return value
    raise RuntimeError('Cannot compile pse: {}'.format(BASE_DIR))

def pse(kernals, min_area=5):
    '''
    :param kernals: 二分后的图像
    :param min_area: 保留的最小面积
    :return:
    '''
    #TODO 算了一下这里的kernal是从大到小的，顺序应该反了
    return pse_py.pse(kernals,min_area)
    # TODO 找不到指定模块？
    from .pse import pse_cpp
    kernal_num = len(kernals)
    if not kernal_num:
        return np.array([]), []
    kernals = np.array(kernals)
    #https://blog.csdn.net/Vichael_Chan/article/details/100988503
    # 联通子图
    label_num, label = cv2.connectedComponents(kernals[kernal_num - 1].astype(np.uint8), connectivity=4)
    label_values = []
    for label_idx in range(1, label_num):
        if np.sum(label == label_idx) < min_area:
            label[label == label_idx] = 0
            continue
        label_values.append(label_idx)
    print("label:",label.shape)
    pred = pse_cpp(label, kernals, c=6)
    return pred, label_values


