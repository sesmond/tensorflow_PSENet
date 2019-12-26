#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Title   :
@Title   : 不同格式的样本读取输出相同样本
@File    :   data_reader.py
@Author  : minjianxu
@Time    : 2019/12/26 6:16 下午
@Version : 1.0 
'''
from abc import ABCMeta, abstractmethod
import numpy as np


class BaseReader(metaclass=ABCMeta):
    """

    """

    @abstractmethod
    def load_box(self, line):
        "从一行样本中解析成box"
        pass


class Icdar2015Reader(BaseReader):
    """
     Icdar2015 样本读取器
    """

    def load_box(self, line):
        """
           icadr 2015样本 4点坐标
           :param line:
           :return:
        """
        label = line[-1]
        # x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
        # text_polys.append([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        temp_poly = []
        for idx in range(0, len(line) - 1, 2):
            temp_poly.append([float(line[idx]), float(line[idx + 1])])
        if label == '*' or label == '###' or label == '?':
            tag = True
        else:
            tag = False
        return temp_poly, tag


class Ctw1500Reader(BaseReader):
    """
        曲线文本1500读取器
    """

    def load_box(self, line):
        gt = line
        x1 = np.int(gt[0])
        y1 = np.int(gt[1])
        # TODO 前4项是矩形两点坐标
        bbox = [np.int(gt[i]) for i in range(4, 32)]
        # TODO  坐标相对位置
        bbox = np.array(bbox) + np.array(([x1 * 1.0, y1 * 1.0] * 14))
        new_box = []
        for idx in range(0, len(bbox) - 1, 2):
            new_box.append([float(bbox[idx]), float(bbox[idx + 1])])
        return np.array(new_box), False
