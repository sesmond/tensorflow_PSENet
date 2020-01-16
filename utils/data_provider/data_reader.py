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
import os
import csv


class BaseReader(metaclass=ABCMeta):
    """
     坐标读取基类
    """
    @abstractmethod
    def load_box(self, line):
        """从一行样本中解析成box"""
        pass

    @abstractmethod
    def get_text_file_name(self, image_name):
        """根据图片名找到对应的文件名"""
        pass

    def _load_annotation(self,txt_file):
        """
        load annotation from the text file
        # 从标注文件中读取 坐标
        :param txt_file:
        :return:
        """
        text_polys = []
        text_tags = []
        if not os.path.exists(txt_file):
            return np.array(text_polys, dtype=np.float32)

        with open(txt_file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
                # TODO 解析样本行
                temp_poly, temp_tag = self.load_box(line)
                text_polys.append(temp_poly)
                text_tags.append(temp_tag)
            return np.array(text_polys, dtype=np.float32), np.array(text_tags, dtype=np.bool)

    def get_annotation(self,im_fn,txt_path):
        # 文本路径+文本名
        txt_name = self.get_text_file_name(im_fn)
        txt_fn = os.path.join(txt_path, txt_name)
        success = True

        if not os.path.exists(txt_fn):
            # logger.error("文件：%r ,不存在", txt_fn)
            success = False
        # text_tags 是否是文本 True False
        text_polys, text_tags = self._load_annotation(txt_fn)
        return success,text_polys, text_tags

class Icdar2015Reader(BaseReader):
    """
     Icdar2015 样本读取器
    """

    def get_text_file_name(self, image_name):
        # 替换后缀名
        image_name = 'gt_' + os.path.basename(image_name).split('.')[0]+'.txt'
        return image_name

    def load_box(self, line):
        """
           icadr 2015样本 4点坐标
           :param line:
           :return:
        """
        label = line[-1]
        x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, line[:8]))
        temp_poly = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

        if label == '*' or label == '###' or label == '?':
            tag = True
        else:
            tag = False
        return temp_poly, tag


class Ctw1500Reader(BaseReader):
    """
        曲线文本1500读取器
    """

    def get_text_file_name(self, image_name):
        txt_fn = os.path.basename(image_name).split('.')[0] + '.txt'
        return txt_fn

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


class PlateReader(BaseReader):
    """
    车牌样本读取 TODO
    """
    def __parse(self, f_name):
        data = f_name.split("-")

        four_points = [d.split("&") for d in data[3].split("_")]  # 499&580_409&557_418&525_508&548
        points = []
        for p in four_points:
            points.append([int(fp) for fp in p])

        return points

    def get_annotation(self,im_fn,txt_path):
        """
               覆盖父类方法
        """
        try:
            points = self.__parse(os.path.basename(im_fn))
        except Exception:
            # print("文件名：", im_fn)
            return False,None,None
        # 文本路径+文本名
        success = True
        return success,np.array([points], dtype=np.float32), np.array([False], dtype=np.bool)

    def get_text_file_name(self, image_name):
        return None

    def load_box(self, line):
        return None



def test1():
    '''
    获取后缀名为exts的所有文件
    TODO 路径training_data_path 没做参数也没校验
    :param exts:
    :return:
    '''
    import glob
    exts=['jpg']
    files = []
    # TODO 可以有多个路径
    for ext in exts:
        # glob.glob 得到所有文件名
        # 一层 2层子目录都取出来
        files.extend(glob.glob(os.path.join("./data/plate", '*.{}'.format(ext))))
        files.extend(glob.glob(os.path.join("./data/plate", '*', '*.{}'.format(ext))))
    reader = PlateReader()
    for f in files:
        reader.get_annotation(f,"")


if __name__ == '__main__':
    # f_n = "/image/1846-2_2-0&440_617&690-616&660_0&690_0&470_617&440-0_0_21_25_6_24_24-74-79.jpg"
    # reader = PlateReader()
    #
    # _,resuylt,_=reader.get_annotation(f_n,"")
    # print(resuylt)
    test1()