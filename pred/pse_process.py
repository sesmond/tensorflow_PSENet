"""
pse相关预处理/后处理方法
"""

from abc import ABCMeta, abstractmethod
import numpy as np
import os
import csv
from utils.utils_tool import logger
import cv2
from  pred import image_utils



class BaseProcessor(metaclass=ABCMeta):
    """

    """
    @abstractmethod
    def detect_box(self, label_value, mask_res_resized):
        """框探测"""
        pass


class RectProcessor(BaseProcessor):
    def detect_box(self,label_value, mask_res_resized):
        """
        探测最小外接矩形
        :param label_value:
        :param mask_res_resized:
        :return:
        """
        # (y,x)
        points = np.argwhere(mask_res_resized == label_value)
        points = points[:, (1, 0)]
        logger.info("DECT box:%r,%r",label_value,points)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)

        return True,box


class PolygonProcessor(BaseProcessor):
    def detect_box(self, label_value, mask_res_resized):
        """
        探测外接多边形轮廓
        :param label_value:
        :param mask_res_resized:
        :return:
        """
        # 初始化底色
        binary = np.zeros(mask_res_resized.shape, dtype='uint8')
        # 底色填值
        binary[mask_res_resized == label_value] = 1
        # 轮廓查找
        # https://www.cnblogs.com/GaloisY/p/11062065.html
        _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) <= 0:
            return False,None
        contour = contours[0]
        # TODO 寻找凸包
        bbox = cv2.convexHull(contour)
        # bbox = contour
        # 小于两个点放弃
        if bbox.shape[0] <= 2:
            return False,None
        bbox = bbox.astype('int32')
        new_box = bbox.reshape(-1, 2)  # 转换成2点坐标
        return True,new_box


class ParallelogramProcessor(BaseProcessor):
    def detect_box(self, label_value, mask_res_resized):
        processor = PolygonProcessor()
        success,box=processor.detect_box(label_value,mask_res_resized)
        if not success:
            return success,box
        area, v1, v2, v3, v4, _, _ = image_utils.mep(box)
        box = [v1, v2, v3, v4]
        box = np.array(box)
        return box


all_reader = {
    'rect': RectProcessor(),
    'poly': PolygonProcessor(),
    'para': ParallelogramProcessor()
}


def get_processor(output_type):
    """
        获取应该用数据读取器
    :return:
    """
    real_processor = all_reader.get(output_type, RectProcessor())
    return real_processor




# 如果距离小于第二个参数的点则被舍弃掉 适合正方形不适合矩形 ，这是一步步缩小点
# dp_poly  =cv2.approxPolyDP(curve=points,epsilon=10,closed=True)
# 检测四边形：
# https://stackoverflow.com/questions/37942132/opencv-detect-quadrilateral-in-python
