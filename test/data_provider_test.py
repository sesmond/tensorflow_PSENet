#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Title   :TODO
@File    :   data_provider_test.py    
@Author  : minjianxu
@Time    : 2019/12/19 2:39 下午
@Version : 1.0 
'''
import cv2
import numpy as np
import  image_test


#1. 测试图片多点坐标划入是什么样得？
def test1():
    print("")
    img = cv2.imread("0002.jpg")
    # seg_map = cv2.fillPoly(seg_map, [np.array(shrinked_poly).astype(np.int32)], 1)
    shrinked_poly=[ 0,0,113,0,227,0,340,0,454,0,567,0,681,0,681,144,567,144,454,144,340,144,227,144,113,144,0,144
                  ]

    shrinked_poly =[6, 0, 153, 29, 300, 57, 440, 112, 575, 185, 725, 205, 871, 240, 822, 368, 684, 335, 545, 307, 414, 247, 283, 190, 144, 161, 0, 154]
    # 1333, 813, 2014, 957
    #TODO 哪个点在最前面 还有顺时针还是逆时针！！
    # shrinked_poly = [50,50,300,50,300,300,50,400]
    #TODO 一维变二维 TODO 顺时针旋转
    shrinked_poly = image_test.poly_convert(shrinked_poly)
    image_test.show(img,"划线前")
    shrinked_poly =  np.asarray(shrinked_poly)
    #TODO 是不是单通道才行？
    img = cv2.fillPoly(img, shrinked_poly.astype(np.int32)[np.newaxis, :, :], 0)
    # cv2.polylines(img,shrinked_poly,False)
    cv2.polylines(img, shrinked_poly, False, color=(0, 255, 0), thickness=5)

    # seg_map = cv2.drawContours(img, [np.array(shrinked_poly).astype(np.int32)], -1, 1, -1)
    image_test.show(img,"划线后")



if __name__ == '__main__':
    test1()