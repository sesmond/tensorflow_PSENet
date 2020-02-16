# -*- coding:utf-8 -*-
import cv2
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from utils.utils_tool import logger, cfg
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形
import random

# --test_data_path =./ data / pred / input / \
#                      --checkpoint_path =./ model / \
#                                            --output_dir =./ data / pred / output
# TODO 验证集数据
import pred

tf.app.flags.DEFINE_string('validate_img_path', './data/plate_new/test/test_img', '')
tf.app.flags.DEFINE_string('validate_txt_path', './data/plate_new/test/test_txt', '')

FLAGS = tf.app.flags.FLAGS

logger.setLevel(cfg.debug)

"""
    验证数据，通过计算IOU进行判断
"""


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.validate_img_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    logger.info('Find {} images'.format(len(files)))
    return files

def validate(params):
    # 批量验证数据 计算F1 recall等
    im_fn_list = get_images()
    #TODO 筛选100个
    im_fn_list = random.sample(im_fn_list,100)
    logger.info("随机抽取样本数进行验证：%r",len(im_fn_list))
    # 计算IOU 大于0.7的就算预测正确
    cnt_true = 0 # 正确条数 TODO 召回率
    for im_fn in im_fn_list:
        logger.debug('image file:{}'.format(im_fn))
        im = cv2.imread(im_fn)

        base_name = os.path.splitext(os.path.basename(im_fn))[0]
        txt_p = os.path.join(FLAGS.validate_txt_path, base_name+ ".txt")
        line = open(txt_p).readlines()[0]
        # line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
        line = list( map(float,line.split(",")[:-1]))
        gt_box = np.array(line).reshape(4,2)
        # gt_box = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

        boxes = pred.pred(params, im, im_fn)
        for box in boxes:
            logger.debug('gt_box:%s,box:%s',str(gt_box),str(box))
            iou = cal_iou(gt_box, box)
            logger.info("图片 ：%s,iou:%r", os.path.basename(im_fn), iou)
            if iou > 0.7:
                logger.info("图片 ：%s,iou:%r ,识别正确！",os.path.basename(im_fn),iou)
                cnt_true += 1
                break
    logger.info("验证图片总数：%r,验证正确总条数：%r",len(im_fn_list),cnt_true)
    F1 = cnt_true / len(im_fn_list)
    return F1


def cal_iou(box_a, box_b):
    # line1 = [908, 215, 934, 312, 752, 355, 728, 252]  # 四边形四个点坐标的一维数组表示，[x,y,x,y....]
    # a = np.array(line1).reshape(4, 2)  # 四边形二维坐标表示
    poly1 = Polygon(box_a).convex_hull  # python四边形对象，会自动计算四个点，最后四个点顺序为：左上 左下  右下 右上 左上
    # print(Polygon(a).convex_hull)  # 可以打印看看是不是这样子

    # line2 = [923, 308, 758, 342, 741, 262, 907, 228]
    # b = np.array(line2).reshape(4, 2)
    poly2 = Polygon(box_b).convex_hull
    # print(Polygon(b).convex_hull)

    union_poly = np.concatenate((box_a, box_b))  # 合并两个box坐标，变为8*2
    # print(union_poly)
    print(MultiPoint(union_poly).convex_hull)  # 包含两四边形最小的多边形点
    if not poly1.intersects(poly2):  # 如果两四边形不相交
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area  # 相交面积
            print(inter_area)
            # union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            print(union_area)
            if union_area == 0:
                iou = 0
            # iou = float(inter_area) / (union_area-inter_area)  #错了
            iou = float(inter_area) / union_area
            # iou=float(inter_area) /(poly1.area+poly2.area-inter_area)
            # 源码中给出了两种IOU计算方式，第一种计算的是: 交集部分/包含两个四边形最小多边形的面积
            # 第二种： 交集 / 并集（常见矩形框IOU计算方式）
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou


if __name__ == '__main__':
    # tf.app.run()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    params = pred.initialize()
    F1 =validate(params)
    print("正确率：",F1)