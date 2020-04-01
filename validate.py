# -*- coding:utf-8 -*-
import cv2
import time
import os
import numpy as np
import tensorflow as tf
from utils.utils_tool import logger, cfg
from utils.data_provider import data_provider
from utils.data_provider import data_reader
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形
import random

# TODO 验证集数据
import pred.pred as pred

tf.app.flags.DEFINE_string('validate_data_config', './cfg/validate_data.cfg', '')

FLAGS = tf.app.flags.FLAGS

logger.setLevel(cfg.debug)

"""
    验证数据，通过计算IOU进行判断
"""


def get_images(path):
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    logger.info('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=1200):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.

    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 + 1) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 + 1) * 32
    logger.info('resize_w:{}, resize_h:{}'.format(resize_w, resize_h))
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)


def predit_one_img(img, text_polys, text_tags, scale_ratio=np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])):
    # 色彩转换 BGR 转 RGB
    im_new = img[:, :, ::-1]
    im_resized, (ratio_h, ratio_w) = resize_image(im_new)
    h, w, _ = im_resized.shape

    text_polys[:, :, 0] *= ratio_w
    text_polys[:, :, 1] *= ratio_h

    seg_map_gt, training_mask = data_provider.generate_seg((h, w), text_polys, text_tags,
                                                           im_resized, scale_ratio)
    return im_resized, seg_map_gt, training_mask


def validate(params):
    # 参与训练的所有图片名
    paths = open(FLAGS.validate_data_config, "r").readlines()
    print("validate paths:", paths)
    for sel_type in paths:
        sel_type = sel_type.rstrip('\n')
        print("sel_type:", sel_type)
        img_path, label_path, data_type = sel_type.split(" ")
        im_fn_list = get_images(img_path)
        logger.info('{} validate images in {}'.format(len(im_fn_list), img_path))
        # 索引数组
        real_reader = data_reader.get_data_reader(data_type)
        # 批量验证数据 计算F1 recall等
        # TODO 筛选100个
        im_fn_list = random.sample(im_fn_list, 100)
        logger.info("随机抽取样本数进行验证：%r", len(im_fn_list))
        # 计算IOU 大于0.7的就算预测正确
        cnt_true = 0  # 正确条数 TODO 召回率
        for im_fn in im_fn_list:
            logger.debug('image file:{}'.format(im_fn))
            img = cv2.imread(im_fn)
            # 根据图片名找到对应样本标注
            success, text_polys, text_tags = real_reader.get_annotation(im_fn, label_path)
            if not success:
                continue
            im_resized, seg_gt, mask = predit_one_img(img, text_polys, text_tags)
            seg_gt_maps=[]
            training_masks=[]
            seg_gt_maps.append(seg_gt[::4, ::4, :].astype(np.float32))
            training_masks.append(mask[::4, ::4, np.newaxis].astype(np.float32))

            # resnet预测得到F（S1，...S6）
            # TODO!!!! 如果没有pse则不用他的，单独拿出来即可
            seg_pred = pred.predict_by_network(params, im_resized)

            t_l = loss(np.array(seg_gt_maps), seg_pred, np.array([training_masks]))
            if t_l < 0.3:
                cnt_true += 1
        logger.info("验证图片总数：%r,验证正确总条数：%r", len(im_fn_list), cnt_true)
        accuracy = cnt_true / len(im_fn_list)
        return accuracy


def loss(y_true_cls, y_pred_cls,
         training_mask):
    """
    损失函数计算
    :param y_true_cls: gt
    :param y_pred_cls: 预测值
    :param training_mask: 掩码
    :return:
    """
    g1, g2, g3, g4, g5, g6 = tf.split(value=y_true_cls, num_or_size_splits=6, axis=3)
    s1, s2, s3, s4, s5, s6 = tf.split(value=y_pred_cls, num_or_size_splits=6, axis=3)
    Gn = [g1, g2, g3, g4, g5, g6]
    Sn = [s1, s2, s3, s4, s5, s6]
    # 比较最大的框，计算出Lc，即表示没有进行缩放时候的损失函数
    _, Lc = dice_coefficient(Gn[5], Sn[5], training_mask=training_mask)

    one = tf.ones_like(Sn[5])
    zero = tf.zeros_like(Sn[5])
    W = tf.where(Sn[5] >= 0.5, x=one, y=zero)
    D = 0
    for i in range(5):
        di, _ = dice_coefficient(Gn[i]*W, Sn[i]*W, training_mask=training_mask)
        D += di
    #Ls 是缩放后的5个框的损失函数取平均值
    Ls = 1-D/5.
    # 原框Lc所占比例
    lambda_ = 0.7
    L = lambda_*Lc + (1-lambda_)*Ls
    return L


def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls: ground truth
    :param y_pred_cls: predict
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    dice = 2 * intersection / union
    loss = 1. - dice
    # tf.summary.scalar('classification_dice_loss', loss)
    return dice, loss


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
    F1 = validate(params)
    print("正确率：", F1)
