# -*- coding:utf-8 -*-
import cv2
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from utils.utils_tool import logger, cfg
from pse import pse
from utils import plate_utils
from utils import model_util

tf.app.flags.DEFINE_string('pred_data_path', './data/pred/input', '')
tf.app.flags.DEFINE_string('pred_gpu_list', '', '')
tf.app.flags.DEFINE_string('pred_model_path', './model/multi_pb', '')
tf.app.flags.DEFINE_string('output_dir', './data/pred/output', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')


FLAGS = tf.app.flags.FLAGS

logger.setLevel(cfg.debug)

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.pred_data_path):
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


def detect(seg_maps, timer, image_w, image_h, min_area_thresh=10, seg_map_thresh=0.9, ratio = 1):
    '''
    restore text boxes from score map and geo map
    :param seg_maps:
    :param timer:
    :param min_area_thresh:
    :param seg_map_thresh: threshhold for seg map
    :param ratio: compute each seg map thresh
    :return:
    '''
    #多出一维的时候去掉，得到 w,h,6
    if len(seg_maps.shape) == 4:
        seg_maps = seg_maps[0, :, :, ]
    # seg_maps shape w,h,6
    #get kernals, sequence: 0->n, max -> min
    kernals = []
    # (w,h) 都是1，都是0
    one = np.ones_like(seg_maps[..., 0], dtype=np.uint8)
    zero = np.zeros_like(seg_maps[..., 0], dtype=np.uint8)
    thresh = seg_map_thresh
    for i in range(seg_maps.shape[-1]-1, -1, -1):
        #TODO 根据阈值（0.9）二分为0，1
        kernal = np.where(seg_maps[..., i]>thresh, one, zero)
        kernals.append(kernal)
        thresh = seg_map_thresh*ratio
    start = time.time()
    mask_res, label_values = pse(kernals, min_area_thresh)
    timer['pse'] = time.time()-start
    mask_res = np.array(mask_res)
    mask_res_resized = cv2.resize(mask_res, (image_w, image_h), interpolation=cv2.INTER_NEAREST)
    boxes = []
    # plt.imshow(mask_res_resized)
    # ()

    for label_value in label_values:
        #(y,x)
        points = np.argwhere(mask_res_resized==label_value)
        points = points[:, (1,0)]
        #TODO 这里不一定是retangle吧
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        #TODO 这里找出来的点 points 是一个小框，然后这些点怎么换算成坐标！
        # rect = cv2.minAreaRect(points)
        binary = np.zeros(mask_res_resized.shape, dtype='uint8')
        binary[mask_res_resized == label_value] = 1
        #TODO !!
        # https://www.cnblogs.com/GaloisY/p/11062065.html
        _, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # 如果距离小于第二个参数的点则被舍弃掉 适合正方形不适合矩形
        # dp_poly  =cv2.approxPolyDP(curve=points,epsilon=10,closed=True)


        # 检测四边形：
        # https://stackoverflow.com/questions/37942132/opencv-detect-quadrilateral-in-python

        #TODO!!
        if len(contours)<=0:
            continue
        contour = contours[0]
        #TODO 寻找凸包
        bbox = cv2.convexHull(contour)

        if bbox.shape[0] <= 2:
            continue

        # bbox = bbox * scale
        bbox = bbox.astype('int32')
        new_box = bbox.reshape(-1,2) # 转换成2点坐标

        # 切为2点
        # area, v1, v2, v3, v4, _, _ = plate_utils.mep(new_box)
        # box = [v1, v2, v3, v4]
        # box = np.array(box)

        # print("new_box and box :\n", new_box,box)
        #TODO 画图并展示
        # pts = np.array(new_box, np.int32)
        # pts = pts.reshape(-1,1,2)
        # TODO 划线 多余4点坐标
        # cv2.polylines(mask_res_resized,[pts], True, color=(200, 200,200),
        #               thickness=3)
        # plt.imshow(mask_res_resized)
        # plt.show()
        boxes.append(new_box)
        # boxes.append(box)

    return np.array(boxes), kernals, timer


# 调用前向运算来计算
def predict_by_network(params,img):
    # 还原各类张量
    t_input_images = params["input_images"]
    t_seg_maps_pred = params["seg_maps_pred"]
    session = params["session"]
    g = params["graph"]

    with g.as_default():
        # logger.debug("通过session预测：%r",img.shape)
        seg_maps = session.run(t_seg_maps_pred, feed_dict={t_input_images: [img]})

    return seg_maps


# 定义图，并且还原模型，创建session
def initialize():
    logger.info("恢复模型，路径：%s",FLAGS.pred_model_path)
    return model_util.restore_model_by_dir(FLAGS.pred_model_path)


def pred(params, im, im_fn):
    """
        预测单张图片
    :param params: tensorflow 参数
    :param im: 图像文件BGR
    :param im_fn: 文件名（打日志用）
    :return: 返回单张图片的文字区域坐标 TODO 把几点坐标做成动态的
    """
    # 色彩转换 BGR 转 RGB
    im_new = im[:, :, ::-1]
    start_time = time.time()
    im_resized, (ratio_h, ratio_w) = resize_image(im_new)
    h, w, _ = im_resized.shape
    timer = {'net': 0, 'pse': 0}
    start = time.time()
    # resnet预测得到F（S1，...S6）
    seg_maps = predict_by_network(params, im_resized)
    timer['net'] = time.time() - start
    # pse算法后处理，合并多层预测结果为一个框
    boxes, kernels, timer = detect(seg_maps=seg_maps, timer=timer, image_w=w, image_h=h)
    # print("pse后box：", boxes.shape)
    logger.info('{} : net {:.0f}ms, pse {:.0f}ms'.format(
        im_fn, timer['net'] * 1000, timer['pse'] * 1000))
    if boxes is not None:
        # boxes = boxes.reshape((-1, -1, 2))
        h, w, _ = im_new.shape
        # 图片大小还原
        for box in boxes:
            box[:, 0] = box[:, 0] / ratio_w
            box[:, 1] = box[:, 1] / ratio_h
            # 最小是0，最大是h？ 操作完之后防止产生小于0 大于边界的值
            box[:, 0] = np.clip(box[:, 0], 0, w)
            box[:, 1] = np.clip(box[:, 1], 0, h)
    duration = time.time() - start_time
    logger.info('[timing] {}'.format(duration))
    logger.info("pred box len:{}".format(len(boxes)))
    return boxes

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.pred_gpu_list

    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    params = initialize()

    # 遍历所有图片预测
    im_fn_list = get_images()
    for im_fn in im_fn_list:
        logger.debug('image file:{}'.format(im_fn))
        im = cv2.imread(im_fn)
        # 预测
        boxes = pred(params, im, im_fn)

        # save to file
        if boxes is not None:
            res_file = os.path.join(
                FLAGS.output_dir,
                '{}.txt'.format(os.path.splitext(os.path.basename(im_fn))[0]))
            with open(res_file, 'w') as f:
                num =0
                for i in range(len(boxes)):
                    # to avoid submitting errors
                    box = boxes[i]
                    print("预测box：",box)
                    #TODO
                    # if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    #     continue

                    num += 1
                    # 左下开始逆时针坐标： 左下 左上 右上 右下 ，还是四点坐标
                    f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0], box[3, 1]))
                    # 划线
                    cv2.polylines(im, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=2)
                    #TODO 测试转换后的小图

                    # warped = plate_utils.four_point_transform(im,box)
                    # img_path = os.path.join(FLAGS.output_dir, "plate_" + os.path.basename(im_fn) + str(i) + ".jpg")
                    # cv2.imwrite(img_path, warped)


        if not FLAGS.no_write_images:
            img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
            cv2.imwrite(img_path, im)


if __name__ == '__main__':
    tf.app.run()
