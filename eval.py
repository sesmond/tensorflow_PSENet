# -*- coding:utf-8 -*-
import cv2
import time
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.client import timeline
from utils.utils_tool import logger, cfg
import matplotlib.pyplot as plt

# --test_data_path =./ data / pred / input / \
#                      --checkpoint_path =./ model / \
#                                            --output_dir =./ data / pred / output
tf.app.flags.DEFINE_string('test_data_path', './data/pred/input', '')
tf.app.flags.DEFINE_string('gpu_list', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', '/Users/minjianxu/Documents/ocr/psnet/model/model', '')
tf.app.flags.DEFINE_string('output_dir', './data/pred/output', '')
tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

from nets import model
from pse import pse

FLAGS = tf.app.flags.FLAGS

logger.setLevel(cfg.debug)

def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
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

    #ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w


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
        #TODO

        hull = cv2.convexHull(contour)


        bbox = hull
        if bbox.shape[0] <= 2:
            continue
        else:
            print("多边形：",bbox.shape)
        # bbox = bbox * scale
        bbox = bbox.astype('int32')
        new_box = bbox.reshape(-1,2) # 转换成2点坐标
        # print("new_box and box :\n", new_box,box)
        #TODO 画图并展示
        pts = np.array(new_box, np.int32)
        pts = pts.reshape(-1,1,2)
        # TODO 划线 多余4点坐标
        # cv2.polylines(mask_res_resized,[pts], True, color=(200, 200,200),
        #               thickness=3)
        # plt.imshow(mask_res_resized)
        # plt.show()

        # boxes.append(new_box)
        boxes.append(box)

    return np.array(boxes), kernals, timer

def show_score_geo(color_im, kernels, im_res):
    fig = plt.figure()
    cmap = plt.cm.hot
    #
    ax = fig.add_subplot(241)
    im = kernels[0]*255
    ax.imshow(im)

    ax = fig.add_subplot(242)
    im = kernels[1]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(243)
    im = kernels[2]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(244)
    im = kernels[3]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(245)
    im = kernels[4]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(246)
    im = kernels[5]*255
    ax.imshow(im, cmap)

    ax = fig.add_subplot(247)
    im = color_im
    ax.imshow(im)

    ax = fig.add_subplot(248)
    im = im_res
    ax.imshow(im)

    fig.show()


# 调用前向运算来计算
def predict_by_network(params,img):
    # 还原各类张量
    t_input_images = params["input_images"]
    t_seg_maps_pred = params["seg_maps_pred"]
    session = params["session"]
    g = params["graph"]

    with g.as_default():
        logger.debug("通过session预测：%r",img.shape)
        seg_maps = session.run(t_seg_maps_pred, feed_dict={t_input_images: [img]})

    return seg_maps


# 定义图，并且还原模型，创建session
def initialize():
    params = {}
    g = tf.get_default_graph()
    with g.as_default():
        #https://blog.csdn.net/JerryZhang__/article/details/85058005
        # input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        # global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        # seg_maps_pred = model.model(input_images, is_training=False)
        #
        # variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        # saver = tf.train.Saver(variable_averages.variables_to_restore())
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # print("checkpoint path:",FLAGS.checkpoint_path)
        # ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
        # model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
        # logger.info('Restore from {}'.format(model_path))
        # saver.restore(sess, model_path)
        # #TODO !!!
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], "./model/plate/100004")
        signature = meta_graph_def.signature_def
        in_tensor_name = signature['serving_default'].inputs['input_data'].name
        out_tensor_name = signature['serving_default'].outputs['output'].name

        input_images = sess.graph.get_tensor_by_name(in_tensor_name)
        seg_maps_pred = sess.graph.get_tensor_by_name(out_tensor_name)

        params["input_images"] = input_images
        params["seg_maps_pred"] = seg_maps_pred
        params["session"] = sess
        params["graph"] = g

    return params


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
    # options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
    # run_metadata = tf.RunMetadata()
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
    # TODO!!!
    if boxes is not None:
        # TODO 缩放比例
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
    return boxes

def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list

    try:
        os.makedirs(FLAGS.output_dir)
    except OSError as e:
        if e.errno != 17:
            raise

    params = initialize()

    #TODO 遍历所有图片预测
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

        if not FLAGS.no_write_images:
            img_path = os.path.join(FLAGS.output_dir, os.path.basename(im_fn))
            cv2.imwrite(img_path, im)
        # show_score_geo(im_resized, kernels, im)



if __name__ == '__main__':
    tf.app.run()
