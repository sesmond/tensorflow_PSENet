# encoding:utf-8
"""
训练时抽取数据的工具集
"""
import os
import glob
import time
import random
import traceback
import cv2
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from utils.utils_tool import logger
from utils.data_provider.data_util import GeneratorEnqueuer
import tensorflow as tf
import pyclipper
from utils.data_provider import data_reader


tf.app.flags.DEFINE_string('data_type', 'plate', 'dataset type')  # 必须指定
tf.app.flags.DEFINE_string('train_data_config', "cfg/train_data.cfg",
                           'training data config file ')
tf.app.flags.DEFINE_integer('max_image_large_side', 1280,
                            'max image size of training')
tf.app.flags.DEFINE_integer('max_text_size', 800,
                            'if the text in the input image is bigger than this, then we resize'
                            'the image according to this')
tf.app.flags.DEFINE_integer('min_text_area_size', 5,
                            'if the text area size is smaller than this, we ignore it during training')
tf.app.flags.DEFINE_integer('min_text_width', 2,
                            '框的左右或上下如果相距小于这个值，认为框太小不参与训练。')
tf.app.flags.DEFINE_integer('min_text_height', 3,
                            '框的左右或上下如果相距小于这个值，认为框太小不参与训练。')
tf.app.flags.DEFINE_float('min_crop_side_ratio', 0.1,
                          'when doing random crop from input image, the'
                          'min length of min(H, W')

FLAGS = tf.app.flags.FLAGS


def get_files(data_path):
    """
    获取目录下以及子目录下的图片
    :param data_path:
    :return:
    """
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for ext in exts:
        # glob.glob 得到所有文件名
        # 一层 2层子目录都取出来
        files.extend(glob.glob(os.path.join(data_path, '*.{}'.format(ext))))
        files.extend(glob.glob(os.path.join(data_path, '*', '*.{}'.format(ext))))
    return files


def check_and_validate_polys(polys, tags, xxx_todo_changeme):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    # 调整为顺时针方向 从左下开始？TODO 待细看
    (h, w) = xxx_todo_changeme
    if polys.shape[0] == 0:
        return [], []
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    validated_polys = []
    validated_tags = []
    # TODO !!
    for poly, tag in zip(polys, tags):
        w_len = max(poly[:, 0]) - min(poly[:, 0])
        h_len = max(poly[:, 1]) - min(poly[:, 1])
        if w_len < FLAGS.min_text_width or h_len < FLAGS.min_text_height:
            continue
        # 文本框面积小于20则不用做训练，太小了不认为是文字
        if abs(pyclipper.Area(poly)) < FLAGS.min_text_area_size:
            continue
        # clockwise
        if pyclipper.Orientation(poly):
            poly = poly[::-1]

        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def crop_area(im, polys, tags, crop_background=False, max_tries=50):
    """
    make random crop from the input image / 切割原图，二次生成样本
    :param im:
    :param polys:
    :param tags:
    :param crop_background:
    :param max_tries:
    :return:
    """
    h, w, _ = im.shape
    pad_h = h // 10
    pad_w = w // 10
    # 边缘扩充10%
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    for poly in polys:
        # 四舍五入
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        # 横向最大最小之间设为1
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        # 纵向最大最小之间设为1
        h_array[miny + pad_h:maxy + pad_h] = 1
    # 以上 所有框合并

    # ensure the cropped area not across a text
    # 找出剩下的没有盖到的位置区域
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        # 说明横向或者竖向都被文字填充满了，但是上面+padding了，所以是不可能走到这里的
        return im, polys, tags
    for i in range(max_tries):
        # 最多尝试50次
        # x y 方向任选两个没有文字区域的点
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        # 如果空白框小于原图10%，则再次尝试
        if xmax - xmin < FLAGS.min_crop_side_ratio * w or ymax - ymin < FLAGS.min_crop_side_ratio * h:
            # area too small
            continue
        # 如果有文本框
        if polys.shape[0] != 0:
            # 坐标在上面生成的框之内
            # 所有点的x坐标在范围内，y坐标再范围内
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            # 统计坐标点在范围内的个数 ！！！ 4点坐标是等于4 ，这里应该要么是0，要么就是所有点 改为>=0 暂时适配所有
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) >= 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area 切割框内无文本 TODO 没看懂
            if crop_background:
                return im[ymin:ymax + 1, xmin:xmax + 1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        # 切割图片
        im = im[ymin:ymax + 1, xmin:xmax + 1, :]
        # 切割范围内的文本框
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        # 相对坐标切换成新图
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags
    # 如果尝试50次没有取到则直接返回原图
    return im, polys, tags


def perimeter(poly):
    # 计算周长
    try:
        p = 0
        nums = poly.shape[0]
        for i in range(nums):
            p += abs(np.linalg.norm(poly[i % nums] - poly[(i + 1) % nums]))
        # logger.debug('perimeter:{}'.format(p))
        return p
    except Exception as e:
        traceback.print_exc()
        raise e


def shrink_poly(poly, r):
    """
        收缩多边形
    :param poly: 多边形
    :param r: 收缩比例
    :return:
    """
    # TODO debug调试 TODO 这里四点坐标有问题
    try:
        area_poly = abs(pyclipper.Area(poly))
        perimeter_poly = perimeter(poly)
        poly_s = []
        pco = pyclipper.PyclipperOffset()
        if perimeter_poly:
            d = area_poly * (1 - r * r) / perimeter_poly
            # offset = min((int)(area * (1 - rate) / (peri + 0.001) + 0.5), max_shr)

            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            # TODO 什么逻辑？ 这个没搞明白
            # 缩小后返回多边形
            poly_s = pco.Execute(-d)
        # print("原框：",poly)
        # print("缩小倍数：",r)
        # print("返回结果：",poly_s[0])
        # TODO 可能一个都没有 处理一下
        # shrinked_bbox = np.array(poly_s[0])
        # print("缩放后点个数:",shrinked_bbox.shape[0])
        # TODO 这里如果不是6个怎么办？
        if len(poly_s)>0:
            return [poly_s[0]]
        else:
            # logger.error("shrink poly is too small：%r,%r",poly,r)
            return []
        # TODO 可能前面的坐标转换有问题
        # return [poly]
    except Exception as e:
        traceback.print_exc()
        raise e


# TODO:filter small text(when shrincked region shape is 0 no matter what scale ratio is)
def generate_seg(im_size, polys, tags, image_name, scale_ratio):
    '''
    :param im_size: input image size
    :param polys: input text regions
    :param tags: ignore text regions tags
    :param image_index: for log
    :param scale_ratio:ground truth scale ratio, default[0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    :return:
    seg_maps: segmentation results with different scale ratio, save in different channel
    training_mask: ignore text regions
    '''
    # TODO 一张生成6张seomap
    h, w = im_size
    # mark different text poly 最终输出6张
    seg_maps = np.zeros((h, w, 6), dtype=np.uint8)
    # mask used during traning, to ignore some hard areas
    training_mask = np.ones((h, w), dtype=np.uint8)
    ignore_poly_mark = []
    for i in range(len(scale_ratio)):
        seg_map = np.zeros((h, w), dtype=np.uint8)
        # 兼容多点坐标
        for poly_idx, poly_tag in enumerate(zip(polys, tags)):
            # 对每个多边形操作
            poly = poly_tag[0]
            tag = poly_tag[1]

            # ignore ### icdar样本会有些没文字的标志
            if i == 0 and tag:
                # 填0 TODO
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_poly_mark.append(poly_idx)

            # seg map
            shrinked_polys = []
            if poly_idx not in ignore_poly_mark:
                # 收缩原框，返回缩小后的小框
                # 这里缩小框点个数不确定，不一定多少个
                shrinked_polys = shrink_poly(poly.copy(), scale_ratio[i])
            #
            if not len(shrinked_polys) and poly_idx not in ignore_poly_mark:
                logger.info("before shrink poly area:{} len(shrinked_poly) is 0,image {}".format(
                    abs(pyclipper.Area(poly)), image_name))
                # if the poly is too small, then ignore it during training
                cv2.fillPoly(training_mask, poly.astype(np.int32)[np.newaxis, :, :], 0)
                ignore_poly_mark.append(poly_idx)
                continue
            # 将缩放后的样本gt框画到seg_map上
            for shrinked_poly in shrinked_polys:
                # TODO color1 是黑色？
                # 看看是不是这个填充的问题，先画框试试
                seg_map = cv2.fillPoly(seg_map, [np.array(shrinked_poly).astype(np.int32)], 1)
                # seg_map = cv2.drawContours(seg_map, [np.array(shrinked_poly).astype(np.int32)], -1, 1, -1)
        # (h,w,6) 返回6张图
        seg_maps[..., i] = seg_map
    return seg_maps, training_mask


# TODO batch size =32
def generator(input_size=512, batch_size=2,
              background_ratio=3. / 8,
              random_scale=np.array([0.125, 0.25, 0.5, 1, 2.0, 3.0]),
              vis=False,
              scale_ratio=np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])):
    '''
    reference from https://github.com/argman/EAST
    :param input_size:
    :param batch_size:
    :param background_ratio:
    :param random_scale:
    :param vis:
    :param scale_ratio:ground truth scale ratio
    :return:
    '''
    # 参与训练的所有图片名
    paths = open(FLAGS.train_data_config, "r").readlines()
    print("paths:", paths)
    sel_type = random.choice(paths)
    sel_type = sel_type.rstrip('\n')
    print("sel_type:", sel_type)
    img_path, label_path, data_type = sel_type.split(" ")

    image_list = np.array(get_files(img_path))

    logger.info('{} training images in {}'.format(
        image_list.shape[0], img_path))
    # 索引数组
    index = np.arange(0, image_list.shape[0])
    real_reader = data_reader.get_data_reader(data_type)
    while True:
        # 随机排序
        np.random.shuffle(index)
        images = []
        image_fns = []
        seg_maps = []
        training_masks = []
        for i in index:
            try:
                im_fn = image_list[i]
                # logger.info("读取文件：%s", im_fn)
                im = cv2.imread(im_fn)
                if im is None:
                    logger.info("图像没有找到：%s", im_fn)
                    continue
                h, w, _ = im.shape
                # 根据图片名找到对应样本标注
                success, text_polys, text_tags = real_reader.get_annotation(im_fn, label_path)
                if not success:
                    # logger.error("没有解析到文本框：%r ,",im_fn)
                    continue
                # 没有标注框
                if text_polys.shape[0] == 0:
                    continue
                #TODO resize之后图片会缩小，所以坐标也会相应缩小，在这里校验似乎不太好？
                text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))
                # 这是要缩放图片吗？是为了放大字体营造样本多样性？
                # random scale this image
                rd_scale = np.random.choice(random_scale)
                # TODO debug竟然卡死在这里！
                im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale, interpolation=cv2.INTER_AREA)
                text_polys *= rd_scale
                # random crop a area from image
                if np.random.rand() < background_ratio:
                    # crop background 从原图中切出不带文字的图作为负样本
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=True)
                    # 如果切出来的图里有文本则舍弃掉继续
                    if text_polys.shape[0] > 0:
                        # 切出来的框里有文本，则不是负样本舍弃掉继续
                        # TODO 但是这样其实浪费了很多资源，这只是为了找一个没有文本框的切图制造负样本而已
                        # cannot find background
                        continue
                    # 画一下负样本
                    # pad and resize image
                    new_h, new_w, _ = im.shape
                    # max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = cv2.resize(im_padded, dsize=(input_size, input_size))
                    seg_map_per_image = np.zeros((input_size, input_size, scale_ratio.shape[0]), dtype=np.uint8)
                    training_mask = np.ones((input_size, input_size), dtype=np.uint8)
                else:
                    im, text_polys, text_tags = crop_area(im, text_polys, text_tags, crop_background=False)
                    if text_polys.shape[0] == 0:
                        continue

                    # pad the image to the training input size or the longer side of image
                    new_h, new_w, _ = im.shape
                    # max_h_w_i = np.max([new_h, new_w, input_size])
                    im_padded = np.zeros((new_h, new_w, 3), dtype=np.uint8)
                    im_padded[:new_h, :new_w, :] = im.copy()
                    im = im_padded
                    # resize the image to input size
                    new_h, new_w, _ = im.shape
                    resize_h = input_size
                    resize_w = input_size
                    # TODO 本地这里会卡死
                    # print(resize_w,resize_h)
                    im = cv2.resize(im, dsize=(resize_w, resize_h))
                    resize_ratio_3_x = resize_w / float(new_w)
                    resize_ratio_3_y = resize_h / float(new_h)
                    text_polys[:, :, 0] *= resize_ratio_3_x
                    text_polys[:, :, 1] *= resize_ratio_3_y
                    new_h, new_w, _ = im.shape
                    seg_map_per_image, training_mask = generate_seg((new_h, new_w), text_polys, text_tags,
                                                                    image_list[i], scale_ratio)
                    if not len(seg_map_per_image):
                        logger.info("len(seg_map)==0 image: %d " % i)
                        continue
                # DEBUG
                _debug_show(vis, im, seg_map_per_image, training_mask)

                # TODO 单通道？
                images.append(im[..., ::-1].astype(np.float32))
                image_fns.append(im_fn)
                seg_maps.append(seg_map_per_image[::4, ::4, :].astype(np.float32))
                training_masks.append(training_mask[::4, ::4, np.newaxis].astype(np.float32))

                if len(images) == batch_size:
                    # logger.debug("获取样本数：%d", len(images))
                    # 返回 图片，文件名，gt，mask
                    yield images, image_fns, seg_maps, training_masks
                    images = []
                    image_fns = []
                    seg_maps = []
                    training_masks = []
            except Exception as e:
                traceback.print_exc()
                logger.error("解析出错 file:%s",im_fn)
                continue


def _debug_show(vis, im, seg_map_per_image, training_mask):
    """
        debug 调试时显示照片
    """
    if vis:
        # debug调试用的 原图 6张图 &掩码图
        fig, axs = plt.subplots(3, 3, figsize=(20, 30))
        axs[0, 0].imshow(im[..., ::-1])
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        axs[0, 1].imshow(seg_map_per_image[..., 0])
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].imshow(seg_map_per_image[..., 1])
        axs[0, 2].set_xticks([])
        axs[0, 2].set_yticks([])
        axs[1, 0].imshow(seg_map_per_image[..., 2])
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(seg_map_per_image[..., 3])
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        axs[1, 2].imshow(seg_map_per_image[..., 4])
        axs[1, 2].set_xticks([])
        axs[1, 2].set_yticks([])
        axs[2, 0].imshow(seg_map_per_image[..., 5])
        axs[2, 0].set_xticks([])
        axs[2, 0].set_yticks([])
        axs[2, 1].imshow(training_mask)
        axs[2, 1].set_xticks([])
        axs[2, 1].set_yticks([])
        plt.tight_layout()
        plt.show()
        plt.close()


def get_batch(num_workers, **kwargs):
    try:
        # TODO 是否使用多线程
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    # print("休眠0.01")
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


if __name__ == '__main__':
    # gen = get_batch(num_workers=1, vis=True)
    # while True:
    #     images, image_fns, seg_maps, training_masks = next(gen)
    #     logger.debug('done')
    # # print("")
    gen  =generator(vis=True,batch_size=10)
    images, image_fns, seg_maps, training_masks = next(gen)
    print("")