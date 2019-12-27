import time
import datetime

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from utils.utils_tool import logger, cfg

tf.app.flags.DEFINE_string('name', 'psenet', '')
tf.app.flags.DEFINE_integer('input_size', 512, '')
tf.app.flags.DEFINE_integer('batch_size_per_gpu', 1, '')
tf.app.flags.DEFINE_integer('num_readers', 1, '')
tf.app.flags.DEFINE_float('learning_rate', 0.00001, '')
tf.app.flags.DEFINE_integer('max_steps', 100000, '')
# TODO 设置早停loss

tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_string('gpu_list', '1', '')
tf.app.flags.DEFINE_string('checkpoint_path', './model/', '')
tf.app.flags.DEFINE_string('tboard_path', './tboard/', '')
tf.app.flags.DEFINE_integer('save_checkpoint_steps', 1000, '')
tf.app.flags.DEFINE_integer('save_summary_steps', 100, '')
tf.app.flags.DEFINE_boolean('restore', False, 'whether to resotre from checkpoint')
# TODO 二次训练的旧模型 微调使用
tf.app.flags.DEFINE_string('pretrained_model_path', None, '')


from nets import model
from utils.data_provider import data_provider

FLAGS = tf.app.flags.FLAGS

gpus = list(range(len(FLAGS.gpu_list.split(','))))

logger.setLevel(cfg.debug)


def tower_loss(images, seg_maps_gt, training_masks, reuse_variables=None):
    """
        损失函数计算
    :param images:
    :param seg_maps_gt:
    :param training_masks:
    :param reuse_variables:
    :return:
    """
    # Build inference graph
    # 加载模型
    print("seg_maps_gt:", seg_maps_gt.shape)
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        seg_maps_pred = model.model(images, is_training=True)
    # TODO 损失函数
    model_loss = model.loss(seg_maps_gt, seg_maps_pred, training_masks)
    # TODO
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # add summary
    if reuse_variables is None:
        tf.summary.image('input', images)
        # TODO 这里列的是第几个？ 反正不是最大的那个 可以把最大的和最小的都画出来看看
        # TODO 这里有问题 gt不应该有问题
        tf.summary.image('seg_map_0_gt', seg_maps_gt[:, :, :, 0:1] * 255)
        tf.summary.image('seg_map_0_pred', seg_maps_pred[:, :, :, 0:1] * 255)
        tf.summary.image('training_masks', training_masks)
        tf.summary.scalar('model_loss', model_loss)
        tf.summary.scalar('total_loss', total_loss)

    return total_loss, model_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    if not tf.gfile.Exists(FLAGS.checkpoint_path):
        tf.gfile.MkDir(FLAGS.checkpoint_path)
    # else:
    # TODO 没必要删除吧
    # if not FLAGS.restore:
    #     tf.gfile.DeleteRecursively(FLAGS.checkpoint_path)
    #     tf.gfile.MkDir(FLAGS.checkpoint_path)
    # print("")

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    input_seg_maps = tf.placeholder(tf.float32, shape=[None, None, None, 6], name='input_score_maps')
    input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.94,
                                               staircase=True)
    # add summary
    tf.summary.scalar('learning_rate', learning_rate)
    # opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.9)
    opt = tf.train.AdamOptimizer(learning_rate)
    # opt = tf.train.MomentumOptimizer(learning_rate, 0.9)

    # split
    input_images_split = tf.split(input_images, len(gpus))
    input_seg_maps_split = tf.split(input_seg_maps, len(gpus))
    input_training_masks_split = tf.split(input_training_masks, len(gpus))

    tower_grads = []
    reuse_variables = None
    for i, gpu_id in enumerate(gpus):
        with tf.device('/gpu:%d' % gpu_id):
            with tf.name_scope('model_%d' % gpu_id) as scope:
                iis = input_images_split[i]
                # groundtruth 标注数据
                isegs = input_seg_maps_split[i]
                itms = input_training_masks_split[i]
                total_loss, model_loss = tower_loss(iis, isegs, itms, reuse_variables)
                batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope))
                reuse_variables = True

                grads = opt.compute_gradients(total_loss)
                tower_grads.append(grads)

    grads = average_gradients(tower_grads)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    summary_op = tf.summary.merge_all()
    # save moving average
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # batch norm updates
    with tf.control_dependencies([variables_averages_op, apply_gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')  # 什么都不做，仅做为点位符使用控制边界。TODO

    saver = tf.train.Saver(tf.global_variables())
    today = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = os.path.join(FLAGS.tboard_path, today)
    # tboard PATH
    summary_writer = tf.summary.FileWriter(summary_dir, tf.get_default_graph())

    init = tf.global_variables_initializer()

    if FLAGS.pretrained_model_path is not None:
        variable_restore_op = slim.assign_from_checkpoint_fn(FLAGS.pretrained_model_path,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
    gpu_options = tf.GPUOptions(allow_growth=True)
    # gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)) as sess:
        if FLAGS.restore:
            logger.info('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
            logger.debug(ckpt)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)
            if FLAGS.pretrained_model_path is not None:
                variable_restore_op(sess)

        data_generator = data_provider.get_batch(num_workers=FLAGS.num_readers,
                                                 input_size=FLAGS.input_size,
                                                 batch_size=FLAGS.batch_size_per_gpu * len(gpus))

        start = time.time()
        for step in range(FLAGS.max_steps):
            # TODO 返回迭代器的下一个项目 TODO feed_dict 三个参数都是在这里被赋值的，样本解析就在这里。
            data = next(data_generator)
            # 每步提取10张图片
            # TODO
            logger.info("训练步数:%d", step)
            ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: data[0],
                                                                                input_seg_maps: data[2],
                                                                                input_training_masks: data[3]})
            # 为null
            if np.isnan(tl):
                logger.error('Loss diverged, stop training')
                break

            if step % 10 == 0:
                # 每10次打一次日志？
                avg_time_per_step = (time.time() - start) / 10
                avg_examples_per_second = (10 * FLAGS.batch_size_per_gpu * len(gpus)) / (time.time() - start)
                start = time.time()
                # TODO
                logger.info(
                    'Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step, {:.2f} examples/second'.format(
                        step, ml, tl, avg_time_per_step, avg_examples_per_second))

            # 每1000次保存一次模型
            if step % FLAGS.save_checkpoint_steps == 0:
                # TODO 如果不超过还要保存吗？ 没有记录上次最好数据只是强制保存？ 还有是否设计早停
                # 模型名为了不冲突最好加上别的名字
                train_start_time = time.strftime('%Y%m%d-%H%M', time.localtime(time.time()))
                model_name = 'model_{:s}.ckpt-{:s}'.format(str(train_start_time), str(tl))
                model_save_path = os.path.join(FLAGS.checkpoint_path, model_name)
                saver.save(sess, model_save_path, global_step=global_step)
            # 每100 次算一下损失函数写入tensorboard
            if step % FLAGS.save_summary_steps == 0:
                # TODO !!  这里主要是为了执行summary_op 吧
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                             input_seg_maps: data[2],
                                                                                             input_training_masks: data[
                                                                                                 3]})

                logger.info("write into board,Step {:06d}, model loss {:.4f}, total loss {:.4f}".format(step, ml, tl))
                summary_writer.add_summary(summary_str, global_step=step)


if __name__ == '__main__':
    tf.app.run()
