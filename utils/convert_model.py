# coding: utf-8
"""
模型转换 ckpt > pb
"""
import os

import tensorflow as tf

from nets import model

tf.app.flags.DEFINE_boolean('debug', True, '')
tf.app.flags.DEFINE_string('ckpt_mod_path', "./model/multi", '')
tf.app.flags.DEFINE_string('save_mod_dir', "./model/multi_pb", '')


FLAGS = tf.app.flags.FLAGS


def mk_dir(path):
    if not os.path.exists(path):
        print("创建目录：",path)
        os.makedirs(path)

def convert():
    # 保存转换好的模型目录
    saveModDir = FLAGS.save_mod_dir
    mk_dir(saveModDir)
    # 每次转换都生成一个版本目录
    for i in range(100000, 9999999):
        cur = os.path.join(saveModDir, str(i))
        if not tf.gfile.Exists(cur):
            saveModDir = cur
            break

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
    ses_config = tf.ConfigProto(gpu_options=gpu_options)
    print("模型保存目录", saveModDir)
    # 原ckpt模型
    ckptModPath = FLAGS.ckpt_mod_path
    print("CKPT模型目录", ckptModPath)
    # 获取字符库

    # 定义张量
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    seg_maps_pred = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    model_path = tf.train.latest_checkpoint(ckptModPath)
    saver.restore(sess, model_path)


    # 保存转换训练好的模型
    builder = tf.saved_model.builder.SavedModelBuilder(saveModDir)
    inputs = {
        "input_data": tf.saved_model.utils.build_tensor_info(input_images)
    }
    # B方案.直接输出一个整个的SparseTensor
    output = {
        "output": tf.saved_model.utils.build_tensor_info(seg_maps_pred),
    }

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs=inputs,
        outputs=output,
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder.add_meta_graph_and_variables(
        sess=sess,
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={ # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature
        }
    )
    builder.save()
    print("转换模型结束", saveModDir)


if __name__ == '__main__':
    convert()
