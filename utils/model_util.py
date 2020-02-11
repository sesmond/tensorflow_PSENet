#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    模型相关工具类
    1、 恢复模型
    2、 保存模型
"""

# coding: utf-8
"""
模型转换 ckpt > pb
"""
import os
import tensorflow as tf

from nets import model



def save_model(sess,global_step,save_mod_dir):
    # 每次转换都生成一个版本目录
    for i in range(100000, 9999999):
        cur = os.path.join(save_mod_dir, str(i))
        if not tf.gfile.Exists(cur):
            save_mod_dir = cur
            break
    tf.get_variable_scope().reuse_variables()
    # 定义张量
    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    seg_maps_pred = model.model(input_images, is_training=False)
    # 保存转换训练好的模型
    builder = tf.saved_model.builder.SavedModelBuilder(save_mod_dir)
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
    print("转换模型结束", save_mod_dir)


def restore_model_by_dir(model_path):
    """
        从目录下寻找最新的模型加载
    :param model_path:
    :return:
    """
    f_list = os.listdir(model_path)
    dirs = [i for i in f_list if os.path.isdir(os.path.join(model_path, i))]
    max_dir = max(dirs)
    return restore_model(os.path.join(model_path, max_dir))


def restore_model(model_path):
    """
        直接指定模型
    :param model_path:
    :return:
    """
    print("恢复模型：",model_path)
    params={}
    g = tf.get_default_graph()
    with g.as_default():
        #从pb模型直接恢复
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        init = tf.global_variables_initializer()
        sess.run(init)
        meta_graph_def = tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path)
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

if __name__ == '__main__':
    save_model()
