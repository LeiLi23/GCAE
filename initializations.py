import tensorflow as tf
import numpy as np

def weight_variable_glorot(input_dim, output_dim, name=""):  #定义一个权重矩阵
    init_range = np.sqrt(6.0 / (input_dim + output_dim)) # 输入维度和输出维度
    initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
                                maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)   #创建可以变化的W权重矩阵
