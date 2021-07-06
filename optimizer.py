import tensorflow as tf

flags = tf.app.flags      #用于为Tensorflow程序实现命令行标志
FLAGS = flags.FLAGS      #用来命令行运行代码时传递参数

class OptimizerGCAE(object):   #利用损失函数和梯度下降对模型进行优化
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        preds_sub = preds  #前驱函数
        labels_sub = labels   #标签
        # 正则化* tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值
        self.cost = norm * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)   # 添加操作到图里面通过更新变量（variables）列表来最小化代价。
        self.grads_vars = self.optimizer.compute_gradients(self.cost)  #利用损失函数计算梯度下降
        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))   #布尔值
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
