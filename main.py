#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import os 
os.environ['CUDA_VISIBLE_DEVICES']='0, 2, 3'

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

# def load_vgg(sess, vgg_path):
#     """
#     Load Pretrained VGG Model into TensorFlow.
#     :param sess: TensorFlow Session
#     :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
#     :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
#     """
#     # TODO: Implement function
#     #   Use tf.saved_model.loader.load to load the model and weights
#     vgg_tag = 'vgg16'
#     vgg_input_tensor_name = 'image_input:0'
#     vgg_keep_prob_tensor_name = 'keep_prob:0'
#     vgg_layer3_out_tensor_name = 'layer3_out:0'
#     vgg_layer4_out_tensor_name = 'layer4_out:0'
#     vgg_layer7_out_tensor_name = 'layer7_out:0'

#     return None, None, None, None, None

def load_vgg(sess, vgg_path):

  # 加载模型和权重
  model = tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)

  # 获取需要返回的 Tensor
  graph = tf.get_default_graph()
  image_input = graph.get_tensor_by_name('image_input:0')
  keep_prob = graph.get_tensor_by_name('keep_prob:0')
  layer3 = graph.get_tensor_by_name('layer3_out:0')
  layer4 = graph.get_tensor_by_name('layer4_out:0')
  layer7 = graph.get_tensor_by_name('layer7_out:0')

  return image_input, keep_prob, layer3, layer4, layer7
tests.test_load_vgg(load_vgg, tf)


# def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
#     """
#     Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
#     :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
#     :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
#     :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
#     :param num_classes: Number of classes to classify
#     :return: The Tensor for the last layer of output
#     """
#     # TODO: Implement function
#     return None

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):

    # 使用简化的变量名
    layer3, layer4, layer7 = vgg_layer3_out, vgg_layer4_out, vgg_layer7_out

    # 应用 1x1 卷积替代全连接
    fcn8 = tf.layers.conv2d(layer7, filters=num_classes, kernel_size=1, name="fcn8")

    # 对 fcn8 进行上采样以匹配 layer 4 的维度
    fcn9 = tf.layers.conv2d_transpose(fcn8, filters=layer4.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn9")

    # 在 fcn9 和 layer 4 之间建立跳远连接
    fcn9_skip_connected = tf.add(fcn9, layer4, name="fcn9_plus_vgg_layer4")

    # 再次执行上采样
    fcn10 = tf.layers.conv2d_transpose(fcn9_skip_connected, filters=layer3.get_shape().as_list()[-1],
    kernel_size=4, strides=(2, 2), padding='SAME', name="fcn10_conv2d")

    # 添加跳远连接
    fcn10_skip_connected = tf.add(fcn10, layer3, name="fcn10_plus_vgg_layer3")

    # 再次执行上采样
    fcn11 = tf.layers.conv2d_transpose(fcn10_skip_connected, filters=num_classes,
    kernel_size=16, strides=(8, 8), padding='SAME', name="fcn11")

    return fcn11
tests.test_layers(layers)


# def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
#     """
#     Build the TensorFLow loss and optimizer operations.
#     :param nn_last_layer: TF Tensor of the last layer in the neural network
#     :param correct_label: TF Placeholder for the correct label image
#     :param learning_rate: TF Placeholder for the learning rate
#     :param num_classes: Number of classes to classify
#     :return: Tuple of (logits, train_op, cross_entropy_loss)
#     """
#     # TODO: Implement function
#     return None, None, None

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):

  # 将 4D tensors 转换为 2D, 每一行代表一个像素, 每一列代表一类
  logits = tf.reshape(nn_last_layer, (-1, num_classes), name="fcn_logits")
  correct_label_reshaped = tf.reshape(correct_label, (-1, num_classes))

  # 使用交叉熵计算预测与真实标签之间的差异
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=correct_label_reshaped[:])
  # 计算均值作为损失
  loss_op = tf.reduce_mean(cross_entropy, name="fcn_loss")

  # 模型应用这个操作来寻找权重/参数，以获得正确像素标签
  train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op, name="fcn_train_op")

  return logits, train_op, loss_op
tests.test_optimize(optimize)


# def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
#              correct_label, keep_prob, learning_rate):
#     """
#     Train neural network and print out the loss during training.
#     :param sess: TF Session
#     :param epochs: Number of epochs
#     :param batch_size: Batch size
#     :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
#     :param train_op: TF Operation to train the neural network
#     :param cross_entropy_loss: TF Tensor for the amount of loss
#     :param input_image: TF Placeholder for input images
#     :param correct_label: TF Placeholder for label images
#     :param keep_prob: TF Placeholder for dropout keep probability
#     :param learning_rate: TF Placeholder for learning rate
#     """
#     # TODO: Implement function
#     pass

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op,
             cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):

  keep_prob_value = 0.5
  learning_rate_value = 0.001
  for epoch in range(epochs):
      # 创建函数获取 batches
      total_loss = 0
      for X_batch, gt_batch in get_batches_fn(batch_size):

          loss, _ = sess.run([cross_entropy_loss, train_op],
          feed_dict={input_image: X_batch, correct_label: gt_batch,
          keep_prob: keep_prob_value, learning_rate:learning_rate_value})

          total_loss += loss;

      print("EPOCH {} ...".format(epoch + 1))
      print("Loss = {:.3f}".format(total_loss))
      print()
tests.test_train_nn(train_nn)


# def run():
#     num_classes = 2
#     image_shape = (160, 576)  # KITTI dataset uses 160x576 images
#     data_dir = './data'
#     runs_dir = './runs'
#     tests.test_for_kitti_dataset(data_dir)

#     # Download pretrained vgg model
#     helper.maybe_download_pretrained_vgg(data_dir)

#     # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
#     # You'll need a GPU with at least 10 teraFLOPS to train on.
#     #  https://www.cityscapes-dataset.com/

#     with tf.Session() as sess:
#         # Path to vgg model
#         vgg_path = os.path.join(data_dir, 'vgg')
#         # Create function to get batches
#         get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

#         # OPTIONAL: Augment Images for better results
#         #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

#         # TODO: Build NN using load_vgg, layers, and optimize function

#         # TODO: Train NN using the train_nn function

#         # TODO: Save inference data using helper.save_inference_samples
#         #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

#         # OPTIONAL: Apply the trained model to a video

def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    EPOCHS = 40
    BATCH_SIZE = 16
    learning_rate = 0.001
    vgg_path = os.path.join(data_dir, 'vgg')
    DROPOUT = 0.75

    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    # 下载预训练 vgg 模型
    helper.maybe_download_pretrained_vgg(data_dir)

    # 获取 batch 函数
    get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

    with tf.Session() as session:

        # 从 vgg 架构中返回输入、三个层和keep probability
        image_input, keep_prob, layer3, layer4, layer7 = load_vgg(session, vgg_path)

        # 完整网络架构
        model_output = layers(layer3, layer4, layer7, num_classes)

        # 返回输出预测，训练操作和损失操作
        # - logits: 每一行代表一个像素, 每一列代表一类
        # - train_op: 获取正确参数，使得模型能够正确地标记像素
        # - cross_entropy_loss: 输出需要最小化的损失，更低的损失可以获得更高的准确率
        logits, train_op, cross_entropy_loss = optimize(model_output, correct_label, learning_rate, num_classes)

        # 初始化所有变量
        session.run(tf.global_variables_initializer())
        session.run(tf.local_variables_initializer())

        print("Model build successful, starting training")

        saver = tf.train.Saver()

        # 训练神经网络
        train_nn(session, EPOCHS, BATCH_SIZE, get_batches_fn, 
                    train_op, cross_entropy_loss, image_input,
                    correct_label, keep_prob, learning_rate)
        model_path = "/dta/lulx/CarND-Semantic-Segmentation/data/vgg/model.ckpt"
        saver.save(session, model_path)
        print('Save The Model Susscefully!!!')

        # 在测试图像上运行模型，保存每个输出图像（道路为绿色）
        saver.restore(session, "/dta/lulx/CarND-Semantic-Segmentation/data/vgg/model.ckpt")
        helper.save_inference_samples(runs_dir, data_dir, session, image_shape, logits, keep_prob, image_input)

        print("All done!")

if __name__ == '__main__':
    run()
