
#######################################################
#
# Setup CNN, dCNN, and FC layers
# Code adapted from:
#  https://github.com/tensorflow/models/blob/master/real_nvp/real_nvp_utils.py
#  https://github.com/machrisaa/tensorflow-vgg
#  And from Tensorflow CIFAR10 example.
#
#######################################################


import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_float('weight_decay', 0.0005,
                          """ """)

tf.app.flags.DEFINE_float('weights_init', .1,
                            """initial weights for fc layers""")

def _activation_summary(x):
  """Helper to create summaries for activations.

  Creates a summary that provides a histogram of activations.
  Creates a summary that measure the sparsity of activations.

  Args:
    x: Tensor
  Returns:
    nothing
  """
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

  # used by cifar10 and inception in tensorflow for multi-GPU systems that have no P2P.
  # But Titan X's have DMA P2P, so change to /gpu:0
  #https://github.com/tensorflow/tensorflow/issues/4881
def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.

  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable

  Returns:
    Variable Tensor
  """
#  with tf.device('/cpu:0'):
  with tf.device('/gpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.

  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape,
                         tf.truncated_normal_initializer(stddev=stddev))
  if wd:
    weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
    weight_decay.set_shape([])
    tf.add_to_collection('losses', weight_decay)
  return var

def cnn2d_layer(inputs, kernel, stride, features, idx, linear = False):
  # below scope means this layer is shared for all calls unless idx is different.
  with tf.variable_scope('{0}_cnn'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3] # rgb

    weights = _variable_with_weight_decay('weights', shape=[kernel,kernel,input_channels,features],stddev=0.01, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('biases',[features],tf.constant_initializer(0.01))

    cnn = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
    cnn_biased = tf.nn.bias_add(cnn, biases)
    if linear:
      return cnn_biased
    cnn_rect = tf.nn.elu(cnn_biased,name='{0}_cnn'.format(idx))
    return cnn_rect

def dcnn2d_layer(inputs, kernel, stride, features, idx, linear = False):
  # below scope means this layer is shared for all calls unless idx is different.
  with tf.variable_scope('{0}_dcnn'.format(idx)) as scope:
    input_channels = inputs.get_shape()[3] # rgb
    
    weights = _variable_with_weight_decay('deweights', shape=[kernel,kernel,features,input_channels], stddev=0.01, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('debiases',[features],tf.constant_initializer(0.01))
    batch_size = tf.shape(inputs)[0]
    output_shape = tf.pack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, features]) 
    dcnn = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
    dcnn_biased = tf.nn.bias_add(dcnn, biases)
    if linear:
      return dcnn_biased
    dcnn_rect = tf.nn.elu(dcnn_biased,name='{0}_dcnn'.format(idx))
    return dcnn_rect
     

def fc_layer(inputs, hiddens, idx, flat = False, linear = False):
  with tf.variable_scope('{0}_fc'.format(idx)) as scope:
    input_shape = inputs.get_shape().as_list()
    if flat:
      dim = input_shape[1]*input_shape[2]*input_shape[3]
      inputs_processed = tf.reshape(inputs, [-1,dim])
    else:
      dim = input_shape[1]
      inputs_processed = inputs
    
    weights = _variable_with_weight_decay('fcweights', shape=[dim,hiddens],stddev=FLAGS.weights_init, wd=FLAGS.weight_decay)
    biases = _variable_on_cpu('fcbiases', [hiddens], tf.constant_initializer(FLAGS.weights_init))
    if linear:
      return tf.add(tf.matmul(inputs_processed,weights),biases,name=str(idx)+'_fc')
  
    ip = tf.add(tf.matmul(inputs_processed,weights),biases)
    return tf.nn.elu(ip,name=str(idx)+'_fc')

