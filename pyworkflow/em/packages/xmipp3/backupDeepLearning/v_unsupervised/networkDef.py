import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib
import tflearn
import math
'''
Pruebas de particle screening semisupervised
'''
DROPOUT_KEEP_PROB=0.5

def lrelu(x, leak=0.2, name="lrelu"):
  """Leaky rectifier.
  Parameters
  ----------
  x : Tensor
      The tensor to apply the nonlinearity to.
  leak : float, optional
      Leakage parameter.
  name : str, optional
      Variable scope to use.
  Returns
  -------
  x : Tensor
      Output of the nonlinearity.
  """
  with tf.variable_scope(name):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)

def corrupt(x, corrFrac=0.1):
  """Take an input tensor and add uniform masking.
  Parameters
  ----------
  x : Tensor/Placeholder
      Input to corrupt.
  Returns
  -------
  x_corrupted : Tensor
      corrFrac %% of values corrupted.
  """
  
  x_shape= x.get_shape().as_list()[1:]
  probMask= np.array((corrFrac)*np.ones(x_shape), dtype= np.float32) 
  corrupMask = tf.where(tf.random_uniform(x_shape)- probMask > 0.0, tf.ones(x_shape), tf.zeros(x_shape))
  return tf.multiply(x, corrupMask)
 
DTYPE= tf.float32                                               
def _weight_variable(name, shape):
#  return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))
  return tf.get_variable(name, shape, DTYPE, initializer=tf.contrib.layers.xavier_initializer() )

def _bias_variable(name, shape):
  return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.0, dtype=DTYPE))
       
def main_network(x, labels, num_labels, learningRate, globalStep):
  '''
    4D-tensor x,  [imageNumber, sizeAxis1, sizeAxis3, nChann]
  '''
  
#  gpu_list= get_available_gpus()

  n_filters=[1, 16, 18, 24]
  filter_sizes=[5, 5, 3, 3]
  corruption=True
  
  img_aug= tflearn.data_augmentation.ImageAugmentation()
  img_aug.add_random_flip_leftright()
  img_aug.add_random_flip_updown()

  x_augm = tflearn.layers.core.input_data(placeholder=x, data_augmentation=img_aug)
  current_input = x_augm
  
  encoder = []
  shapes = []
  for layer_i, n_output in enumerate(n_filters[1:]):
      n_input = current_input.get_shape().as_list()[3]
      shapes.append(current_input.get_shape().as_list())
      W = tf.Variable(
          tf.random_uniform([
              filter_sizes[layer_i],
              filter_sizes[layer_i],
              n_input, n_output],
              -1.0 / math.sqrt(n_input),
              1.0 / math.sqrt(n_input)))
      b = tf.Variable(tf.zeros([n_output]))
      encoder.append(W)
      output = lrelu(
          tf.add(tf.nn.conv2d(
              current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
      print(output)              
      current_input = output
      
  z = current_input
  representationLayer= tflearn.layers.core.flatten(z)
  encoder.reverse()
  shapes.reverse()

  # Build the decoder using the same weights
  for layer_i, shape in enumerate(shapes):
      W = encoder[layer_i]
      b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
      output = lrelu(tf.add(
          tf.nn.conv2d_transpose(
              current_input, W,
              tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
              strides=[1, 2, 2, 1], padding='SAME'), b))
      current_input = output
  y_pred = current_input
  
  recons_error= tf.squeeze(tf.reduce_sum(tf.reduce_sum( tf.square(y_pred - x) , axis=1), axis=1))
  cost= tf.reduce_mean(recons_error)
  
  if not globalStep is None:
    learningRate= 1e-3
    optimizer = tf.train.AdamOptimizer(learning_rate= learningRate, epsilon= 1e-8)
#    optimizer = tf.train.GradientDescentOptimizer(0.01)
    optimizer= optimizer.minimize(recons_error, global_step= globalStep)

  tf.summary.scalar( 'loss', cost )
#  tf.summary.image('input', x)
#  tf.summary.image('compression', z)
#  tf.summary.image('output', y_pred)
  mergedSummaries= tf.summary.merge_all()    

  return representationLayer, mergedSummaries, optimizer, cost



'''
2017-05-09 19:53:18.340286: F tensorflow/core/kernels/conv_ops.cc:659] Check failed: stream->parent()->GetConvolveAlgorithms(&algorithms) 
'''
