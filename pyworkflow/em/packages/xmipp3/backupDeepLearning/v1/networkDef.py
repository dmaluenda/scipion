import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.python.client import device_lib

'''
NET DEFINITION

SMALL INCEPTION

'''

NUM_FILTERS=[16, 32, 48, 64]
KERNEL_SIZE=[7, 5, None, None]
MAX_POOL=[2,2]
N_LAYERS_CONV=len(NUM_FILTERS)

FC_SIZE = [2048, 2048]
NUM_FC= len(FC_SIZE)

DTYPE = tf.float32


def parametric_relu(_x, name="prelu"):
  with tf.variable_scope('prelu'+str(name) ) as scope:
    alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                         initializer=tf.constant_initializer(0.0),
                          dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

  return pos + neg
  
relu_function= tf.nn.relu
##relu_function= parametric_relu  

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
    
def _weight_variable(name, shape):
#  return tf.get_variable(name, shape, DTYPE, tf.truncated_normal_initializer(stddev=0.1))
  return tf.get_variable(name, shape, DTYPE, initializer=tf.contrib.layers.xavier_initializer() )

def _bias_variable(name, shape):
  return tf.get_variable(name, shape, DTYPE, tf.constant_initializer(0.0, dtype=DTYPE))


def createFullyConLayer(layerNum, prev_layer_flat, n_neu, dropout_keep_prob=None):
  with tf.variable_scope('fullyCon'+str(layerNum) ) as scope:
    dim = np.prod(prev_layer_flat.get_shape().as_list()[1:])
    weights = _weight_variable('weights', [dim, n_neu])
    biases = _bias_variable('biases', [n_neu])
    fc1 = relu_function(tf.matmul(prev_layer_flat, weights) + biases, name=scope.name)
  if not dropout_keep_prob is None:
    with tf.name_scope( 'dropoutFc'+str(layerNum) ):
      prev_layer = tf.nn.dropout(fc1, dropout_keep_prob)
  else:
    prev_layer=fc1
  return prev_layer
  
def createLastLayer( prev_layer_flat, n_labels):
  with tf.variable_scope('softmax_layer' ) as scope:
    dim = np.prod(prev_layer_flat.get_shape().as_list()[1:])
    weights = _weight_variable('weights', [dim, n_labels])
    biases = _bias_variable('biases', [n_labels])
    logits = tf.matmul(prev_layer_flat, weights) + biases
    y_pred = tf.nn.softmax(logits)
  return logits, y_pred
   
def conv2d_s1(prev_layer, W):
  with tf.variable_scope('conv2d') as scope:
    return tf.nn.conv2d(prev_layer,W,strides=[1,1,1,1],padding='SAME')
 
def max_pool_3x3_s1(prev_layer):
  with tf.variable_scope('max_pool3x3' ) as scope:
    return tf.nn.max_pool(prev_layer,ksize=[1,3,3,1], strides=[1,1,1,1],padding='SAME')

def avg_pool_5x5_s1(prev_layer):
  with tf.variable_scope('avg_pool5x5' ) as scope:
    return tf.nn.avg_pool(prev_layer,ksize=[1,5,5,1], strides=[1,2,2,1],padding='SAME')


def createOneConvLayer(layerNum, prev_layer_out, outChanNum, kernerSize=3, poolingStep=1):

  in_filters= prev_layer_out.get_shape().as_list()[-1]
  with tf.variable_scope('conv'+str(layerNum) ) as scope:
    kernel = _weight_variable('weights', [kernerSize, kernerSize, in_filters, outChanNum])
    conv = tf.nn.conv2d(prev_layer_out, kernel, [1, 1, 1, 1], padding='SAME')
    biases = _bias_variable('biases', [outChanNum])
    bias = tf.nn.bias_add(conv, biases)
    conv_out = relu_function(bias, name="relu")
    if poolingStep>1:
      with tf.variable_scope('maxpool') as scope:
        conv_out = tf.nn.max_pool( conv_out, ksize=[1, kernerSize, kernerSize, 1], 
                                     strides=[1, poolingStep, poolingStep, 1], padding='SAME')
    return conv_out
        
def createInceptionModule(layerNum, prev_layer, outChanNum):
  with tf.variable_scope('inceptionModule_'+str(layerNum) ) as scope:
    prev_layer_shapes= prev_layer.get_shape().as_list()
    
    with tf.variable_scope('conv_1x1' ) as scope:
      W_conv_1x1_1= _weight_variable("conv_1x1_1weight", shape= [1,1, prev_layer_shapes[-1], outChanNum] )
      b_conv_1x1_1= _bias_variable("conv_1x1_1bias", shape= [outChanNum] )
      conv_1x1_1 = conv2d_s1(prev_layer,W_conv_1x1_1) + b_conv_1x1_1
      
    with tf.variable_scope('conv_3x3' ) as scope:
      with tf.variable_scope('conv_1x1' ) as scope:
        W_conv_1x1_2= _weight_variable("conv_1x1_2weight", shape= [1,1, prev_layer_shapes[-1], outChanNum//2] )
        b_conv_1x1_2= _bias_variable("conv_1x1_2bias", shape= [outChanNum//2] )
        conv_1x1_2 = relu_function(conv2d_s1(prev_layer,W_conv_1x1_2) + b_conv_1x1_2)
        
      with tf.variable_scope('conv_3x3' ) as scope:
        W_conv_3x3= _weight_variable("conv_3x3_weight", shape= [3,3, outChanNum//2, outChanNum] )
        b_conv_3x3= _bias_variable("conv_3x3_bias", shape= [outChanNum] )  
        conv_3x3 = conv2d_s1(conv_1x1_2, W_conv_3x3)+ b_conv_3x3
      
    with tf.variable_scope('conv_5x5' ) as scope:
      with tf.variable_scope('conv_1x1' ) as scope:    
        W_conv_1x1_3= _weight_variable("conv_1x1_3weight", shape= [1,1, prev_layer_shapes[-1], outChanNum//2] )
        b_conv_1x1_3= _bias_variable("conv_1x1_3bias", shape= [outChanNum//2] ) 
        conv_1x1_3 = relu_function(conv2d_s1(prev_layer, W_conv_1x1_3)+b_conv_1x1_3)
        
      with tf.variable_scope('conv_5x5' ) as scope:      
        W_conv_5x5= _weight_variable("conv_5x5_weight", shape= [5,5, outChanNum//2, outChanNum] )
        b_conv_5x5= _bias_variable("conv_5x5_bias", shape= [outChanNum] )  
        conv_5x5 = conv2d_s1(conv_1x1_3, W_conv_5x5)+ b_conv_5x5

    with tf.variable_scope('max_pool_3x3' ) as scope:
      with tf.variable_scope('max_pool_3x3' ) as scope:    
        maxpool1 = max_pool_3x3_s1(prev_layer)
      with tf.variable_scope('conv_1x1' ) as scope:          
        W_conv_1x1_4= _weight_variable("conv_1x1_4weight", shape= [1,1, prev_layer_shapes[-1], outChanNum] )
        b_conv_1x1_4= _bias_variable("conv_1x1_4bias", shape= [outChanNum] )
        conv_1x1_4 = conv2d_s1(maxpool1, W_conv_1x1_4)+b_conv_1x1_4
      
    output= relu_function(tf.concat( values=[conv_1x1_1, conv_3x3, conv_5x5, conv_1x1_4], axis=3 ) )
  return output

def main_network(x, labels, num_labels, dropout_keep_prob):
  '''
    4D-tensor r_cubes,  [imageNumber, sizeAxis1, sizeAxis3, nChann]
  '''
  
#  gpu_list= get_available_gpus()


  
  prev_layer= x
 
  prev_layer= createOneConvLayer(0, prev_layer, NUM_FILTERS[0], kernerSize=KERNEL_SIZE[0], poolingStep=MAX_POOL[0])
  prev_layer= tf.nn.local_response_normalization(prev_layer)
  prev_layer= createOneConvLayer(1, prev_layer, NUM_FILTERS[1], kernerSize=KERNEL_SIZE[1], poolingStep=MAX_POOL[1])
  
  prev_layer= createInceptionModule(2, prev_layer, NUM_FILTERS[2] )
  prev_layer= createInceptionModule(3, prev_layer, NUM_FILTERS[3] )
  prev_layer= avg_pool_5x5_s1(prev_layer)
  prev_layer_flat = tf.reshape( prev_layer , [-1, np.prod(prev_layer.get_shape().as_list()[1:])] )
  prev_layer_flat= createFullyConLayer(6, prev_layer_flat, FC_SIZE[0], dropout_keep_prob= None)
  prev_layer_flat= createFullyConLayer(7, prev_layer_flat, FC_SIZE[1], dropout_keep_prob= dropout_keep_prob)
  logits, y_pred = createLastLayer( prev_layer_flat, num_labels)

  cross_entropy= tf.reduce_mean(
                              tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= labels) )
  
  with tf.name_scope('PERFORMANCE'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(labels, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      
    tf.summary.scalar( 'cross_entropy', cross_entropy )
    tf.summary.scalar( 'accuracy', accuracy )
    mergedSummaries= tf.summary.merge_all()

  return y_pred, mergedSummaries, cross_entropy, accuracy

