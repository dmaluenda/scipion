'''
rsanchez@cnb.csic.es

'''
import numpy as np
import tensorflow as tf
import tflearn
from scipy.ndimage.filters import gaussian_filter

DTYPE = tf.float32
MODEL = "VarSize"  #Can be ConvNet or ResNet or Trial or "VarSize"

def blurInput(batch):
  for i in range(len(batch)):
    batch[i] = gaussian_filter(batch[i], sigma=0.5)
  return batch

def add_blur(batch):
  newBatch= np.zeros( (batch.shape[:-1], batch.shape[-1]+1) )
  print( newBatch.shape)
  for i in range(len(batch)):
    newBatch[i] = batch[i, ...,:-1]
    newBatch[i,...,-1] = gaussian_filter(batch[i], sigma=0.5)
  return newBatch

def myResNetBlock(network, nPairsOfLayers, nkernels, kernelSize, reductionStep=1):
  with tf.variable_scope("myResNetBlock_%d_%d_Reduc_%d"%(nkernels, kernelSize, reductionStep) ) as scope:
    input_layer= network

    if reductionStep>1:
      network = tflearn.layers.conv.conv_2d(network, nkernels, kernelSize, strides= reductionStep, activation='relu')
      network = tflearn.layers.normalization.batch_normalization(network)
      network = tflearn.activations.relu( network )
      network = tflearn.layers.conv.conv_2d(network, nkernels, kernelSize, strides=1, activation='linear')
      network = tflearn.layers.normalization.batch_normalization(network)

      input_layer = tflearn.layers.conv.conv_2d(input_layer, nkernels, 1, strides=  reductionStep)

      input_layer = tflearn.layers.normalization.batch_normalization( input_layer)
      network = tflearn.activations.relu( network + input_layer )
      input_layer= network
      nPairsOfLayers-= 1
      assert nPairsOfLayers >=0

    for i in range(nPairsOfLayers):
#      print(input_layer, network)
      network = tflearn.layers.conv.conv_2d(network, nkernels, kernelSize, strides=1, activation='linear')
      network = tflearn.layers.normalization.batch_normalization(network)
      network = tflearn.activations.relu( network)
      network = tflearn.layers.conv.conv_2d(network, nkernels, kernelSize, strides=1, activation='linear')
      network = tflearn.layers.normalization.batch_normalization(network)
      network = tflearn.activations.relu( network + input_layer)
      input_layer= network
  return network


import scipy.stats as st
def gkern(kernlen=5, nsig=1):
  """Returns a 2D Gaussian kernel array."""
  interval = (2*nsig+1.)/(kernlen)
  x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
  kern1d = np.diff(st.norm.cdf(x))
  kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
  kernel = kernel_raw/kernel_raw.sum()
  return kernel.astype(np.float32)

def main_network(x, labels, num_labels, learningRate, globalStep, nData):
  '''
    4D-tensor x,  [imageNumber, height, width, nChanns]
    2D-tensor labels, [imageNumber, 2]
    float learningRate
    tf.Variable globalStep
    int nData Expected data size (used to select model size)
  '''

  x =  tf.image.resize_bicubic(x, size=[128,128])
  network =  x
  DROPOUT_KEEP_PROB= 0.5
  L2_CONST= 1e-5

  if MODEL=="Trial":
    network = tflearn.layers.conv.conv_2d(network, 8, 30, strides=1, activation='relu')
    network = tflearn.layers.conv.conv_2d(network, 8, 30, strides=1, activation='linear')
    network = tflearn.layers.normalization.batch_normalization(network)
    network = tflearn.activations.relu(network)
    network = tflearn.layers.conv.avg_pool_2d(network, kernel_size=4, strides=2)

    network = tflearn.layers.core.fully_connected( network, 2**9, activation='relu', regularizer='L2', weight_decay= L2_CONST)
    network = tflearn.layers.core.dropout(network, DROPOUT_KEEP_PROB)
    logits =  tflearn.layers.core.fully_connected( network, num_labels, activation='linear')
    y_pred = tf.nn.softmax(logits)
    current_lr = tf.train.exponential_decay(learningRate, globalStep, 500, 0.90, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate= learningRate)
  elif MODEL=="VarSize":
    if nData<1500:
      modelDepth=1
    elif 1500<=nData<20000:
      modelDepth=2
    else:
      modelDepth=3
    print("Model depth: %d"%modelDepth)
    for i in range(1, modelDepth+1):
      network = tflearn.layers.conv.conv_2d(network, 2**(2+i), 30//2**i, strides=1, activation='relu',regularizer='L2', weight_decay= L2_CONST)
      network = tflearn.layers.conv.conv_2d(network, 2**(2+i), 30//2**i, strides=1, activation='linear')
      network = tflearn.layers.normalization.batch_normalization(network)
      network = tflearn.activations.relu(network)
      if i!=modelDepth:
    	    network = tflearn.layers.conv.max_pool_2d(network, kernel_size=7-(2*(i-1)), strides=2)

    network = tflearn.layers.conv.avg_pool_2d(network, kernel_size=4, strides=2)
    network = tflearn.layers.core.fully_connected( network, 2**9, activation='relu', regularizer='L2', weight_decay= L2_CONST)
    network = tflearn.layers.core.dropout(network, DROPOUT_KEEP_PROB)
    logits =  tflearn.layers.core.fully_connected( network, num_labels, activation='linear')
    y_pred = tf.nn.softmax(logits)
    current_lr = tf.train.exponential_decay(learningRate, globalStep, 200*modelDepth, 0.90, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate= learningRate)
   
  elif MODEL=="ResNet":
    network= myResNetBlock(network, 1, 32, 15, reductionStep=2)
    network= myResNetBlock(network, 2, 32, 15, reductionStep=1)
    network= myResNetBlock(network, 1, 64, 11, reductionStep=2)
    network= myResNetBlock(network, 2, 64, 11, reductionStep=1)
    network= myResNetBlock(network, 1, 128, 9, reductionStep=2)
    network= myResNetBlock(network, 2, 128, 9, reductionStep=1)
    network= myResNetBlock(network, 1, 256, 7, reductionStep=2)
    network= myResNetBlock(network, 2, 256, 7, reductionStep=1)

#    network = tflearn.layers.conv.global_avg_pool(network)

    network = tflearn.layers.core.fully_connected( network, 2**10, activation='relu', regularizer='L2', weight_decay= L2_CONST)
    network = tflearn.layers.core.dropout(network, DROPOUT_KEEP_PROB)
    logits =  tflearn.layers.core.fully_connected( network, num_labels, activation='linear')
    y_pred = tf.nn.softmax(logits)
    current_lr = tf.train.exponential_decay(learningRate, globalStep, 5000, 0.90, staircase=True)
#    optimizer = tf.train.AdamOptimizer(learning_rate= current_lr)
    optimizer = tf.train.MomentumOptimizer(learning_rate= current_lr, momentum=0.9, use_nesterov=False)

  elif MODEL=="ConvNet":

    network = tflearn.layers.conv.conv_2d(network, 32, 15, strides=1, activation='relu')
    network = tflearn.layers.conv.conv_2d(network, 32, 15, strides=1, activation='linear')
    network = tflearn.layers.normalization.batch_normalization(network)
    network = tflearn.activations.relu(network)
    network = tflearn.layers.conv.max_pool_2d(network, kernel_size=7, strides=4)

    network = tflearn.layers.conv.conv_2d(network, 64, 11, strides=1, activation='relu')
    network = tflearn.layers.conv.conv_2d(network, 64, 11, strides=1, activation='linear')
    network = tflearn.layers.normalization.batch_normalization(network)
    network = tflearn.activations.relu(network)
    network = tflearn.layers.conv.max_pool_2d(network, kernel_size=5, strides=3)

    network = tflearn.layers.conv.conv_2d(network, 128, 9, strides=1, activation='linear')
    network = tflearn.layers.normalization.batch_normalization(network)
    network = tflearn.activations.relu(network)
    network = tflearn.layers.conv.avg_pool_2d(network, kernel_size=4, strides=2)

    network = tflearn.layers.core.fully_connected( network, 2**10, activation='relu', regularizer='L2', weight_decay= L2_CONST)
    network = tflearn.layers.core.dropout(network, DROPOUT_KEEP_PROB)
    logits =  tflearn.layers.core.fully_connected( network, num_labels, activation='linear')
    y_pred = tf.nn.softmax(logits)
    current_lr = tf.train.exponential_decay(learningRate, globalStep, 5000, 0.90, staircase=True)
#    current_lr = learningRate
    optimizer = tf.train.AdamOptimizer(learning_rate= learningRate)


  reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
  cross_entropy= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= labels) + sum(reg_losses) )
  
  optimizer= optimizer.minimize(cross_entropy, global_step= globalStep)
  
  with tf.name_scope('PERFORMANCE'):
    with tf.name_scope('correct_prediction'):
      correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(labels, 1))
    with tf.name_scope('accuracy'):
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      
    tf.summary.scalar( 'cross_entropy', cross_entropy )
    tf.summary.scalar( 'accuracy', accuracy )
#    tf.summary.histogram('histogram', prev_layer)
#    tf.summary.image( '1_images', prev_layer, max_outputs= 3)

#    tf.summary.image("3_kernel_0",  tf.transpose (w_0, [3, 0, 1, 2]), max_outputs= NUM_FILTERS[0]) 
#    for kerNum in range( NUM_FILTERS[0]):
#      tf.summary.image("2_activations_0", tf.expand_dims(prev_layer0[:,:,:,kerNum], 3), max_outputs= 3)
#    for kerNum in range( NUM_FILTERS[1]):
#      tf.summary.image("3_activations_0", tf.expand_dims(prev_layer1[:,:,:,kerNum], 3), max_outputs= 3)            
#    for kerNum in range( NUM_FILTERS[2]):
#      tf.summary.image("3_activations_0", tf.expand_dims(prev_layer2[:,:,:,kerNum], 3), max_outputs= 3)
    mergedSummaries= tf.summary.merge_all()

  return y_pred, mergedSummaries, optimizer, cross_entropy, accuracy

