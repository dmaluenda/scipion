from __future__ import print_function

from six.moves import range
import numpy as np
import sys, os
#import tensorflow as tf 
import numpy as np 
import time
from os.path import expanduser
from pyworkflow.utils import Environ
from pyworkflow.em.packages.xmipp3.networkDef import main_network, get_available_gpus, NUM_FILTERS

import xmipp
import pyworkflow.em.metadata as md

def updateEnviron():
    """ Create the needed environment for TensorFlow programs. """
    environ = Environ(os.environ)
    if  os.environ['CUDA']:
        environ.update({'LD_LIBRARY_PATH': os.environ['CUDA_LIB']}, position=Environ.BEGIN)
        environ.update({'LD_LIBRARY_PATH': os.environ['CUDA_HOME']+"/extras/CUPTI/lib64"}, position=Environ.BEGIN)
updateEnviron()

import tensorflow as tf 

LEARNING_RATE= 1e-4
EPSILION=1e-8
DROPOUT_KEEP_PROB=0.5

BATCH_SIZE= 256
EVALUATE_AT= 1
CHECK_POINT_AT= 50


def printAndOverride(msg):
  print("\r%s"%(msg), end="")
    
class DeepTFSupervised(object):
  def __init__(self, rootPath):

    self.rootPath= rootPath
#    if not os.path.isdir(self.rootPath):
#      os.mkdir(self.rootPath)
    self.checkPointsNames= os.path.join(rootPath,"tfchkpoints")
#    if not os.path.isdir(self.checkPointsNames):
#      os.mkdir(self.checkPointsNames)
    self.checkPointsNames= os.path.join(self.checkPointsNames,"screening")

    self.logsSaveName= os.path.join(rootPath,"tflogs")
#    if not os.path.isdir(self.logsSaveName):
#      os.mkdir(self.logsSaveName)
        
    self.batch_size= BATCH_SIZE
    self.num_labels=2
    self.num_channels=None
    self.image_size=None
    
    # Tensorflow objects
    self.X= None
    self.Y= None
    self.saver = None
    self.session = None
    self.train_writer = None
    self.test_writer = None
    self.global_step= None
    self.loss= None
    self.optimizer= None
                                              
  def createNet(self, xdim, ydim, num_chan):

    print ("Creating net")
    ##############################################################
    # INTIALIZE INPUT PLACEHOLDERS
    ##############################################################

    self.num_channels= num_chan
    self.image_size= (xdim, ydim)
    num_labels= self.num_labels
    self.X= tf.placeholder( tf.float32, shape=(None, self.image_size[0], self.image_size[1], num_chan), name="X")
    self.Y= tf.placeholder(tf.float32, shape=(None, num_labels), name="Y")
    self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    
    ######################
    # NEURAL NET CREATION
    ######################
    self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    (self.y_pred, self.merged_summaries, 
      self.loss, self.accuracy)= main_network(x= self.X, labels= self.Y, num_labels= num_labels, 
                                              dropout_keep_prob= self.dropout_keep_prob)

    self.optimizer = tf.train.AdamOptimizer(learning_rate= LEARNING_RATE, epsilon= EPSILION).minimize(
                                                                          self.loss, global_step= self.global_step, 
                                                                          name="optimizer")

  
  def startSessionAndInitialize(self, numberOfThreads=8):
    '''
      numberOfThreads==None -> default (use GPU or all threads)
    '''
    print("Initializing tf session")
    
    save_dir, prefixName = os.path.split(self.checkPointsNames)
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)

    self.saver = tf.train.Saver()  
    if numberOfThreads is None:
      self.session = tf.Session()
    else:
      self.session= tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=numberOfThreads))

    try:
      print("Trying to restore last checkpoint from "+ os.path.abspath(save_dir))
      # Use TensorFlow to find the latest checkpoint - if any.
      last_chk_path = tf.train.latest_checkpoint(checkpoint_dir= save_dir)

      # Try and load the data in the checkpoint.
      self.saver.restore(self.session, save_path=last_chk_path)
      # If we get to this point, the checkpoint was successfully loaded.
      print("Restored checkpoint from:", last_chk_path)
    except Exception as e:
      print("last chk_point path:",last_chk_path)
      print(e)
      # If the above failed for some reason, simply
      # initialize all the variables for the TensorFlow graph.
      print("Failed to restore checkpoint. Initializing variables instead.")
      if( os.path.isdir(self.logsSaveName)):
        tf.gfile.DeleteRecursively(self.logsSaveName)
      os.makedirs(os.path.join(self.logsSaveName,"train"))
      os.makedirs(os.path.join(self.logsSaveName,"test"))
      self.session.run( tf.global_variables_initializer() )
      
    self.train_writer = tf.summary.FileWriter(os.path.join(self.logsSaveName,"train"), self.session.graph)
    self.test_writer = tf.summary.FileWriter(os.path.join(self.logsSaveName,"test"), self.session.graph)
    return self.session
  
  def close(self, saveModel= False):
    if saveModel:
      self.saver.save(self.session, save_path= self.checkPointsNames, global_step= self.global_step) 
      print("\nSaved checkpoint.")
    self.train_writer.close()
    self.test_writer.close()
    tf.reset_default_graph()
    self.session.close()


  def reset(self):

    self.train_writer.close()
    self.test_writer.close()
    self.session.close()
    tf.reset_default_graph()
    self.createNet()
    self.startSessionAndInitialize()
    
  def trainNet(self, posImages, negImages, numberOfBatches):

    
    ########################
    # TENSOR FLOW RUN
    ########################

    x_batchTrain= np.concatenate([posImages, negImages])
    labels_batchTrain= np.zeros( (posImages.shape[0] + negImages.shape[0],2) )
    labels_batchTrain[:posImages.shape[0],1]=1
    labels_batchTrain[posImages.shape[0]:,0]=1
    del posImages, negImages
    feed_dict_train= {self.X : x_batchTrain, self.Y: labels_batchTrain, 
                        self.dropout_keep_prob: DROPOUT_KEEP_PROB  }
    i_global, __, stepLoss,= self.session.run( [self.global_step, self.optimizer, self.loss],
                                                 feed_dict=feed_dict_train )
      

#    printAndOverride("iterNum %d/%d trainLoss: %3.4f"%((i_global), numberOfBatches, stepLoss) )
    if i_global % EVALUATE_AT ==0:
      print("iterNum %d/%d trainLoss: %3.4f"%((i_global), numberOfBatches, stepLoss) )
      sys.stdout.flush()
    if (i_global + 1 ) % CHECK_POINT_AT==0:
      self.saver.save(self.session, save_path= self.checkPointsNames, global_step= self.global_step) 
      print("\nSaved checkpoint.")   

#    self.testPerformance( iterNum, trainBatchData, useTestData=True)
    
    
  def accuracy_score(self, labels, predictions ):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])    

  def testPerformance(self, stepNum, trainDataBatch, useTestData= False):

    batch_x, batch_y= trainDataBatch
    feed_dict_train= {self.X : batch_x, self.Y: batch_y, self.dropout_keep_prob: 1.0  }
    c_e_train, y_pred_train, merged = self.session.run( [self.loss, self.y_pred, self.merged_summaries], 
                                                 feed_dict = feed_dict_train )
    train_accuracy=  self.accuracy_score(batch_y, y_pred_train)
    self.train_writer.add_summary(merged, stepNum)

  def predictNet(self, images):

    feed_dict_train= {self.X : images, 
                        self.dropout_keep_prob: 1  }
    y_pred= self.session.run( self.y_pred,
                              feed_dict=feed_dict_train )
    return y_pred[:,1]                                                 

class DataManager(object):
  def __init__(self, posImagesFname, posImagesObject,  negImagesFname, negImagesObject):
    
    self.mdFalse=None
    self.nFalse=0
    self.mdTrue  = md.MetaData() 
    self.mdTrue.randomize( md.MetaData(self._getExtraPath(posImagesFname)) )
    
    xdim, ydim, _     = posImagesObject.getDim()
    self.shape= (xdim,ydim,1)
    self.nTrue= posImagesObject.getSize()    
    self.batchSize= self.estimateBatchSize(xdim,ydim)
    self.splitPoint= self.batchSize//2

    self.batchStack   = np.zeros((self.batchSize, xdim,ydim,1))
    self.batchLabels = np.zeros((self.batchSize, 2))
    if not negImagesFname is None:
      self.mdFalse  = md.MetaData() 
      self.mdFalse.randomize( md.MetaData(self._getExtraPath(negImagesFname)) )
      xdim_1, ydim_1, _ = negImagesObject.getDim()
      self.nFalse= negImagesObject.getSize()    
      assert xdim==xdim_1 and ydim==ydim_1
  
  def estimateBatchSize(self,xdim,ydim):
    if os.environ['CUDA']:
      from tensorflow.python.client import device_lib
      local_device_protos = device_lib.list_local_devices()
      gpu_mem_list=[x.memory_limit for x in local_device_protos if x.device_type == 'GPU']
      nBytes= int(0.2* gpu_mem_list[0]) #20% gpu memory for data. Rest for model
      maxNFilters= np.max(NUM_FILTERS)
      batchSize= nBytes // (xdim*ydim*4*maxNFilters*3) # 4 is float size. 3 for training requirements
      return batchSize
    else:
      return 4096

  def getRandomBatch(self):
    batchSize = self.batchSize
    splitPoint = self.splitPoint
    batchStack   = self.bachStack
    batchLabels  = np.zeros((batchSize, 2))
    mdTrue= self.mdTrue
    mdFalse= self.mdFalse

    I = xmipp.Image()
    n = 0
    for id in mdTrue:
      fnImage = mdTrue.getValue(md.MDL_IMAGE, id)
      I.read(fnImage)
      batchStack[n,...]= np.expand_dims(I.getData(),-1)
      batchLabels[n, 1]= 1
      n+=1
      if n>=splitPoint:
          break
    for id in mdFalse:
      fnImage = mdFalse.getValue(md.MDL_IMAGE, id)
      I.read(fnImage)
      batchStack[n,...]= np.expand_dims(I.getData(),-1)
      batchLabels[n, 0]= 1
      n+=1
      if n>=splitPoint:
          break          
    return batchStack, batchLabels
