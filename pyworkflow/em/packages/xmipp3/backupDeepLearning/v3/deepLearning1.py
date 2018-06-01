from __future__ import print_function

from six.moves import range
import numpy as np
import sys, os
import tensorflow as tf 
import numpy as np 
from math import ceil

import time
from os.path import expanduser
from pyworkflow.utils import Environ
from pyworkflow.em.packages.xmipp3.networkDef import main_network
from sklearn.metrics import roc_auc_score, accuracy_score
import xmipp
import pyworkflow.em.metadata as md
import tflearn

def updateEnviron():
    """ Create the needed environment for TensorFlow programs. """
    environ = Environ(os.environ)
    if  os.environ['CUDA']:
        environ.update({'LD_LIBRARY_PATH': os.environ['CUDA_LIB']}, position=Environ.BEGIN)
        environ.update({'LD_LIBRARY_PATH': os.environ['CUDA_HOME']+"/extras/CUPTI/lib64"}, position=Environ.BEGIN)
updateEnviron()

import tensorflow as tf 

LEARNING_RATE= 1e-4
EPSILION=1e-8 # Para el modelo original
#EPSILION=1e-3

BATCH_SIZE= 256
EVALUATE_AT= 10
CHECK_POINT_AT= 50


def printAndOverride(msg):
  print("\r%s"%(msg), end="")
    
class DeepTFSupervised(object):
  def __init__(self, rootPath, learningRate=LEARNING_RATE):
    self.lRate= learningRate
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
    print ("Creating net. Learning rate %f"%self.lRate)
    ##############################################################
    # INTIALIZE INPUT PLACEHOLDERS
    ##############################################################

    self.num_channels= num_chan
    self.image_size= (xdim, ydim)
    num_labels= self.num_labels
    tflearn.config.init_training_mode()
    self.X= tf.placeholder( tf.float32, shape=(None, self.image_size[0], self.image_size[1], num_chan), name="X")
    self.Y= tf.placeholder(tf.float32, shape=(None, num_labels), name="Y")
    
    ######################
    # NEURAL NET CREATION
    ######################
    self.global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    (self.y_pred, self.merged_summaries, self.optimizer,
     self.loss, self.accuracy)= main_network(self.X,self.Y, num_labels, learningRate= self.lRate, globalStep= self.global_step)
  
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
    
#  def trainNet(self, numberOfBatches, dataManagerTrain, dataManagerTest=None):
#
#    
#    ########################
#    # TENSOR FLOW RUN
#    ########################
#    print("Training net for %d batches of size %d"%(numberOfBatches,batchSize))
#
#    x_batchTrain= np.concatenate([posImages, negImages])
#    labels_batchTrain= np.zeros( (posImages.shape[0] + negImages.shape[0],2) )
#    labels_batchTrain[:posImages.shape[0],1]=1
#    labels_batchTrain[posImages.shape[0]:,0]=1
#    del posImages, negImages
#    feed_dict_train= {self.X : x_batchTrain, self.Y: labels_batchTrain }
#    tflearn.is_training(True, session=self.session)
#    i_global, __, stepLoss,= self.session.run( [self.global_step, self.optimizer, self.loss],
#                                                 feed_dict=feed_dict_train )
#      
#
##    printAndOverride("iterNum %d/%d trainLoss: %3.4f"%((i_global), numberOfBatches, stepLoss) )
#    if i_global % EVALUATE_AT ==0:
#      print("iterNum %d/%d trainLoss: %3.4f"%((i_global), numberOfBatches, stepLoss) )
#      sys.stdout.flush()
#    if (i_global + 1 ) % CHECK_POINT_AT==0:
#      self.saver.save(self.session, save_path= self.checkPointsNames, global_step= self.global_step) 
#      print("\nSaved checkpoint.")   
#
##    self.testPerformance( iterNum, trainBatchData, useTestData=True)

  def trainNet(self, numberOfBatches, dataManagerTrain, dataManagerTest=None):

    
    ########################
    # TENSOR FLOW RUN
    ########################
    trainDataBatch= dataManagerTrain.getRandomBatch()
    x_batchTrain, labels_batchTrain= trainDataBatch
    feed_dict_train= {self.X : x_batchTrain, self.Y: labels_batchTrain}
    tflearn.is_training(False, session=self.session)
    i_global,stepLoss= self.session.run( [self.global_step, self.loss], feed_dict= feed_dict_train)
    numberOfRemainingBatches= max(0, numberOfBatches- i_global)
    if numberOfBatches >0:
      print("Training net for %d batches of size %d"%(numberOfRemainingBatches, dataManagerTrain.getBatchSize()))
      print("Initial loss %f"%stepLoss)
      self.testPerformance(i_global,trainDataBatch,dataManagerTest)
    else:
      return
    time0 = time.time()
    tflearn.is_training(True, session=self.session)
    for iterNum in range(numberOfRemainingBatches):
      trainDataBatch= dataManagerTrain.getRandomBatch()
      x_batchTrain, labels_batchTrain = trainDataBatch
      feed_dict_train= {self.X : x_batchTrain, self.Y: labels_batchTrain }
      i_global, __, stepLoss,= self.session.run( [self.global_step, self.optimizer, self.loss],
                                                 feed_dict=feed_dict_train )
      

      print("iterNum %d/%d trainLoss: %3.4f"%((i_global), numberOfBatches, stepLoss))
      sys.stdout.flush()      
      if i_global % EVALUATE_AT ==0:
        timeBatches= time.time() -time0
        timeTest=time.time()
        self.testPerformance(i_global,trainDataBatch,dataManagerTest)
        print("%d batches time: %f s. time for test %f s"%(EVALUATE_AT, timeBatches, time.time() -timeTest))
        time0 = time.time()
        sys.stdout.flush()
      if (i_global + 1 ) % CHECK_POINT_AT==0:
        self.saver.save(self.session, save_path= self.checkPointsNames, global_step= self.global_step) 
        print("\nSaved checkpoint.")   

    self.testPerformance(i_global,trainDataBatch,dataManagerTest)
    
  def accuracy_score(self, labels, predictions ):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])    

  def testPerformance(self, stepNum, trainDataBatch, testDataManager):
    tflearn.is_training(False, session=self.session)      

    batch_x, batch_y= trainDataBatch
    feed_dict_train= {self.X : batch_x, self.Y: batch_y}
    c_e_train, y_pred_train, merged = self.session.run( [self.loss, self.y_pred, self.merged_summaries], 
                                                 feed_dict = feed_dict_train )
    train_accuracy=  self.accuracy_score(batch_y, y_pred_train)
    train_auc= roc_auc_score(batch_y, y_pred_train)
    self.train_writer.add_summary(merged, stepNum)
    if not testDataManager is None:
      y_pred_list=[]
      labels_list=[]
      for images, labels, __ in testDataManager.getIteratorTestBatch(20):
        feed_dict_train= {self.X : images, self.Y : labels }
        y_pred, merged= self.session.run( [self.y_pred,self.merged_summaries], feed_dict=feed_dict_train )
        y_pred_list.append(y_pred)
        labels_list.append(labels)

      y_pred= np.concatenate(y_pred_list)
      labels= np.concatenate(labels_list)
      test_accuracy= self.accuracy_score(labels, y_pred)
      test_auc= roc_auc_score(labels, y_pred)

      self.test_writer.add_summary(merged, stepNum)
      print("iterNum %d accuracy(train/test) %f / %f   auc  %f / %f"%(stepNum,train_accuracy,test_accuracy,
                                                                      train_auc, test_auc))
    tflearn.is_training(True, session=self.session)      
    

#  def predictNet(self, images):
#
#    feed_dict_train= {self.X : images}
#    y_pred= self.session.run( self.y_pred,
#                              feed_dict=feed_dict_train )
#    return y_pred[:,1]                                                 

  def predictNet(self, dataManger):
    tflearn.is_training(False, session=self.session)      
    y_pred_list=[]
    labels_list=[]
    metadataAndId_list=[]
    for images, labels, metadataAndId in dataManger.getIteratorPredictBatch():
      feed_dict_train= {self.X : images}
      y_pred= self.session.run( self.y_pred, feed_dict=feed_dict_train )
      y_pred_list.append(y_pred)
      labels_list.append(labels)
      metadataAndId_list+= metadataAndId
    y_pred= np.concatenate(y_pred_list)
    labels= np.concatenate(labels_list)
    accuracy= self.accuracy_score(labels, y_pred)
    auc= roc_auc_score(labels, y_pred)
    y_pred=y_pred[:,1]
    pos_labels= labels[:,1]
    print("GLOBAL test accuracy: %f  auc: %f"%(accuracy,auc))
    posScores= y_pred[pos_labels==1]
    negScores= y_pred[pos_labels==0]
    print("pos_mean %f pos_sd %f neg_mean %f neg_perc90 %f neg_perc95 %f"%(np.mean(posScores),np.std(posScores),
                                                         np.mean(negScores), np.percentile(negScores,90),
                                                         np.percentile(negScores,95)))
    with open("/home/rsanchez/app/scipion/pyworkflow/em/packages/xmipp3/backupDeepLearning/scores.tab","w") as f:
      best_accuracy= 0
      thr=0
      for score, label in zip(y_pred, pos_labels):
        f.write("%f\t%d\n"%(float(score),float(label)))
        tmpY=[ 1 if elem>=score else 0 for elem in y_pred]
        tmp_accu=accuracy_score(pos_labels, tmpY)
        if tmp_accu> best_accuracy:
          best_accuracy= tmp_accu
          thr= score
    print("best thr %f --> accuracy %f"%(thr,best_accuracy))
    return y_pred , labels, metadataAndId_list

class DataManager(object):
  
  @staticmethod
  def getFREE_GPU_MEM():
    if os.environ['CUDA']:
      from tensorflow.python.client import device_lib
      local_device_protos = device_lib.list_local_devices()
      gpu_mem_list=[x.memory_limit for x in local_device_protos if x.device_type == 'GPU']
      nBytes= gpu_mem_list[0]
      return nBytes
    else:
      return None
  FREE_GPU_MEM=  getFREE_GPU_MEM.__func__()
  def __init__(self, posImagesFname, posImagesObject,  negImagesFname=None, negImagesObject=None):
    
    self.mdFalse=None
    self.nFalse=0
    self.mdTrue  = md.MetaData() 
    self.mdTrue.randomize( md.MetaData(posImagesFname) )
    
    xdim, ydim, _     = posImagesObject.getDim()
    self.shape= (xdim,ydim,1)
    self.nTrue= posImagesObject.getSize()    
#    self.batchSize= self.estimateBatchSize()
    self.batchSize= 128
#    self.batchSize= 256  # Para la primera version del modelo
#    self.batchSize= 512  # Para la primera version del modelo

    self.splitPoint= self.batchSize//2

    self.batchStack   = np.zeros((self.batchSize, xdim,ydim,1))
    if not ( negImagesFname is None or negImagesObject is None):
      self.mdFalse  = md.MetaData() 
      self.mdFalse.randomize( md.MetaData(negImagesFname) )
      xdim_1, ydim_1, _ = negImagesObject.getDim()
      self.nFalse= negImagesObject.getSize()    
      assert xdim==xdim_1 and ydim==ydim_1
      self.getRandomBatch= self.getRandomBatchWorker 
    else:
        self.getRandomBatch= self.NOgetRandomBatchWorker
  
  def getMetadata(self):
    return self.mdTrue, self.mdFalse
  
#  def estimateBatchSize(self):
#    if os.environ['CUDA']:
#      nBytes= int(0.3* DataManager.FREE_GPU_MEM) #30% gpu memory for data. Rest for model
##      print(nBytes, np.prod(self.shape),PICK_MEM, nBytes // (np.prod(self.shape)*PICK_MEM))
#      batchSize= nBytes // (np.prod(self.shape)*PICK_MEM)
##      print("Batchsize %d"%batchSize)
#      return batchSize
#    else:
#      return 4096
    
  
  def getNBatches(self, Nepochs):
    return  int(ceil(2*self.nTrue*Nepochs/self.batchSize))

  def getBatchSize(self):
    return self.batchSize
  
  def NOgetRandomBatchWorker(self):
    raise ValueError("needs positive and negative images to compute random batch. Just pos provided")
    
  def getRandomBatchWorker(self):
    batchSize = self.batchSize
    splitPoint = self.splitPoint
    batchStack   = self.batchStack
    batchLabels  = np.zeros((batchSize, 2))
    mdTrue  = md.MetaData() 
    mdTrue.randomize( self.mdTrue )
    mdFalse  = md.MetaData() 
    mdFalse.randomize( self.mdFalse )
    
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
      if n>=batchSize:
          break
    shuffInd= np.random.choice(n,n, replace=False)
    batchStack= batchStack[shuffInd, ...]
    batchLabels= batchLabels[shuffInd, ...]    
    return batchStack, batchLabels

  def getIteratorPredictBatch(self):
    batchSize = self.batchSize
    xdim,ydim,nChann= self.shape
    batchStack = np.zeros((self.batchSize, xdim,ydim,nChann))
    batchLabels  = np.zeros((batchSize, 2))
    mdTrue  = self.mdTrue
    I = xmipp.Image()
    n = 0
    metaDataList= []
    for id in mdTrue:
#      print("True id", id)
      fnImage = mdTrue.getValue(md.MDL_IMAGE, id)
      I.read(fnImage)
      batchStack[n,...]= np.expand_dims(I.getData(),-1)
      batchLabels[n, 1]= 1
      metaDataList.append( (mdTrue, id) )
      n+=1
      if n>=batchSize:
        yield batchStack, batchLabels,  metaDataList
        n=0
        metaDataList= []
        batchLabels  = np.zeros((batchSize, 2))

    if not self.mdFalse is None:
      mdFalse= self.mdFalse
      for id in mdFalse:
        fnImage = mdFalse.getValue(md.MDL_IMAGE, id)
        I.read(fnImage)
        batchStack[n,...]= np.expand_dims(I.getData(),-1)
        batchLabels[n, 0]= 1
        metaDataList.append( (mdFalse, id) )        
        n+=1
        if n>=batchSize:
          yield batchStack, batchLabels, metaDataList
          n=0
          metaDataList= []
          batchLabels  = np.zeros((batchSize, 2))
    if n>0:
      yield batchStack[:n,...], batchLabels[:n,...], metaDataList[:n]
      
  def getIteratorTestBatch(self, nBatches= 10):
    batchSize = self.batchSize
    xdim,ydim,nChann= self.shape
    batchStack = np.zeros((self.batchSize, xdim,ydim,nChann))
    batchLabels  = np.zeros((batchSize, 2))
    mdTrue  = self.mdTrue
    mdFalse = self.mdFalse
    I = xmipp.Image()
    n = 0
    metaDataList= []

    currNBatches=0
    for idTrue, idFalse in zip(mdTrue, mdFalse):
      fnImageTrue = mdTrue.getValue(md.MDL_IMAGE, idTrue)
      I.read(fnImageTrue)
      batchStack[n,...]= np.expand_dims(I.getData(),-1)
      batchLabels[n, 1]= 1
      metaDataList.append( (mdTrue, idTrue) )
      n+=1
      if n>=batchSize:
        yield batchStack, batchLabels,  metaDataList
        n=0
        metaDataList= []
        batchLabels  = np.zeros((batchSize, 2))
        currNBatches+=1
        if currNBatches>=nBatches:
          break     
      fnImageFalse = mdFalse.getValue(md.MDL_IMAGE, idFalse)
      I.read(fnImageFalse)
      batchStack[n,...]= np.expand_dims(I.getData(),-1)
      batchLabels[n, 0]= 1
      metaDataList.append( (mdTrue, idFalse) )
      n+=1
      if n>=batchSize:
        yield batchStack, batchLabels,  metaDataList
        n=0
        metaDataList= []
        batchLabels  = np.zeros((batchSize, 2))
        currNBatches+=1
        if currNBatches>=nBatches:
          break
    if n>0:
      yield batchStack[:n,...], batchLabels[:n,...], metaDataList[:n]

         
