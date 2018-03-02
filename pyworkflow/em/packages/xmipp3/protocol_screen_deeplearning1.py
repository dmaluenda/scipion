# **************************************************************************
# *
# * Authors:  Ruben Sanchez (rsanchez@cnb.csic.es), April 2017
# *
# * Unidad de  Bioinformatica of Centro Nacional de Biotecnologia , CSIC
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************


import os

from pyworkflow.utils.path import copyFile, cleanPath, copyTree
import pyworkflow.protocol.params as params
from pyworkflow.em.protocol import ProtProcessParticles
import pyworkflow.em.metadata as md
from pyworkflow.em.packages.xmipp3.convert import writeSetOfParticles, setXmippAttributes


class XmippProtScreenDeepLearning1(ProtProcessParticles):
    """ Protocol for screening particles using deep learning. """
    _label = 'screen deep learning 1'
    
    #--------------------------- DEFINE param functions --------------------------------------------
    def _defineParams(self, form):
        form.addSection(label='Input')
        form.addParam('doContinue', params.BooleanParam, default=False,
                      label='Continue training from a previous run?',
                      help='If you set to *Yes*, you should select a previous'
                      'run of type *%s* class and some of the input parameters'
                      'will be taken from it.' % self.getClassName())
        form.addParam('continueRun', params.PointerParam, pointerClass=self.getClassName(),
                      condition='doContinue', allowsNull=True,
                      label='Select previous run',
                      help='Select a previous run to continue from.')

        form.addParam('inPosSetOfParticles', params.PointerParam, label="Consensus particles (mostly true particles)", 
                      pointerClass='SetOfParticles', 
                      help='Select the intersection set of particles (mostly true particles).\n'
               'Recomended set: good Z-score AND consensus of several pickers')


        form.addParam('numberOfNegativeSets', params.IntParam, default='1',
                          label='Number of different negative dataset',
                          help='Data from all negative datasets will be used for training. Maximun number is 4.\n'
                       'Recomended negative data sets are 3:\n'
                   '1) Random picked (pick noise protocol)\n'
                   '2) Bad Z-score from multiple picker OR consensus\n'
                   '3) Bad Z-score from multiple picker picked just by one\n')
        for num in range( 1, 5):
            form.addParam('negativeSet_%d'%num, params.PointerParam, 
                            condition='numberOfNegativeSets<=0 or numberOfNegativeSets >=%d'%num,
                            label="Set of negative train particles %d"%num, 
                            pointerClass='SetOfParticles',
                            help='Select the set of negative particles for training.')
            form.addParam('inNegWeight_%d'%num, params.IntParam, default='1',
                            condition='numberOfNegativeSets<=0 or numberOfNegativeSets >=%d'%num,
                            label="Weight of negative train particles %d"%num, 
                            help='Select the weigth for the negative set of particles.')


        form.addParam('predictSetOfParticles', params.PointerParam, label="Set of putative particles to predict", 
                      pointerClass='SetOfParticles',
                      help='Select the set of putative particles particles to classify.')

        form.addParam('auto_stopping',params.BooleanParam, default=True,
                      label='Auto stop training when convergency is detected?',
                      help='If you set to *Yes*, the program will automatically stop training'
                      'if there is no improvement for consecutive 2 epochs, learning rate will be decreased'
                      'by a factor 10. If learningRate_t < 0.01*learningrate_0 training will stop. Warning: '
                      'Sometimes convergency seems to be reached, but after time, improvement can still happen. '
                      'Not recommended for very small data sets (<100 true particles)')

        if 'CUDA' in os.environ and not os.environ['CUDA']=="False":

            form.addParam('gpuToUse', params.IntParam, default='0',
                          label='Which GPU to use:',
                          help='Currently just one GPU will be use, by '
                         'default GPU number 0 You can override the default '
                               'allocation by providing other GPU number, p.e. 2')
        else:
            form.addParallelSection(threads=8, mpi=0)

        form.addParam('nEpochs', params.FloatParam, label="Number of epochs", default=10.0, expertLevel=params.LEVEL_ADVANCED,
                      help='Number of epochs for neural network training.')  
        form.addParam('learningRate', params.FloatParam, label="Learning rate", default=1e-3, expertLevel=params.LEVEL_ADVANCED,
                      help='Learning rate for neural network training')
        form.addParam('nModels', params.IntParam, label="Number of models for ensemble", default=3, expertLevel=params.LEVEL_ADVANCED,
                      help='Number of models to fit in order to build an ensamble. Tipical values are 3 to 10. The more the better'
                      'until a point where no gain is obtained')  

        form.addSection(label='testingData')
        form.addParam('doTesting', params.BooleanParam, default=False,
                      label='Perform testing during training?',
                      help='If you set to *Yes*, you should select a testing positive set '
                      'and a testing negative set')
        form.addParam('testPosSetOfParticles', params.PointerParam, condition='doTesting',
                      label="Set of positive test particles", 
                      pointerClass='SetOfParticles',
                      help='Select the set of ground true positive particles.')
        form.addParam('testNegSetOfParticles', params.PointerParam, condition='doTesting',
                      label="Set of negative test particles", 
                      pointerClass='SetOfParticles',
                      help='Select the set of ground false positive particles.')

    
    #--------------------------- INSERT steps functions --------------------------------------------
    def _insertAllSteps(self):
        '''
            negSetDict= { fname: [(SetOfParticles, weight:int)]}
        '''
        posSetTrainDict={self._getExtraPath("inputTrueParticlesSet.xmd"):(self.inPosSetOfParticles.get(), 1)}
        negSetTrainDict={}
        for num in range( 1, 5):
            if self.numberOfNegativeSets<=0 or self.numberOfNegativeSets >=num:
                negSetTrainDict[self._getExtraPath("negativeSet_%d.xmd"%num)]=(self.__dict__["negativeSet_%d"%num].get(),
                                                                               self.__dict__["inNegWeight_%d"%num].get())
        setPredict={self._getExtraPath("predictSetOfParticles.xmd"): (self.predictSetOfParticles.get(),1) }
        setTestPos={self._getExtraPath("testTrueParticlesSet.xmd"):  (self.testPosSetOfParticles.get(),1)}
        setTestNeg={self._getExtraPath("testFalseParticlesSet.xmd"): (self.testNegSetOfParticles.get(),1)}
  
        self._insertFunctionStep('convertInputStep', posSetTrainDict, negSetTrainDict, setPredict, setTestPos, setTestNeg) 
        self._insertFunctionStep('train',  posSetTrainDict, negSetTrainDict, setTestPos, setTestNeg, self.learningRate.get()) 
        self._insertFunctionStep('predict',setTestPos,setTestNeg, setPredict, posSetTrainDict, negSetTrainDict)
        self._insertFunctionStep('createOutputStep')
        
    #--------------------------- STEPS functions --------------------------------------------   
    def convertInputStep(self, *dataDicts):
        for dataDict in dataDicts:
          for fnameParticles in sorted(dataDict):
            print(fnameParticles, dataDict[fnameParticles][0], not dataDict[fnameParticles][0] is None)
            if not dataDict[fnameParticles][0] is None:
                writeSetOfParticles(dataDict[fnameParticles][0], fnameParticles)
            
    def train(self, posTrainDict, negTrainDict, posTestDict, posTestNeg, learningRate):
        '''
        posTrainDict, negTrainDict, posTestDict, posTestNeg: { fname: [(SetOfParticles, weight:int)]}
        learningRate: float
        '''
        if self.doContinue.get():
            previousRun=self.continueRun.get()
            print( previousRun._getExtraPath('nnetData') )
            copyTree(previousRun._getExtraPath('nnetData'),self._getExtraPath('nnetData'))
      
        from pyworkflow.em.packages.xmipp3.deepLearning1 import  DeepTFSupervised, DataManager, updateEnviron, tf_intarnalError
        
        if self.gpuToUse:
            updateEnviron( self.gpuToUse.get() )
            numberOfThreads=None
        else:
            numberOfThreads=self.numberOfThreads.get()

        trainDataManager= DataManager( posSetDict= posTrainDict,
                                       negSetDict= negTrainDict)

        if not list(posTestDict.values())[0][0] is None and not list(negTestDict.values())[0][0] is None:
            testDataManager= DataManager(posSetDict= posTestDict,negSetDict= negTestDict)
        else:
            testDataManager= None
            nEpochs= self.nEpochs.get()
            numberOfBatches = trainDataManager.getNBatches(nEpochs)
            numModels= self.nModels.get()
            assert numModels>=1, "Error, nModels>=1"
            for i in range(numModels):
                print("Training model %d/%d"%((i+1), numModels))
                nnet = DeepTFSupervised(rootPath=self._getExtraPath("nnetData"), modelNum=i)
                try: 
                    nnet.createNet(trainDataManager.shape[0], trainDataManager.shape[1], trainDataManager.shape[2], trainDataManager.nTrue)
                    nnet.startSessionAndInitialize(numberOfThreads)
                except tf_intarnalError as e:
                    if e._error_code==13:
                        raise Exception("Out of gpu Memory. gpu # %d"%(self.gpuToUse.get()))
                    else:
                        raise e
                nnet.trainNet(numberOfBatches, trainDataManager, learningRate, testDataManager, self.auto_stopping.get())
                nnet.close(saveModel= False) #Models will be automatically saved during training, so True no needed
        #        self.predict( testPosSetOfParticles, testNegSetOfParticles, inPosSetOfParticles, inNegSetOfParticles)
        #        raise ValueError("Debug mode")
                del nnet


    def predict(self, posTestDict, negTestDict, setPredict, posTrainDict, negTrainDict):
        from pyworkflow.em.packages.xmipp3.deepLearning1 import  DeepTFSupervised, DataManager, updateEnviron
        import numpy as np
        from sklearn.metrics import accuracy_score, roc_auc_score

        if self.gpuToUse:
            updateEnviron( self.gpuToUse.get() )
            numberOfThreads=None
        else:
            numberOfThreads=self.numberOfThreads.get()

        trainDataManager= DataManager( posSetDict= posTrainDict,
                                       negSetDict= negTrainDict)

        predictDataManager= DataManager( posSetDict= setPredict,
                                         negSetDict= None)
        numModels= self.nModels.get()

        resultsDictPos={}
        resultsDictNeg={}
        for i in range(numModels):
            print("Predicting with model %d/%d"%((i+1), numModels))

            nnet = DeepTFSupervised(rootPath=self._getExtraPath("nnetData"), modelNum=i)
            nnet.createNet( trainDataManager.shape[0], trainDataManager.shape[1], trainDataManager.shape[2], trainDataManager.nTrue)
            nnet.startSessionAndInitialize(numberOfThreads)
            y_pred, labels, typeAndIdList = nnet.predictNet(predictDataManager)
            nnet.close(saveModel= False)

            for score, label, (mdIsPosType, mdId, mdNumber) in zip(y_pred , labels, typeAndIdList):
              if mdIsPosType==True:
                 try:
                     resultsDictPos[(mdId, mdNumber)]+= float(score)/float(numModels)
                 except KeyError:
                     resultsDictPos[(mdId, mdNumber)]= float(score)/float(numModels)
              else:
                 try:
                     resultsDictNeg[(mdId, mdNumber)]+= float(score)/float(numModels)
                 except KeyError:
                     resultsDictNeg[(mdId, mdNumber)]= float(score)/float(numModels)

        metadataPosList, metadataNegList= predictDataManager.getMetadata(None)
        for (mdId, mdNumber) in resultsDictPos:
             metadataPosList[mdNumber].setValue(md.MDL_ZSCORE_DEEPLEARNING1, resultsDictPos[(mdId, mdNumber)], mdId)

        for (mdId, mdNumber) in resultsDictNeg:
             metadataNegList[mdNumber].setValue(md.MDL_ZSCORE_DEEPLEARNING1, resultsDictNeg[(mdId, mdNumber)], mdId)

        metadataPosList[0].write(self._getPath("particles.xmd"))
        assert len(metadataPosList)==1, "Just one SetOfParticles to predict allowed"
        if not list(posTestDict.values())[0][0] is None and not list(negTestDict.values())[0][0] is None:
          testDataManager= DataManager(posSetDict= posTestDict,
                               negSetDict= negTestDict)

          nnet.close(saveModel= False)
          scores_list=[]
          labels_list=[]
          cum_acc_list=[]
          cum_auc_list=[]
          for i in range(numModels):
            print("Predicting test data with model %d/%d"%((i+1), numModels))
            labels_list.append([])
            scores_list.append([])
            nnet = DeepTFSupervised(rootPath=self._getExtraPath("nnetData"), modelNum=i)
            nnet.createNet( trainDataManager.shape[0], trainDataManager.shape[1], trainDataManager.shape[2], trainDataManager.nTrue)
            nnet.startSessionAndInitialize(numberOfThreads)
            y_pred, labels, typeAndIdList = nnet.predictNet(testDataManager)
            scores_list[-1].append(y_pred)
            labels= [ 0 if label[0]==1.0 else 1 for label in labels]
            labels_list[-1].append(labels)
            curr_auc= roc_auc_score(labels, y_pred)
            curr_acc= accuracy_score(labels, [1 if y>0.5 else 0 for y in y_pred])
            cum_acc_list.append(curr_acc)
            cum_auc_list.append(curr_auc)
            print("Model %d test accuracy: %f  auc: %f"%(i, curr_acc, curr_auc))
            nnet.close(saveModel= False)
          labels= np.mean( labels_list, axis=0)[0,:]
          assert np.all( (labels==1) | (labels==0)), "Error, labels mismatch"
          scores= np.mean(scores_list, axis=0)[0,:]
          auc_val= roc_auc_score(labels, scores)
          scores[ scores>0.5]=1
          scores[ scores<=0.5]=0
          accuracy_score(labels, scores)
          print(">>>>>>>>>>>>\nEnsemble test accuracy            : %f  auc: %f"%(accuracy_score(labels, scores) , auc_val))
          print("Mean single model test accuracy: %f  auc: %f"%(np.mean(cum_acc_list) , np.mean(cum_auc_list)))

    def createOutputStep(self):
        imgSet = self.predictSetOfParticles.get()
        partSet = self._createSetOfParticles()
        partSet.copyInfo(imgSet)
        partSet.copyItems(imgSet,
                            updateItemCallback=self._updateParticle,
                            itemDataIterator=md.iterRows(self._getPath("particles.xmd"), sortByLabel=md.MDL_ITEM_ID))
        self._defineOutputs(outputParticles=partSet)
        self._defineSourceRelation(imgSet, partSet)

    
    #--------------------------- INFO functions --------------------------------------------
    def _summary(self):
        summary = []
        return summary
    
    def _methods(self):
        pass
    
    #--------------------------- UTILS functions --------------------------------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, md.MDL_ZSCORE_DEEPLEARNING1)
        if row.getValue(md.MDL_ENABLED) <= 0:
            item._appendItem = False
        else:
            item._appendItem = True
