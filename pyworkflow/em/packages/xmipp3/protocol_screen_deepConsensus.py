# **************************************************************************
# *
# * Authors:    Carlos Oscar Sorzano (coss@cnb.csic.es)
# *             Tomas Majtner (tmajtner@cnb.csic.es)  -- streaming version
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
# *  e-mail address 'coss@cnb.csic.es'
# *
# **************************************************************************
"""
Consensus picking protocol
"""
import os, time
from math import sqrt
from glob import glob
from pyworkflow.object import Set, String
import pyworkflow.utils as pwutils
from pyworkflow.utils.path import cleanPath, makePath, moveFile, copyFile, createLink, cleanPattern, replaceExt, cleanPath
import pyworkflow.protocol.params as params
from pyworkflow.em.protocol import ProtParticlePicking
from pyworkflow.protocol.constants import *
from pyworkflow.em.constants import RELATION_CTF

from pyworkflow.em.data import SetOfCoordinates, Coordinate, SetOfParticles
from pyworkflow.em.packages.xmipp3.protocol_pick_noise import (pickAllNoiseWorker, writeSetOfCoordsFromFnames,
                                                               writePosFilesStepWorker)
from  pyworkflow.em.packages.xmipp3.protocol_screen_deeplearning1 import trainWorker, predictWorker, XmippProtScreenDeepLearning1
import pyworkflow.em.metadata as MD
import xmipp
from pyworkflow.em.packages.xmipp3.convert import readSetOfParticles, setXmippAttributes, micrographToCTFParam
import numpy as np
from joblib import Parallel, delayed

DEEP_PARTICLE_SIZE= 128
#DEEP_PARTICLE_SIZE= 300

MIN_NUM_CONSENSUS_COORDS= 100

class XmippProtScreenDeepConsensus(ProtParticlePicking, XmippProtScreenDeepLearning1):
    """ TODO: HELP
    Protocol to estimate the agreement between different particle picking
    algorithms. The protocol takes several Sets of Coordinates calculated
    by different programs and/or different parameter settings. Let's say:
    we consider N independent pickings. Then, a coordinate is considered
    to be a correct particle if M pickers have selected the same particle
    (within a radius in pixels specified in the form).
    
    If you want to be very strict, then set M=N; that is, a coordinate
    represents a particle if it has been selected by all particles (this
    is the default behaviour). Then you may relax this condition by setting
    M=N-1, N-2, ...
    
    If you want to be very flexible, set M=1, in this way it suffices that
    1 picker has selected the coordinate to be considered as a particle. Note
    that in this way, the cleaning of the dataset has to be performed by other
    means (screen particles, 2D and 3D classification, ...).
    """

    _label = 'deep consensus picking'
    CONSENSUS_COOR_PATH_TEMPLATE="regularConsensus_%s"
    def __init__(self, **args):
        ProtParticlePicking.__init__(self, **args)
        self.stepsExecutionMode = params.STEPS_SERIAL

    def _defineParams(self, form):
        form.addSection(label='Input')
        #CONTINUE FROM PREVIOUS TRAIN
        form.addParam('doContinue', params.BooleanParam, default=False,
                      label='use previously trained model?',
                      help='If you set to *Yes*, you should select a previous'
                      'run of type *%s* or *%s* class and some of the input parameters'
                      'will be taken from it.' %(self.getClassName(), None) )
        form.addParam('continueRun', params.PointerParam, pointerClass=self.getClassName(),
                      condition='doContinue', allowsNull=True,
                      label='Select previous run',
                      help='Select a previous run to continue from.')
        form.addParam('keepTraining', params.BooleanParam, default=True,condition='doContinue',
                      label='continue training on previously trainedModel?',
                      help='If you set to *Yes*, you should provide training set')
        #CONTINUE PARAMETERS
        form.addParam('inputCoordinates', params.MultiPointerParam,
                      pointerClass='SetOfCoordinates',
                      label="Input coordinates",
                      help='Select the set of coordinates to compare')
        form.addParam('consensusRadius', params.FloatParam, default=0.1,
                      label="Relative Radius", expertLevel=params.LEVEL_ADVANCED,
                      help="All coordinates within this radius (as fraction of particle size) "
                           "are presumed to correspond to the same particle")
        
        if 'CUDA' in os.environ and not os.environ['CUDA']=="False":

            form.addParam('gpuToUse', params.IntParam, default='0',
                          label='Which GPU to use:',
                          help='Currently just one GPU will be use, by '
                               'default GPU number 0 You can override the default '
                               'allocation by providing other GPU number, p.e. 2'
                               '\nif -1 no gpu but all cpu cores will be used')
            
        
        form.addSection(label='Preprocess')
        form.addParam('notePreprocess', params.LabelParam,
                      label='What have you done with the micrographs from which you have picked your coordinates?')

        form.addParam('doInvert', params.BooleanParam, default=None,
                      label='Invert contrast', 
                      help='Invert the contrast if your particles are black '
                           'over a white background.  Xmipp, Spider, Relion '
                           'and Eman require white particles over a black '
                           'background. Frealign (up to v9.07) requires black '
                           'particles over a white background')
        
        form.addParam('doFlip', params.BooleanParam, default=None,
                      label='Phase flipping',
                      help='Use the information from the CTF to compensate for '
                           'phase reversals.\n'
                           'Phase flip is recommended in Xmipp or Eman\n'
                           '(even Wiener filtering and bandpass filter are '
                           'recommended for obtaining better 2D classes)\n'
                           'Otherwise (Frealign, Relion, Spider, ...), '
                           'phase flip is not recommended.')
                           
        form.addParam('ctfRelations', params.RelationParam, allowsNull=True,
                      relationName=RELATION_CTF, condition="doFlip",
                      attributeName='getInputMicrographs',
                      label='CTF estimation',
                      help='Choose some CTF estimation related to input '
                           'micrographs. \n CTF estimation is needed if you '
                           'want to do phase flipping or you want to '
                           'associate CTF information to the particles.')
        
        form.addSection(label='Training')

        form.addParam('auto_stopping',params.BooleanParam, default=True, condition="not doContinue or keepTraining",
                      label='Auto stop training when convergency is detected?', 
                      help='If you set to *Yes*, the program will automatically stop training'
                      'if there is no improvement for consecutive 2 epochs, learning rate will be decreased'
                      'by a factor 10. If learningRate_t < 0.01*learningrate_0 training will stop. Warning: '
                      'Sometimes convergency seems to be reached, but after time, improvement can still happen. '
                      'Not recommended for very small data sets (<100 true particles)')
        
        form.addParam('nEpochs', params.FloatParam, label="Number of epochs", default=5.0,
                      condition="not doContinue or keepTraining", help='Number of epochs for neural network training.')
        form.addParam('learningRate', params.FloatParam, label="Learning rate", default=1e-4,
                      condition="not doContinue or keepTraining", help='Learning rate for neural network training')
        form.addParam('nModels', params.IntParam, label="Number of models for ensemble", default=2, expertLevel=params.LEVEL_ADVANCED,
                      condition="not doContinue",
                      help='Number of models to fit in order to build an ensamble. Tipical values are 1 to 5. The more the better'
                      'until a point where no gain is obtained. Each model increases running time linearly')

        form.addParam('l2RegStrength', params.FloatParam, label="Regularization strength", default=1e-5, expertLevel=params.LEVEL_ADVANCED,
                      condition="not doContinue or keepTraining", help='L2 regularization for neural network weights. Make it '
                                                                  'bigger if suffering overfitting. Typical values range from 1e-1 to 1e-6')  
        
        form.addSection(label='Additional training data')
        form.addParam('noteAdditionalTrainingData', params.LabelParam,
                      label='You can provide additional sets of true particles and false particles for training. See help '
                            'for the requirements',  help='Our method internally, uses particles that are extracted from '
                            'preprocess micrographs: 1) donwsampled to the required size such that the particle box size is 128\n'
                            '2) phase flipped using CTF.\n3)Normalize to 0 mean and 1 std\n Then, particles are extracted with no '
                            'further alteration ')
        
        form.addSection(label='Testing data')
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

        form.addParallelSection(threads=2, mpi=1)
        
#--------------------------- INSERT steps functions ---------------------------
    def _insertAllSteps(self):
        
        self.inputMicrographs=None
        self.boxSize= None
        self.coordinatesDict= {}

        allMicIds= self.getInputMicrographs().getIdSet()        
        
        self._insertFunctionStep("initializeDirsStep")
        self.insertMicsPreprocSteps(allMicIds)
#        self._insertFunctionStep('calculateConsensusStep', allMicIds, 'OR') #OR before noise always
        self.insertCaculateConsensusSteps( allMicIds, 'OR') #OR before noise always

        if not self.doContinue.get()  or self.keepTraining.get():
            self._insertFunctionStep('pickNoise')
#            self._insertFunctionStep('calculateConsensusStep', allMicIds, 'AND')
            self.insertCaculateConsensusSteps( allMicIds, 'AND')

            self._insertFunctionStep('extractParticles', allMicIds, 'AND')
            self._insertFunctionStep('joinSetOfParticles', allMicIds, 'AND')
            
            self._insertFunctionStep('extractParticles', allMicIds, 'NOISE')
            self._insertFunctionStep('joinSetOfParticles', allMicIds, 'NOISE')

        self._insertFunctionStep('extractParticles', allMicIds, 'OR')
        self._insertFunctionStep('joinSetOfParticles', allMicIds, 'OR')
            
        self._insertFunctionStep('trainCNN', )
        self._insertFunctionStep('predictCNN', )
        self._insertFunctionStep('createOutputStep')
    
    def initializeDirsStep(self):
        '''
            create paths where data will be saved
        '''        
        makePath( self._getExtraPath('preProcMics') )
        for mode in ["AND", "OR", "NOISE"]:
            consensusCoordsPath= XmippProtScreenDeepConsensus.CONSENSUS_COOR_PATH_TEMPLATE%mode
            makePath(self._getExtraPath(consensusCoordsPath))
        
    def getInputMicrographs(self):
        
        if not self.inputMicrographs:
            self.inputMicrographs= self.inputCoordinates[0].get().getMicrographs()
        return self.inputMicrographs

    def getBoxSize(self):
      
        if not self.boxSize:
          firstCoords = self.inputCoordinates[0].get()
          self.boxSize= firstCoords.getBoxSize()
          self.downFactor = self.boxSize /float(DEEP_PARTICLE_SIZE)
        return self.boxSize

    def getDownFactor(self):

        if not self.downFactor:
          firstCoords = self.getInputMicrographs()
          self.boxSize= firstCoords.getBoxSize()
          self.downFactor = self.boxSize /float(DEEP_PARTICLE_SIZE)
          assert self.downFactor >= 1, "Error, the particle box size must be greater or equal than 128."
          
        return self.downFactor


    def insertCaculateConsensusSteps(self, micIds, mode):
        # Take the sampling rates
        consensus= -1 if mode=="AND" else 1
        newDataPath= XmippProtScreenDeepConsensus.CONSENSUS_COOR_PATH_TEMPLATE%mode
        makePath(self._getExtraPath(newDataPath))
        for micId in micIds:
            self._insertFunctionStep('calculateConsensusStep', micId, newDataPath, consensus)
            
        self._insertFunctionStep('loadConsensusCoords',  newDataPath, mode, True )
        
    def calculateConsensusStep(self, micId, newDataPath, consensus):

        Tm = []
        for coordinates in self.inputCoordinates:
            Tm.append(coordinates.get().getMicrographs().getSamplingRate())
        
        # Get all coordinates for this micrograph
        coords = []
        Ncoords = 0
        n=0
        for coordinates in self.inputCoordinates:
            coordArray = np.asarray([x.getPosition() 
                                     for x in coordinates.get().iterCoordinates(micId)], dtype=float)
            coordArray *= float(Tm[n])/float(Tm[0])
            coords.append(np.asarray(coordArray,dtype=int))
            Ncoords += coordArray.shape[0]
            n+=1
        
        allCoords = np.zeros([Ncoords,2])
        votes = np.zeros(Ncoords)
        
        # Add all coordinates in the first method
        N0 = coords[0].shape[0]
        inAllMicrographs = consensus <= 0 or consensus == len(self.inputCoordinates)
        if N0==0 and inAllMicrographs:
            return
        elif N0>0:
            allCoords[0:N0,:] = coords[0]
            votes[0:N0] = 1
        
        boxSize= self.getBoxSize()
        consensusNpixels = self.consensusRadius.get() * boxSize
        assert consensusNpixels >=0, "Error, consensusNpixel must be >=0" 
        # Add the rest of coordinates
        Ncurrent = N0
        for n in range(1, len(self.inputCoordinates)):
            for coord in coords[n]:
                if Ncurrent>0:
                    dist = np.sum((coord - allCoords[0:Ncurrent])**2, axis=1)
                    imin = np.argmin(dist)
                    if sqrt(dist[imin]) < consensusNpixels:
                        newCoord = (votes[imin]*allCoords[imin,]+coord)/(votes[imin]+1)
                        allCoords[imin,] = newCoord
                        votes[imin] += 1
                    else:
                        allCoords[Ncurrent,:] = coord
                        votes[Ncurrent] = 1
                        Ncurrent += 1
                else:
                    allCoords[Ncurrent, :] = coord
                    votes[Ncurrent] = 1
                    Ncurrent += 1

        # Select those in the consensus
        if consensus <= 0:
            consensus = len(self.inputCoordinates)

        consensusCoords = allCoords[votes>=consensus,:]
        # Write the consensus file only if there
        # are some coordinates (size > 0)
        if consensusCoords.size:
            np.savetxt(self._getExtraPath(newDataPath, 'consensus_%06d.txt' % micId), consensusCoords)

    def loadConsensusCoords(self, newDataPath, mode, writeSet=False):
        boxSize= self.getBoxSize()
        inputMics = self.getInputMicrographs()
        if os.path.isfile(self._getExtraPath("consensus_%s.sqlite"%mode)):
            cleanPath( self._getExtraPath("consensus_%s.sqlite"%mode) )
        setOfCoordinates = SetOfCoordinates(filename= self._getExtraPath("consensus_%s.sqlite"%mode))
        setOfCoordinates.setMicrographs(inputMics)
        setOfCoordinates.setBoxSize( boxSize )
        # Read all consensus particles
        totalGoodCoords= 0
        for micrograph in inputMics:
            fnTmp = self._getExtraPath(newDataPath, 'consensus_%06d.txt' % micrograph.getObjId())
            if os.path.exists(fnTmp):
                coords = np.loadtxt(fnTmp)
                if coords.size == 2:  # special case with only one coordinate in consensus
                    coords = [coords]
                for coord in coords:
                    aux = Coordinate()
                    aux.setMicrograph(micrograph)
                    aux.setX(coord[0])
                    aux.setY(coord[1])
                    setOfCoordinates.append(aux)
                    totalGoodCoords+= 1
        assert totalGoodCoords> MIN_NUM_CONSENSUS_COORDS, ("Error, the consensus (%s) of your input coordinates was "+
            "too small (%d). It must be > %d. Try a different input" )%(mode, totalGoodCoords, MIN_NUM_CONSENSUS_COORDS)
        self.coordinatesDict[mode]= setOfCoordinates        
        if writeSet:
            setOfCoordinates.write()
        
    def pickNoise(self):
        outputRoot= self._getExtraPath( "noiseCoords")
        makePath(outputRoot)
        if not "OR" in self.coordinatesDict: # fill self.coordinatesDict['OR'] 
            self.loadConsensusCoords(
                                    XmippProtScreenDeepConsensus.CONSENSUS_COOR_PATH_TEMPLATE%'OR', 
                                    'OR', writeSet=False)
        pickWorker, argsList= pickAllNoiseWorker( self.coordinatesDict['OR'] , outputRoot, -1)
        
        Parallel(n_jobs= self.numberOfThreads.get(), backend="multiprocessing", verbose=1)(
                delayed(pickWorker, check_pickle=False)( *arg )  for arg in argsList)
        
        noiseCoords= writeSetOfCoordsFromFnames( self._getExtraPath("noiseCoords"), self.coordinatesDict['OR'],
                                                self._getExtraPath("consensus_NOISE.sqlite") )
        self.coordinatesDict["NOISE"]= noiseCoords    

    def insertMicsPreprocSteps(self, micIds):         
        boxSize = self.getBoxSize()
        self.downFactor = boxSize /float(DEEP_PARTICLE_SIZE)
        mics_= self.getInputMicrographs()

        if self.doFlip.get():
            setOfMicCtf= self.ctfRelations.get()            
            assert not setOfMicCtf is None, "Error, set of CTFs must be provided if doFlip desired"
            self.micToCtf={}
            micsFnameSet= set( [ mic.getMicName() for mic in mics_])
            for ctf in setOfMicCtf:
                ctf_mic= ctf.getMicrograph()
                ctfMicName=ctf_mic.getMicName()
                if ctfMicName in micsFnameSet:
                    self.micToCtf[ctfMicName]= ctf_mic
                    ctf_mic.setCTF(ctf)
            
        samplingRate = self.getInputMicrographs().getSamplingRate()
        for micId in micIds:
            mic= mics_[micId]            
            fnMic = mic.getFileName()
            self._insertFunctionStep("preprocessStep", micId, fnMic, samplingRate)

    def preprocessStep(self, micId, fnMic, samplingRate):
        downFactor= self.getDownFactor()
        fnPreproc = self._getExtraPath('preProcMics', os.path.basename(fnMic))
        if downFactor != 1: 
            args = "-i %s -o %s --step %f --method fourier" % (fnMic, fnPreproc, downFactor)
            self.runJob('xmipp_transform_downsample',args, numberOfMpi=1)
            fnMic= fnPreproc
            
        if self.doFlip.get():
             
            fnCTF = self._getTmpPath("%s.ctfParam" % os.path.basename(fnMic))
            micrographToCTFParam( self.micToCtf[ os.path.basename(fnMic) ], fnCTF)
            
            args = " -i %s -o %s --ctf %s --sampling %f"
            self.runJob('xmipp_ctf_phase_flip',
                        args % (fnMic, fnPreproc, fnCTF,
                                samplingRate*downFactor), numberOfMpi=1)
            fnMic= fnPreproc
            
        args = "-i %s -o %s --method OldXmipp "% (fnMic, fnPreproc)
        if self.doInvert.get():
            args+= "--invert"
        self.runJob('xmipp_transform_normalize',args, numberOfMpi=1)        
        
    def extractParticles(self, micIds, mode):
        boxSize = self.getBoxSize()
        downFactor= self.getDownFactor()

        try:
            coordSet= self.coordinatesDict[mode]
        except KeyError:
            coordSet= SetOfCoordinates(filename=self._getExtraPath('consensus_%s.sqlite'%mode))
        coordSet.setBoxSize(1)           
        mics_= self.getInputMicrographs()        
        coordSet.setMicrographs( mics_ )
        makePath( self._getExtraPath('coord_%s'%mode) )
        writePosFilesStepWorker(coordSet, self._getExtraPath('coord_%s'%mode))        
        makePath( self._getExtraPath('parts_%s'%mode) )
        for micId in micIds:   
            mic = mics_[micId]
            fnMic = mic.getFileName()
            fnMic = self._getExtraPath('preProcMics', os.path.basename(fnMic))
            fnPos= self._getExtraPath('coord_%s'%mode, pwutils.removeBaseExt(fnMic)+".pos") 
            if not os.path.isfile(fnPos): continue
            fnPos = "particles@"+fnPos
        
            outputStack = self._getExtraPath('parts_%s'%mode, pwutils.removeBaseExt(fnMic))
            args = " -i %s --pos %s" % (fnMic, fnPos)
            args += " -o %s --Xdim %d" % (outputStack, int( boxSize/downFactor))
            args += " --downsampling %f --fillBorders" %downFactor

            self.runJob("xmipp_micrograph_scissor", args, numberOfMpi=1)

    def joinSetOfParticles( self, micIds, mode) :

        #Create images.xmd metadata
        fnImages = self._getExtraPath("particles_%s.xmd"%mode)
        imgsXmd = MD.MetaData()
        posFiles = glob(self._getExtraPath("coord_%s"%mode, '*.pos'))
        for posFn in posFiles:
            xmdFn = self._getExtraPath("parts_%s"%mode, pwutils.replaceBaseExt(posFn, "xmd"))
            if os.path.exists(xmdFn):
                mdFn = MD.MetaData(xmdFn)
                mdPos = MD.MetaData('particles@%s' % posFn)
                mdPos.merge(mdFn) 
                imgsXmd.unionAll(mdPos)
            else:
                self.warning("The coord file %s wasn't used for extraction! "
                             % os.path.basename(posFn))
                self.warning("Maybe you are extracting over a subset of "
                             "micrographs")
        imgsXmd.write(fnImages)                
    
    def trainCNN(self):
        netDataPath=self._getExtraPath("nnetData")
        makePath(netDataPath)
        posTrainDict= { self._getExtraPath("particles_AND.xmd"):  1 }
        negTrainDict= { self._getExtraPath("particles_NOISE.xmd"):  1 }

        if hasattr(self, 'gpuToUse'):        
            numberOfThreads=None
            gpuToUse= self.gpuToUse.get()
        else:
            numberOfThreads=self.numberOfThreads.get()
            gpuToUse= None

        nEpochs= self.nEpochs.get()        
        if self.doContinue.get():
            prevRunPath= self.continueRun.get()._getExtraPath('nnetData')
            if not self.keepTraining.get():
                nEpochs=0
        else:
          prevRunPath= None

        trainWorker(netDataPath, posTrainDict, negTrainDict, nEpochs, self.learningRate.get(), 
                    self.l2RegStrength.get(), self.auto_stopping.get(), 
                    self.nModels.get(), gpuToUse, numberOfThreads, prevRunPath= prevRunPath) #TODO: check if works prevRunPath
        
    def predictCNN(self):
        netDataPath= self._getExtraPath('nnetData')
        if hasattr(self, 'gpuToUse'):        
            numberOfThreads=None
            gpuToUse= self.gpuToUse.get()
        else:
            numberOfThreads=self.numberOfThreads.get()
            gpuToUse= None 
            
        predictDict= { self._getExtraPath("particles_OR.xmd"):  1 }
        posTestDict= None
        negTestDict= None 
        outParticlesPath= self._getPath("particles.xmd")
        predictWorker( netDataPath, posTestDict, negTestDict, predictDict, outParticlesPath, 
                        gpuToUse, numberOfThreads)

    def createOutputStep(self):
        # PARTICLES
        cleanPattern( self._getPath("*.sqlite") )
        partSet = self._createSetOfParticles()
        readSetOfParticles(self._getPath("particles.xmd"), partSet)

        boxSize = self.getBoxSize()
        # COORDINATES
        if not "OR" in self.coordinatesDict:
            self.coordinatesDict['OR']= self.getConsensusCoordinates(
                                    XmippProtScreenDeepConsensus.CONSENSUS_COOR_PATH_TEMPLATE%'OR', 
                                    'OR', writeSet=False)

        coordSet = SetOfCoordinates(filename=self._getPath("coordinates.sqlite"))
        coordSet.copyInfo(self.coordinatesDict['OR'])
        coordSet.setBoxSize( boxSize )        
        coordSet.setMicrographs(self.coordinatesDict['OR'].getMicrographs())
        downFactor= self.getDownFactor()
        for part in partSet:
            coord = part.getCoordinate().clone()
            coord.scale( downFactor )
            setattr(  coord,        '_xmipp_%s' % xmipp.label2Str(MD.MDL_ZSCORE_DEEPLEARNING1),
                      getattr(part, '_xmipp_%s' % xmipp.label2Str(MD.MDL_ZSCORE_DEEPLEARNING1)) )           
            coordSet.append(coord)
        
        coordSet.write()
        partSet.write()
        
        self._defineOutputs(outputCoordinates=coordSet)        
#        self._defineOutputs(outputParticles=partSet)
        
        for inSetOfCoords in self.inputCoordinates:
            self._defineSourceRelation( inSetOfCoords.get(), coordSet)
        
        print(self.coordinatesDict['OR'].getBoxSize())
#        raise ValueError("peta")

#    def _getFirstJoinStepName(self):
#        # This function will be used for streaming, to check which is
#        # the first function that need to wait for all mics
#        # to have completed, this can be overriden in subclasses
#        # (e.g., in Xmipp 'sortPSDStep')
#        return 'createOutputStep'
#
#    def _getFirstJoinStep(self):
#        for s in self._steps:
#            if s.funcName == self._getFirstJoinStepName():
#                return s
#        return None
#
#
#    def _loadOutputSet(self, SetClass, baseName):
#        setFile = self._getPath(baseName)
#        if os.path.exists(setFile):
#            outputSet = SetClass(filename=setFile)
#            outputSet.enableAppend()
#        else:
#            outputSet = SetClass(filename=setFile)
#            outputSet.setStreamState(outputSet.STREAM_OPEN)
#
#        outputSet.setBoxSize(self.inputs.getBoxSize())
#        outputSet.setMicrographs(self.inputs.getMicrographs())
#        outputSet.copyInfo(self.inputs)
#        outputSet.copyItems(self.setOfCoords)
#        self.setOfCoords = []
#        return outputSet


#    def _updateOutputSet(self, outputName, outputSet, state=Set.STREAM_OPEN):
#        outputSet.setStreamState(state)
#        if self.hasAttribute(outputName):
#            outputSet.write()  # Write to commit changes
#            outputAttr = getattr(self, outputName)
#            # Copy the properties to the object contained in the protocol
#            outputAttr.copy(outputSet, copyId=False)
#            # Persist changes
#            self._store(outputAttr)
#        else:
#            self._defineOutputs(**{outputName: outputSet})
#            self._store(outputSet)
#
#        # Close set databaset to avoid locking it
#        outputSet.close()

    
    def _summary(self):
        message = []
        for i, coordinates in enumerate(self.inputCoordinates):
            protocol = self.getMapper().getParent(coordinates.get())
            message.append("Method %d %s" % (i + 1, protocol.getClassLabel()))
        message.append("Relative Radius = %f" % self.consensusRadius)
        message.append("Consensus = %d" % self.consensus)
        return message

    def _methods(self):
        return []


    #--------------------------- UTILS functions --------------------------------------------
    def _updateParticle(self, item, row):
        setXmippAttributes(item, row, MD.MDL_ZSCORE_DEEPLEARNING1)
        if row.getValue(MD.MDL_ENABLED) <= 0:
            item._appendItem = False
        else:
            item._appendItem = True