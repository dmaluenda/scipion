# **************************************************************************
# *
# * Authors:     David Maluenda (dmaluenda@cnb.csic.es)
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

from pyworkflow.em.protocol.protocol import EMProtocol
from pyworkflow.protocol.params import MultiPointerParam
from pyworkflow.em import Class3D

class XmippProtConsensusClasses3D(EMProtocol):
    """ Compare several SetOfClasses3D.
        Return the intersection of the input classes.
    """
    _label = 'consensus classes 3D'

    intersectsList = []

    def _defineParams(self, form):
        form.addSection(label='Input')

        form.addParam('inputMultiClasses', MultiPointerParam, important=True,
                      label="Input Classes", pointerClass='SetOfClasses3D',
                      help='Select several sets of classes where '
                           'to evaluate the intersections.')

    # --------------------------- INSERT steps functions -----------------------
    def _insertAllSteps(self):
        """ Inserting one step for each intersections analisis
        """

        self._insertFunctionStep('compareFirstStep', 
                                 self.inputMultiClasses[0].get().getObjId(),
                                 self.inputMultiClasses[1].get().getObjId())

        if len(self.inputMultiClasses)>2:
            for i in range(2,len(self.inputMultiClasses)):
                self._insertFunctionStep('compareOthersStep', i,
                                     self.inputMultiClasses[i].get().getObjId())

        self._insertFunctionStep('computeInitDistancesStep')

        self._insertFunctionStep('createOutputStep')

    def compareFirstStep(self, objId1, objId2):
        """ We found the intersections for the two firsts sets of classes
        """
        set1Id = 0
        set2Id = 1
        set1 = self.inputMultiClasses[set1Id].get()
        set2 = self.inputMultiClasses[set2Id].get()

        print('Computing intersections between classes form set %s and set %s:'
               % (set1.getNameId(), set2.getNameId()))

        newList = []
        for cls1 in set1:
            cls1Id = cls1.getObjId()
            ids1 = cls1.getIdSet()
            rep1 = (set1Id, cls1Id)

            for cls2 in set2:
                cls2Id = cls2.getObjId()
                ids2 = cls2.getIdSet()
                rep2 = (set2Id, cls2Id)

                interTuple = self.intersectClasses(rep1, ids1, rep2, ids2)

                newList.append(interTuple)

        self.intersectsList = newList

    def compareOthersStep(self, set1Id, objId):
        """ We found the intersections for the two firsts sets of classes
        """
        set1 = self.inputMultiClasses[set1Id].get()

        print('Computing intersections between classes form set %s and '
              'the previous ones:' % (set1.getNameId()))

        newList = []
        currDB = self.intersectsList
        for cls1 in set1:
            cls1Id = cls1.getObjId()
            ids1 = cls1.getIdSet()
            rep1 = (set1Id, cls1Id)
            
            for currTuple in currDB:
                ids2 = currTuple[1]
                parents = currTuple[2]
                cl2Size = currTuple[3]
                rep2 = currTuple[4]
                print " "
                print parents
                interTuple = self.intersectClasses(rep1, ids1, rep2, ids2, 
                                                   parents, cl2Size)

                newList.append(interTuple)
                
        self.intersectsList = newList

    def computeInitDistancesStep(self):
        """ We compute the distances and hierarchiraly reclassificate the classes
        """
        conClasses = self.intersectsList
        numInput = len(self.inputMultiClasses)  # P
        numOutput = len(conClasses)
        d = [[0 for x in range(numOutput)] for y in range(numOutput)]
        for i in range(0, numOutput):
            for j in range(i+1, numOutput):

                distCum = 0 # initialize
                for p, inputClass in enumerate(self.inputMultiClasses):
                    dist=0  # initialize
                    numClasses = inputClass.get().getSize()  # Np
                    for k in range(0, numClasses):
                        # print i, j, p, k
                        # print conClasses[0][2][p]
                        di = int(k in conClasses[i][2][p])
                        dj = int(k in conClasses[j][2][p])
                        dist += abs(float(di-dj))/numClasses
                        # print di

                    distCum += dist

                d[i][j] = distCum/numInput
                print('d(%d,%d) = %f' %(i,j,d[i][j]))


    def createOutputStep(self):

        # self.intersectsList.sort(key=lambda e: e[0], reverse=True)

        # print("   ---   S O R T E D:   ---")
        # numberOfPart = 0
        # for classItem in self.intersectsList:
        #     printTuple = (classItem[0], classItem[2])
        #     print(printTuple)
        #     numberOfPart += classItem[0]

        # print('Number of intersections: %d' % len(self.intersectsList))
        # print('Total of particles: %d' % numberOfPart)


        inputParticles = self.inputMultiClasses[0].get().getImages()
        outputClasses = self._createSetOfClasses3D(inputParticles)

        for classItem in self.intersectsList:
            numOfPart = classItem[0]
            partIds = classItem[1]
            setRepId = classItem[4][0]
            clsRepId = classItem[4][1]

            setRep = self.inputMultiClasses[setRepId].get()
            clRep = setRep[clsRepId]

            newClass = Class3D()
            # newClass.copyInfo(clRep)  # It give problems when setReps are of diff. tipe (Xmipp, Relion...)
            newClass.setAcquisition(clRep.getAcquisition())
            newClass.setRepresentative(clRep.getRepresentative())

            outputClasses.append(newClass)

            enabledClass = outputClasses[newClass.getObjId()]
            enabledClass.enableAppend()
            for itemId in partIds:
                enabledClass.append(inputParticles[itemId])

            outputClasses.update(enabledClass)

        self._defineOutputs(outputClasses=outputClasses)
        for item in self.inputMultiClasses:
            self._defineSourceRelation(item, outputClasses)


    # --------------------------- INFO functions -------------------------------
    def _summary(self):
        summary = []
        return summary

    def _methods(self):
        methods = []
        return methods

    def _validate(self):
        errors = [] if len(self.inputMultiClasses)>1 else \
                 ["More than one Input Classes is needed to compute the consensus."]
        return errors


    # --------------------------- UTILS functions ------------------------------
    def intersectClasses(self, rep1, ids1, rep2, ids2, parentsDict={}, clsSize2=None):
        """ Computes the intersection of ids1 and ids2.
            Assign as rep those from the small class.
            clsSize2 is used in the iterative steps.
              It save the original size of the rep2 class
              It's used to know if we are dealing with a first step or the others
            parentsDict is the information from where intersection proceds
        """
        size1 = len(ids1)
        size2 = len(ids2) if clsSize2 is None else clsSize2

        inter = ids1.intersection(ids2)

        if size1 < size2:
            rep = rep1
            clsSize = size1
        else:
            rep = rep2
            clsSize = size2

        if clsSize2 is None:
            parentsDict = {}
            if rep2[0] in parentsDict.keys():
                parentsDict[rep2[0]].append(rep2[1])
            else:
                parentsDict[rep2[0]] = [rep2[1]]

        if rep1[0] in parentsDict.keys():
            parentsDict[rep1[0]].append(rep1[1])
        else:
            parentsDict[rep1[0]] = [rep1[1]]

        print(" ")
        print(" - Intersection of cl%d of set%d (%d part.) and "
                                 "cl%d of set%d (%d part.):"
               % (rep1[1], rep1[0], len(ids1), rep2[1], rep2[0], len(ids2)))
        print("    Size1=%d < Size2=%d = %s" 
               % (size1, size2, size1<size2))
        print("      -> from set %d calss %d, with %d parts. in the intersection." 
               % (rep[0], rep[1], len(inter)))
        print "parents: ", parentsDict
        print(" -  -  -  -  -  -  -  -  -  -")

        return (len(inter), inter, parentsDict, clsSize, rep)
