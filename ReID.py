import cv2
from FeatureExtract import FeatureExtract
from Object import *

GOOD_MATCHES_THRESHOLD = 5

class ReID:

    def __init__(self):
        self.featureExtract = FeatureExtract('sift-flann')
        self.detectedObjects = []
        self.externalDetectedObjects = []
        self.newDetectedObjects = []

    def GetDetectedObjects(self):
        return self.detectedObjects

    def SetExternalDetectedObjects(self, externalDetectedObjects):
        self.externalDetectedObjects = externalDetectedObjects

    def DeleteDetectedObject(self, id):
        for i in range(len(self.detectedObjects)):
            if self.detectedObjects[i].id == id:
                self.detectedObjects.pop(i)
                break
        for i in range(len(self.newDetectedObjects)):
            if self.newDetectedObjects[i].id == id:
                self.newDetectedObjects.pop(i)
                break

    def CheckNewObject(self, ROI, id, classLabel):
        retrievedId = -1 # -1 -> still new, -2 -> become normal unidentified, other -> identified
        index = -1
        color = None
        for i in range(len(self.newDetectedObjects)):
            if self.newDetectedObjects[i].id == id:
                index = i
                break
        
        if index != -1:
            if self.newDetectedObjects[index].newObjectCounter > 0:
                objectToReturn, isNew = self.ObjectReID(ROI, classLabel, True)
                if isNew:
                    self.newDetectedObjects[index].description.extend(objectToReturn.description)
                    self.newDetectedObjects[index].newObjectCounter -= 1
                else:
                    retrievedId = objectToReturn.id
                    color = objectToReturn.color
                    self.newDetectedObjects.pop(index)
            else:
                self.detectedObjects.append(self.newDetectedObjects[index])
                self.newDetectedObjects.pop(index)
                retrievedId = -2

        return retrievedId, color

    def UpdateObjectDescriptors(self, ROI, id):
        index = -1
        for i in range(len(self.detectedObjects)):
            if self.detectedObjects[i].id == id:
                index = i
                break
        bodyDes, bodyKp, bodyKpOriginal = self.featureExtract.Detect(ROI)
        if len(bodyDes) != 0:
            if len(self.detectedObjects[index].description) >= 50:
                self.detectedObjects[index].description.pop(0)
            self.detectedObjects[index].description.append((bodyDes, bodyKp, ROI))
            return True
        return False

    def ObjectReID(self, ROI, classLabel, isCheck = False):
        goodMatchesTh = GOOD_MATCHES_THRESHOLD
        newMatches = None
        objectToReturn = None
        index = -1
        dscIndex = -1
        isExternal = False
        isNew = False
        finalKps = []

        bodyDes, bodyKp, bodyKpOriginal = self.featureExtract.Detect(ROI)

        if(len(bodyDes) != 0):

            if index == -1:
                for n in range(len(self.externalDetectedObjects)):
                    if classLabel == self.externalDetectedObjects[n].classLabel:
                        for descriptionIndex in range(len(self.externalDetectedObjects[n].description)):
                            try:
                                upperBodyMatches = self.featureExtract.matcher.knnMatch(self.externalDetectedObjects[n].description[descriptionIndex][0], bodyDes, k=2)

                                finalKpoints = []

                                for keyPoint in self.externalDetectedObjects[n].description[descriptionIndex][1]:
                                    tempKp = cv2.KeyPoint(x=keyPoint[0][0], y=keyPoint[0][1], size=float(keyPoint[1]), angle=keyPoint[2], response=keyPoint[3], octave=keyPoint[4], class_id=keyPoint[5])
                                    finalKpoints.append(tempKp)

                                matchesMask, matches = self.featureExtract.ransacFilter(upperBodyMatches, bodyKpOriginal, finalKpoints)
                                if matchesMask.count(1) > goodMatchesTh:
                                    finalKps = finalKpoints
                                    goodMatchesTh = matchesMask.count(1)
                                    newMatches = matches
                                    dscIndex = descriptionIndex
                                    upperBodyFit = matchesMask
                                    isExternal = True
                                    index = n
                            except:
                                pass

            if index == -1: 
                for l in range(len(self.detectedObjects)):
                    if classLabel == self.detectedObjects[l].classLabel:
                        for descriptionIndex2 in range(len(self.detectedObjects[l].description)):
                            try:
                                upperBodyMatches = self.featureExtract.matcher.knnMatch(self.detectedObjects[l].description[descriptionIndex2][0], bodyDes, k=2)
                            
                                finalKpoints = []

                                for keyPoint in self.detectedObjects[l].description[descriptionIndex2][1]:
                                    tempKp = cv2.KeyPoint(x=keyPoint[0][0], y=keyPoint[0][1], size=float(keyPoint[1]), angle=keyPoint[2], response=keyPoint[3], octave=keyPoint[4], class_id=keyPoint[5])
                                    finalKpoints.append(tempKp)

                                matchesMask, matches = self.featureExtract.ransacFilter(upperBodyMatches, bodyKpOriginal, finalKpoints)

                                if matchesMask.count(1) > goodMatchesTh:
                                    finalKps = finalKpoints
                                    goodMatchesTh = matchesMask.count(1)
                                    newMatches = matches
                                    dscIndex = descriptionIndex2
                                    upperBodyFit = matchesMask
                                    index = l
                            except:
                                pass

        if index != -1:
            if isExternal:
                tempObject = self.externalDetectedObjects[index]

                drawParams = dict(matchColor = (0, 0, 255), singlePointColor = (255, 0, 0), matchesMask = upperBodyFit, flags = 2)
                matchesImage = cv2.drawMatches(tempObject.description[dscIndex][2], finalKps, ROI, bodyKpOriginal, newMatches, None, **drawParams)
                cv2.imshow(f'Good Matches External - {tempObject.id}', matchesImage)

                if len(tempObject.description) >= 50:
                    tempObject.description.pop(0)
                tempObject.description.append((bodyDes, bodyKp, ROI))
                objectToReturn = tempObject
            else:
                drawParams = dict(matchColor = (0, 0, 255), singlePointColor = (255, 0, 0), matchesMask = upperBodyFit, flags = 2)
                matchesImage = cv2.drawMatches(self.detectedObjects[index].description[dscIndex][2], finalKps, ROI, bodyKpOriginal, newMatches, None, **drawParams)
                cv2.imshow(f'Good Matches - {self.detectedObjects[index].id}', matchesImage)

                if len(self.detectedObjects[index].description) >= 50:
                    self.detectedObjects[index].description.pop(0)
                self.detectedObjects[index].description.append((bodyDes, bodyKp, ROI))
                objectToReturn = self.detectedObjects[index]
        else:
            object = ObjectReID(classLabel, None, (bodyDes, bodyKp, ROI))
            if not isCheck:
                self.newDetectedObjects.append(object)
            objectToReturn = object
            isNew = True

        return objectToReturn, isNew
