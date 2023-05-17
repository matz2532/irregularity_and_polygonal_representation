import numpy as np
import pickle

from copy import deepcopy
from shapely.geometry import MultiPoint, Point
from shapely.ops import nearest_points

class PointOrdererAlongOutline (object):

    allContours=None
    contourOffset=np.asarray([-1, -1])
    triWayJunctions=None
    verbosity=1

    def __init__(self, triWayJunctions=None, allContours=None, contourOffset=None):
        if not contourOffset is None:
            self.contourOffset = contourOffset
        if not allContours is None:
            self.SetAllContours(allContours)
        if not triWayJunctions is None:
            self.SetTriWayJunctions(triWayJunctions)

    def GetAllOrderedJunctions(self):
        return self.allOrderedJunctions

    def SetAllContours(self, allContours):
        self.allContours = allContours

    def SetTriWayJunctions(self, triWayJunctions):
        self.triWayJunctions = triWayJunctions

    def CalcAllOrderedJunctions(self, additionalJunctionsDict=None, **kwargs):
        assert not self.allContours is None and not self.triWayJunctions is None, f"Either the contours or tri-way junctions are not set, {self.allContours=} {self.triWayJunctions=}"
        self.allOrderedJunctions = {}
        for cell in self.allContours.keys():
            additionalJunctions = None
            if not additionalJunctionsDict is None:
                if cell in additionalJunctionsDict:
                    additionalJunctions = additionalJunctionsDict[cell]
            self.allOrderedJunctions[cell] = self.calcOrderedJunctionsOf(cell, additionalJunctions=additionalJunctions, **kwargs)

    def calcOrderedJunctionsOf(self, cell, additionalJunctions=None,
                               distanceThreshold=3, minAllowedNrOfJunctionsPerCell=3,
                               reverseJunctionOrder=True, flipOutlineCoordinates=False):
        contour = deepcopy(self.allContours[cell])
        if flipOutlineCoordinates:
            contour = np.flip(contour, axis=1)
        contour += self.contourOffset
        junctionsCloseToCellRect = self.findJunctionsCloseToCellRect(contour, distanceThreshold=distanceThreshold)
        if not additionalJunctions is None:
            junctionsCloseToCellRect = np.concatenate([junctionsCloseToCellRect, additionalJunctions], axis=0)
        allContourPointNearestToJunction = self.findUniqueContourPointsClosestToJunction(junctionsCloseToCellRect, contour)
        idxOfJunctionProjectedOnContour = []
        for contourPointNearestToJunction in allContourPointNearestToJunction:
            idxOfJunctionProjectedOnContour.append(self.findIdxOfPointOnList(contourPointNearestToJunction, contour))
        idxOfJunctionProjectedOnContour = np.sort(idxOfJunctionProjectedOnContour)
        if len(idxOfJunctionProjectedOnContour) >= minAllowedNrOfJunctionsPerCell:
            orderedJunctions = contour[idxOfJunctionProjectedOnContour, :]
            if reverseJunctionOrder:
                orderedJunctions = orderedJunctions[::-1]
            return orderedJunctions
        else:
            if self.verbosity >= 1:
                print("Cell contour {}, cell in labelled image {} is ignored as it has {} tri-way junctions in distance of {}, but needs at least {} to be included.".format(cell, cell+1, len(idxOfJunctionProjectedOnContour), distanceThreshold, minAllowedNrOfJunctionsPerCell))
            return None

    def findJunctionsCloseToCellRect(self, contour, distanceThreshold=3):
        min = np.min(contour, axis=0) - distanceThreshold
        max = np.max(contour, axis=0) + distanceThreshold
        isMin = np.all(self.triWayJunctions >= min, axis=1)
        isMax = np.all(self.triWayJunctions <= max, axis=1)
        isContourClose = isMin & isMax
        return self.triWayJunctions[isContourClose, :]

    def findUniqueContourPointsClosestToJunction(self, allJunctions, contour):
        allContourPointNearestToJunction = []
        for i, junction in enumerate(allJunctions):
            nearestPointOnContour, _ = nearest_points(MultiPoint(contour), Point(junction))
            contourPointNearestToJunction = np.asarray(list(nearestPointOnContour.coords))
            allContourPointNearestToJunction.append(contourPointNearestToJunction)
        allContourPointNearestToJunction = np.unique(allContourPointNearestToJunction, axis=0)
        return allContourPointNearestToJunction

    def findIdxOfPointOnList(self, point, pointArray):
        isPoint = np.all(point == pointArray, axis=1)
        idx = np.where(isPoint)[0]
        assert len(idx) > 0, "The point {} is not in the list of points {}".format(point, pointArray)
        if len(idx) > 0:
            return idx[0]

def saveOrderedJunctionsOf(folderContent, dataBaseFolder="", orderedJunctionsBaseName="orderedJunctionsPerCell.pkl"):
    triWayJunctions = folderContent.LoadKeyUsingFilenameDict("finalJunctionFilename")
    cellContourDict = folderContent.LoadKeyUsingFilenameDict("cellContours", convertDictKeysToInt=True, convertDictValuesToNpArray=True)
    if folderContent.IsKeyInFilenameDict("additionalJunctionsDict"):
        additionalJunctionsDict = folderContent.LoadKeyUsingFilenameDict("additionalJunctionsDict")
        additionalJunctionsDict = {int(cellAsString):junctions for cellAsString, junctions in additionalJunctionsDict.items()}
    else:
        additionalJunctionsDict = None
    myPointOrdererAlongOutline = PointOrdererAlongOutline(triWayJunctions=triWayJunctions, allContours=cellContourDict, contourOffset=np.asarray([0, 0]))
    myPointOrdererAlongOutline.CalcAllOrderedJunctions(additionalJunctionsDict=additionalJunctionsDict)
    orderedJunctionsPerCell = myPointOrdererAlongOutline.GetAllOrderedJunctions()
    dataFolder = dataBaseFolder + folderContent.GetFolder()
    orderedJunctionsPerCellFilename = dataFolder + orderedJunctionsBaseName
    with open(orderedJunctionsPerCellFilename, "wb") as fh:
        pickle.dump(orderedJunctionsPerCell, fh)
    folderContent.AddDataToFilenameDict(orderedJunctionsPerCellFilename, "orderedJunctionsPerCellFilename")

def main():
    import json
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, "./Code/DataStructures/")
    sys.path.insert(0, "./Code/Analyze/")
    from FolderContent import FolderContent
    from MultiFolderContent import MultiFolderContent
    from PatchCreator import PatchCreator
    allFolderContentsFilename = "Images/Eng2021Cotyledons.pkl"
    selectedReplicateId = "20180618 ktn1-2 S6"
    timePoint = "24h"
    myMultiFolderContent = MultiFolderContent(allFolderContentsFilename)
    folderContent = myMultiFolderContent.GetFolderContentOfReplicateAtTimePoint(selectedReplicateId, timePoint)
    labelledImageFilename = folderContent.LoadKeyUsingFilenameDict("labelledImageFilename")
    triWayJunctions = folderContent.LoadKeyUsingFilenameDict("finalJunctionFilename")
    cellContourDict = folderContent.LoadKeyUsingFilenameDict("cellContours", convertDictKeysToInt=True, convertDictValuesToNpArray=True)
    with open("Images/ktn1-2/20180618 ktn1-2 S6/24h/manuallyAddedJunctionPositionDict.json", "rb") as fh:
        file = json.load(fh)
    additionalJunctionsDict = {int(cellAsString):junctions for cellAsString, junctions in file.items()}
    myPointOrdererAlongOutline = PointOrdererAlongOutline(triWayJunctions=triWayJunctions, allContours=cellContourDict, contourOffset=np.asarray([0, 0]))
    myPointOrdererAlongOutline.CalcAllOrderedJunctions(additionalJunctionsDict=additionalJunctionsDict)
    orderedJunctions = myPointOrdererAlongOutline.GetAllOrderedJunctions()
    # viusalise junctions
    myPatchCreator = PatchCreator()
    fig, ax = plt.subplots(1, 1)
    ax.imshow(labelledImageFilename)
    myPatchCreator.PlotPatchesFromOutlineDictOn(ax, orderedJunctions)
    plt.show()

if __name__ == '__main__':
    main()
