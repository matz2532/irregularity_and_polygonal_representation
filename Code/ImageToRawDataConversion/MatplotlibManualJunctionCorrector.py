import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
import tkinter as tk
import skimage.morphology as mp
import sys

sys.path.insert(0, "./Code/")
sys.path.insert(0, "./Code/Analyze/")
sys.path.insert(0, "./Code/ImageToRawDataConversion/")
from copy import deepcopy
from JunctionCreator import JunctionCreator
from matplotlib.widgets import CheckButtons
from mpl_interactions import zoom_factory, panhandler
from mpl_point_clicker import clicker
from PatchCreator import PatchCreator
from pathlib import Path
from PointOrdererAlongOutline import PointOrdererAlongOutline
from shapely.geometry import MultiPoint, Point
from shapely.ops import nearest_points

class MatplotlibManualJunctionCorrector (JunctionCreator):

    allowedClippingDistance=5
    baseJunctionName="correctedTriWayJunctions.npy"
    colorBar=None
    guardCellArtist=None
    guardCellClassName="guard cell"
    isContourDrawn=False
    junctionClassName="junctions"
    junctionColor="red"
    verbosity=1

    def __init__(self, originalImage, labelledImage, cellContourDict=None,
                 skeletonImage=None, triWayJunctions=None, guardCellJunctionPositions=None,
                 folderToSave="", title=None, **kwargs):
        self.originalImage = originalImage
        self.labelledImage = labelledImage
        self.cellContourDict = cellContourDict
        self.folderToSave = folderToSave
        self.title = title
        self.patchCreator = PatchCreator()
        self.junctionOrderer = PointOrdererAlongOutline(triWayJunctions, cellContourDict, contourOffset=np.asarray([0, 0]))
        assert np.all(self.originalImage.shape[:2] == self.labelledImage.shape[:2]), "The x and y sizes of the originalImage and labelledImage are not the same, {} != {}".format(self.originalImage.shape[:2], self.labelledImage.shape[:2])
        self.canvasSize = np.asarray(self.originalImage.shape[:2])
        if skeletonImage is None or triWayJunctions is None:
            super().__init__(self.labelledImage, **kwargs)
        if not triWayJunctions is None:
            self.alreadyKnownTriWayJunctions = np.flip(triWayJunctions, axis=1)
        else:
            self.alreadyKnownTriWayJunctions = []
        if not guardCellJunctionPositions is None:
            if len(guardCellJunctionPositions.shape) == 1:
                guardCellJunctionPositions = guardCellJunctionPositions
            elif len(guardCellJunctionPositions.shape) == 2:
                guardCellJunctionPositions = np.flip(guardCellJunctionPositions, axis=1)
            else:
                print(f"Not allowed shape of {guardCellJunctionPositions=}")
                sys.exit()
            self.guardCellJunctionPositions = list(guardCellJunctionPositions)
        else:
            self.guardCellJunctionPositions = []
        if not skeletonImage is None:
            self.skeletonImage = skeletonImage
        else:
            if not self.cellContourDict is None:
                self.skeletonImageFromContour = self.createSkeletonImageFromContour(self.cellContourDict)
                self.skeletonImageFromLabelledImg = self.createSkeletonImageFrom(self.labelledImage)
                self.skeletonImage = np.zeros_like(self.skeletonImageFromContour)
                dilatedImg = mp.dilation(self.skeletonImageFromContour, np.ones((3, 3)))
                self.skeletonImage[np.logical_and(self.skeletonImageFromLabelledImg==1, dilatedImg==1)] = 1
            else:
                self.skeletonImage = self.createSkeletonImageFrom(self.labelledImage)
        # The kernel assumes 4-connectivity of the skeleton image
        kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
        self.skeletonHeatImage = cv2.filter2D(self.skeletonImage, -1, kernel)
        skeletonImagePoints = np.asarray(np.where(self.skeletonImage==1)).T
        xYAdaptedSkeletonImagePoints = np.flip(skeletonImagePoints, axis=1)
        self.skeletonImgMultiPoint = MultiPoint(xYAdaptedSkeletonImagePoints)
        self.initWindow()

    def SaveGuardCellJunctionPositions(self, filenameToSaveGuardCellJunctions=None, printSaving=False, flipXYCoordinates=True):
        if filenameToSaveGuardCellJunctions is None:
            filenameToSaveGuardCellJunctions = self.folderToSave + self.baseJunctionName
        if printSaving:
            print("Saving junctions to {}".format(filenameToSaveGuardCellJunctions))
        guardCellJunctionPositions = self.guardCellJunctionPositions.copy()
        if len(self.guardCellJunctionPositions) > 0:
            if flipXYCoordinates:
                guardCellJunctionPositions = np.flip(guardCellJunctionPositions, axis=1)
        np.save(filenameToSaveGuardCellJunctions, guardCellJunctionPositions)
        return filenameToSaveGuardCellJunctions

    def SaveJunctions(self, filenameToSaveJunctions=None, printSaving=False, flipXYCoordinates=True):
        if filenameToSaveJunctions is None:
            filenameToSaveJunctions = self.folderToSave + self.baseJunctionName
        if printSaving:
            print("Saving junctions to {}".format(filenameToSaveJunctions))
        allKnownJunctionList = self.allKnownJunctionList.copy()
        if len(self.allKnownJunctionList) > 0:
            if flipXYCoordinates:
                allKnownJunctionList= np.flip(allKnownJunctionList, axis=1)
        np.save(filenameToSaveJunctions, allKnownJunctionList)
        return filenameToSaveJunctions

    def createSkeletonImageFromContour(self, contourDict):
        skeletonImage = np.zeros(self.canvasSize)
        for cellLabel, cellContour in contourDict.items():
            skeletonImage[cellContour[:, 0], cellContour[:, 1]] = 1
        return skeletonImage

    def createSkeletonImageFrom(self, labelledImage, cellsToIgnore=[0]):
        skeletonImage = np.zeros_like(labelledImage)
        validCells = np.unique(labelledImage)
        validCells = validCells[np.isin(validCells, cellsToIgnore, invert=True)]
        for cellId in validCells:
            contour = self.calcContourFromLabelledImage(labelledImage, cellId)
            skeletonImage[contour[:, 0], contour[:, 1]] = 1
        return skeletonImage

    def calcContourFromLabelledImage(self, labelledImage, cellId):
        img = np.zeros_like(labelledImage)
        img[np.where(labelledImage == cellId)] = 1
        dilatedImg = mp.dilation(img, np.ones((3, 3)))
        contourImg = dilatedImg - img
        contour = np.asarray(np.where(contourImg == 1)).T
        return contour

    def initWindow(self):
        self.fig, self.ax = plt.subplots(constrained_layout=True)
        if not self.title is None:
            self.ax.set_title(self.title)
        self.backgroundImg = self.createBackgroundImg()
        self.ax.imshow(self.backgroundImg, cmap="gray")
        zoom_factory(self.ax)
        ph = panhandler(self.fig, button=2)
        self.klicker = clicker(self.ax, [self.junctionClassName, self.guardCellClassName], markers=["o", "$G$"])
        self.addKnownJunctionsToClicker()
        self.addOrRemoveCellularOutline()
        self.drawPolygonalCellRepresentation()
        self.connectguardCellJunctionPositions()
        self.klicker.on_point_added(self.addManualSelectedJunction)
        self.klicker.on_point_removed(self.removeManualSelectedJunction)
        self.klicker.on_point_added(self.addGuardCellJunction)
        plt.show()

    def createBackgroundImg(self, useOriginalImage=True, rotateAndFlipImage=False, minMaxTransform=True):
        if useOriginalImage:
            img = self.originalImage
        else:
            img =  self.labelledImage
        if rotateAndFlipImage:
            img = self.rotateAndFlipImage(img)
        if minMaxTransform:
            img = self.minMaxTransform(img)
        return img

    def rotateAndFlipImage(self, img):
        img = np.rot90(img)
        img = np.flip(img, axis=0)
        return img

    def minMaxTransform(self, img):
        img = np.array(img)
        img -= np.min(img)
        img = img.astype(float) / (np.max(img) - np.min(img))
        img *= 255
        return img

    def addKnownJunctionsToClicker(self):
        self.allKnownJunctionList = list(deepcopy(self.alreadyKnownTriWayJunctions))
        self.klicker._positions[self.junctionClassName] = self.allKnownJunctionList
        self.klicker._update_points(self.junctionClassName)

    def addOrRemoveCellularOutline(self):
        if not self.isContourDrawn:
            self.drawCellularOutline()
        else:
            self.ax.collections.pop()
        self.isContourDrawn = not self.isContourDrawn

    def drawCellularOutline(self):
        if not self.cellContourDict is None:
            self.patchCreator.PlotPatchesFromOutlineDictOn(self.ax, self.cellContourDict, faceColor="#00000020")

    def connectguardCellJunctionPositions(self):
        self.klicker._positions[self.guardCellClassName] = self.guardCellJunctionPositions
        self.klicker._update_points(self.guardCellClassName)

    def drawPolygonalCellRepresentation(self):
        triWayJunctions = np.asarray(self.allKnownJunctionList)
        self.junctionOrderer.SetTriWayJunctions(triWayJunctions)
        self.junctionOrderer.CalcAllOrderedJunctions(flipOutlineCoordinates=True)
        orderedJuntionsPerCell = self.junctionOrderer.GetAllOrderedJunctions()
        vertexNumberPerCell = {cellLabel:0 if junctionPositions is None else len(junctionPositions) for cellLabel, junctionPositions in orderedJuntionsPerCell.items()}
        self.polygonArtist = self.patchCreator.PlotPatchesFromOutlineDictOn(self.ax, orderedJuntionsPerCell,
                    faceColor=vertexNumberPerCell, alpha=0.5, edgecolor=None,
                    removeColorBar=self.colorBar, flipOutlineCoordinates=False)
        self.colorBar = self.patchCreator.GetColorBar()

    def addManualSelectedJunction(self, position, klass):
        if klass != self.junctionClassName:
            return None
        nearestPointOnSkeletonImg = nearest_points(self.skeletonImgMultiPoint, Point(position))[0] # get the nearest point of the first geometry to the second geometry
        nearestPointOnSkeletonImg = np.asarray(nearestPointOnSkeletonImg, dtype=np.int64)
        junctionPosition = self.clipPointToHighestSekeletonHeatMap(nearestPointOnSkeletonImg, self.allowedClippingDistance)
        if self.isPointPresentInList(junctionPosition, self.allKnownJunctionList[:-1]):
            if self.isPointPresentInList(nearestPointOnSkeletonImg, self.allKnownJunctionList[:-1]):
                position = np.asarray(position, dtype=np.int64)
                if self.isPointPresentInList(position, self.allKnownJunctionList[:-1]):
                    if self.verbosity >= 1:
                        print(f"The click was ignored as the estimated position={junctionPosition}, the nearest point to the skeleton/lamina={nearestPointOnSkeletonImg} and the clicked position={position} are already present.")
                    last = self.allKnownJunctionList.pop()
                    return
                else:
                    junctionPosition = position
            else:
                junctionPosition = nearestPointOnSkeletonImg
        self.allKnownJunctionList[-1] = junctionPosition
        self.klicker._update_points(self.junctionClassName)
        self.updateManualSelectedJunctions()

    def clipPointToHighestSekeletonHeatMap(self, point, clippingDistance: int):
        fromTopLeft = np.asarray(point, dtype=int) - clippingDistance
        toBottomRight = np.asarray(point, dtype=int) + clippingDistance
        if fromTopLeft[0] < 0:
            fromTopLeft[0] = 0
        if fromTopLeft[1] < 0:
            fromTopLeft[1] = 0
        if toBottomRight[0] >= self.canvasSize[0]:
            toBottomRight[0] = self.canvasSize[0] - 1
        if toBottomRight[1] >= self.canvasSize[1]:
            toBottomRight[1] = self.canvasSize[1] - 1
        heatMapSelection = self.skeletonHeatImage[fromTopLeft[1]:toBottomRight[1], fromTopLeft[0]:toBottomRight[0]]
        localMaxima = np.flip(np.asarray(mp.local_maxima(heatMapSelection, indices=True)).T, axis=1)
        localMaxima += fromTopLeft
        if len(localMaxima) > 1:
            if np.any(np.all(point == localMaxima, axis=1)):
                return point
        elif len(localMaxima) == 0:
            return point
        return localMaxima[0]

    def isPointPresentInList(self, point, pointRefList):
        for refferencePoint in pointRefList:
            if list(point) == list(refferencePoint):
                return True
        return False

    def updateManualSelectedJunctions(self):
        self.polygonArtist.remove()
        self.drawPolygonalCellRepresentation()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def removeManualSelectedJunction(self, position, klass, removedIdx):
        self.updateManualSelectedJunctions()

    def addGuardCellJunction(self, position, klass, maxDistThreshold=1):
        if klass != self.guardCellClassName:
            return None
        distanceToJunctions = np.linalg.norm(self.allKnownJunctionList - np.asarray(position), axis=1)
        guardCellJunctionPosition = self.allKnownJunctionList[np.argmin(distanceToJunctions)]
        self.guardCellJunctionPositions[-1] = guardCellJunctionPosition
        distanceToOtherGuardCells = np.linalg.norm(self.guardCellJunctionPositions - guardCellJunctionPosition, axis=1)
        if np.sum(distanceToOtherGuardCells < maxDistThreshold) > 1:
            self.guardCellJunctionPositions.pop()
        self.klicker._update_points(self.guardCellClassName)

def main():
    import sys
    sys.path.insert(0, "./Code/DataStructures/")
    from FolderContent import FolderContent
    from MultiFolderContent import MultiFolderContent
    allFolderContentsFilename = "Images/Eng2021Cotyledons.pkl"
    selectedReplicateId = "20170501 WT S1"
    timePoint = "72h"
    myMultiFolderContent = MultiFolderContent(allFolderContentsFilename)
    folderContent = myMultiFolderContent.GetFolderContentOfReplicateAtTimePoint(selectedReplicateId, timePoint)
    filenameDict = folderContent.GetFilenameDict()
    labelledImage = folderContent.LoadKeyUsingFilenameDict("labelledImageFilename")
    originalImage = folderContent.LoadKeyUsingFilenameDict("originalImageFilename")
    cellContourDict = folderContent.LoadKeyUsingFilenameDict("cellContours", convertDictKeysToInt=True, convertDictValuesToNpArray=True)
    triWayJunctionPositions = folderContent.LoadKeyUsingFilenameDict("correctedTriWayJunctionFilenames")
    guardCellJunctionPositions = folderContent.LoadKeyUsingFilenameDict("guardCellJunctionPositions")
    # filenameDict = {'originalImageFilename': 'Images\\col-0\\20170327 WT S1\\0h\\PM_LTi6b-GFP mChTUA5 WT 0h S1_SMP-based (3) MIP.TIF', 'contourFilename': 'Images\\col-0\\20170327 WT S1\\0h\\Cell_Contours_33_Cells -- LTi6b-GFP mChTUA5 WT 0h S1_.txt', 'labelledImageFilename': 'Images\\col-0\\20170327 WT S1\\0h\\F_BOREALIS_MTs_LTi6b-GFP mChTUA5 WT 0h S1_SMP-based (3) MIP_WATERSHED_CLEAN.TIF', 'visibilityGraphs': 'Images\\col-0\\20170327 WT S1\\0h\\visibilityGraphs.gpickle', 'cellContours': 'Images\\col-0\\20170327 WT S1\\0h\\cellContour.pickle', 'correctedTriWayJunctionFilenames': 'Images\\col-0\\20170327 WT S1\\0h\\correctedTriWayJunctions.npy', 'finalJunctionFilename': 'Images/col-0/20170327 WT S1/0h/correctedTriWayJunctions.npy', 'correspondingCellsFilename': 'Images/col-0/20170327 WT S1/0h/correctedCorresponding.json', 'cellContourFilename': 'Images/col-0/20170327 WT S1/0h/cellContour.pickle', 'orderedJunctionsPerCellFilename': 'Images/col-0/20170327 WT S1/0h/orderedJunctionsPerCell.pkl', 'regularityMeasuresFilename': 'Images/col-0/20170327 WT S1/0h/regularityMeasures.pkl', 'areaMeasuresPerCell': 'Images/col-0/20170327 WT S1/0h/areaMeasuresPerCell.pkl'}
    # folderContent = FolderContent(None)
    # labelledImage = folderContent.loadFile(filenameDict["labelledImageFilename"])
    # originalImage = folderContent.loadFile(filenameDict["originalImageFilename"])
    # cellContourDict = folderContent.loadFile(filenameDict["cellContours"])
    # triWayJunctionPositions = folderContent.loadFile(filenameDict["correctedTriWayJunctionFilenames"])
    skeletonImage = None
    myManualJunctionCorrectorTk = MatplotlibManualJunctionCorrector(originalImage, labelledImage,
                                                                    cellContourDict, skeletonImage,
                                                                    triWayJunctions=triWayJunctionPositions,
                                                                    guardCellJunctionPositions=guardCellJunctionPositions)

if __name__ == '__main__':
    main()
