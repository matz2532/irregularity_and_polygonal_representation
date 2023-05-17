import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import warnings

from FolderContent import FolderContent
# from MultiFolderContent import MultiFolderContent
from pathlib import Path

class CellIdTracker (object):

    labels=None
    ids=None
    linesOfContourFile=None
    headerOffSet=4
    globalVerbosity=1

    def __init__(self, contourFile=None, folderContent=None, asArray=True):
        self.asArray = asArray
        if not contourFile is None or not folderContent is None:
            self.RunCellIdTracker(contourFile, folderContent)

    def GetIds(self):
        return self.ids

    def GetIdsAndLabels(self):
        return self.ids, self.labels

    def GetIdsToLabelsDict(self):
        return dict(zip(self.ids, self.labels))

    def GetLabels(self):
        return self.labels

    def GetLabelsToIdsDict(self):
        return dict(zip(self.labels, self.ids))

    def RunCellIdTracker(self, contourFile=None, folderContent=None):
        self.linesOfContourFile = self.loadLinesOfContourFile(contourFile, folderContent)
        if not self.linesOfContourFile is None:
            self.labels, self.ids = self.extractLabelsAndIds(self.linesOfContourFile, headerOffSet=self.headerOffSet)
        else:
            if self.globalVerbosity >= 1:
                warnings.warn(f"The labels, ids could not be extracted as the linesOfContourFile is None could not be loaded, due to the {folderContent=} and {contourFile=} being None.")

    def loadLinesOfContourFile(self, contourFile=None, folderContent=None):
        if not contourFile is None or not folderContent is None:
            if not contourFile is None:
                if type(contourFile) == str:
                    with open(contourFile, "r") as fh:
                        linesOfContourFile = fh.readlines()
                elif type(contourFile) == list:
                    linesOfContourFile = contourFile
                else:
                    raise TypeError(f"The {contourFile=} should have been of type str or list, but is {type(contourFile)=}")
            else:
                if type(folderContent) == FolderContent:
                    linesOfContourFile = folderContent.LoadKeyUsingFilenameDict("contourFilename")
                else:
                    raise TypeError(f"The {folderContent=} should have been of type FolderContent, but is {type(folderContent)=}")
        else:
            linesOfContourFile = None
            if self.globalVerbosity >= 1:
                warnings.warn(f"The lines of the contour file could not be loaded, due to the {folderContent=} and {contourFile=} being None.")
        return linesOfContourFile

    def extractLabelsAndIds(self, linesOfContourFile, headerOffSet=4):
        labels, ids = [], []
        lineNr = 0 - headerOffSet
        for line in linesOfContourFile:
            if lineNr >= 0:
                if lineNr % 3 == 0:
                    splitLine = line.split("\t")
                    try:
                        id = int(splitLine[0])
                    except ValueError as e:
                        print("The Contour is probably not starting in the {} line".format(headerOffSet))
                        print(e)
                        sys.exit(1)
                    ids.append(id)
                elif lineNr % 3 == 1:
                    splitLine = line.split("\t")
                    try:
                        label = int(splitLine[0])
                    except ValueError as e:
                        print("The Contour is probably not starting in the {} line".format(headerOffSet))
                        print(e)
                        sys.exit(1)
                    labels.append(label)
            lineNr += 1
        if self.asArray:
            labels = np.asarray(labels)
            ids = np.asarray(ids)
        return labels, ids

def checkIdAndLabelledImageProcess(folderContent, trackedCellId=None):
    print(folderContent)
    labelledImage = folderContent.LoadKeyUsingFilenameDict("labelledImageFilename")
    cellContours = folderContent.LoadKeyUsingFilenameDict("cellContours", convertDictKeysToInt=True, convertDictValuesToNpArray=True)
    myCellIdTracker = CellIdTracker(folderContent=folderContent)
    cellIds, cellLabels = myCellIdTracker.GetIdsAndLabels()
    if not trackedCellId is None:
        isTrackedCellId = np.isin(cellIds, trackedCellId)
        assert np.sum(isTrackedCellId) == 1, f"The {trackedCellId=} id should only be present once, but is {isTrackedCellId=}"
        cellLabels = np.asarray(cellLabels)
        correspondingLabel = cellLabels[isTrackedCellId][0]
        print(correspondingLabel)
    else:
        trackedCellId = cellIds[0]
        correspondingLabel = cellLabels[0]
    contour = cellContours[correspondingLabel]
    for x,y in contour:
        labelledImage[x,y]=255
    plt.imshow(labelledImage)
    plt.show()
    return trackedCellId

def mainInspectCellOverTime():
    from MultiFolderContent import MultiFolderContent
    folderContentsName = "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"
    tissueName = "20170327 WT S1"
    folderContents = MultiFolderContent(folderContentsName)
    tissueFolderContents = folderContents.GetFolderContentsOfReplicate(tissueName)
    tissueFolderContents = MultiFolderContent(rawFolderContents=copy.deepcopy(tissueFolderContents))
    exisitingTimePoints = tissueFolderContents.GetTimePoints()
    exisitingTimePoints = pd.unique(exisitingTimePoints)
    trackedCellId = None
    for timePoint in exisitingTimePoints:
        timePointsFolderContent = tissueFolderContents.GetFolderContentsOfTimePoint(timePoint)[0]
        trackedCellId = checkIdAndLabelledImageProcess(timePointsFolderContent, trackedCellId=trackedCellId)

def main():
    from MultiFolderContent import MultiFolderContent
    contourFilename = "Images\\col-0\\20170327 WT S1\\0h\\Cell_Contours_33_Cells -- LTi6b-GFP mChTUA5 WT 0h S1_.txt"
    with open(contourFilename, "r") as fh:
        contourLines = fh.readlines()
    folderContentsName = "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"
    folderContents = MultiFolderContent(folderContentsName)
    selectedFolderContent = folderContents.GetFolderContentOfReplicateAtTimePoint("20170327 WT S1", "0h")
    myCellIdTracker = CellIdTracker(contourFile=contourFilename)
    print(myCellIdTracker.GetIdsToLabelsDict())
    myCellIdTracker = CellIdTracker(contourFile=contourLines)
    print(myCellIdTracker.GetIdsToLabelsDict())
    myCellIdTracker = CellIdTracker(folderContent=selectedFolderContent)
    print(myCellIdTracker.GetIdsToLabelsDict())

if __name__ == '__main__':
    # main()
    mainInspectCellOverTime()
