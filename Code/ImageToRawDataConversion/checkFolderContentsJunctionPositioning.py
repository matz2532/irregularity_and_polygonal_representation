import json
import numpy as np
import skimage.io
import sys

sys.path.insert(0, "./Code/DataStructures/")
from MatplotlibManualJunctionCorrector import MatplotlibManualJunctionCorrector
from MultiFolderContent import MultiFolderContent
from PreProcessorDummy import PreProcessorDummy
from TriWayJunctionFinder import TriWayJunctionFinder

def extractAndSaveContourAndBaseJunctions(folderContent, dataBaseFolder,
                                   simpleJunctionBaseName="baseTriWayJunctions.npy",
                                   globalVerbosity=3):
    tissueFolderExtension = folderContent.GetFolder()
    dataFolder = dataBaseFolder + tissueFolderExtension
    # load contour file in text format to convert it in simple dictionary where entries correpsond to cell label and values to contour positions
    labelledImageFilename = folderContent.GetFilenameDictKeyValue("labelledImageFilename")
    contourFilename = folderContent.GetFilenameDictKeyValue("contourFilename")
    myPreProcessorDummy = PreProcessorDummy(labelledImageFilename, contourFilename=contourFilename)
    cellContour = myPreProcessorDummy.GetContoursOfValdidLabels()
    cellContourFilename = dataFolder + "cellContour.json"
    folderContent.SaveDataFilesTo(cellContour, cellContourFilename, convertDictValuesToList=True)
    folderContent.AddDataToFilenameDict(cellContourFilename, "cellContours")
    # calculate tri-way junctions from labelled image
    labelledImage = folderContent.LoadKeyUsingFilenameDict("labelledImageFilename")
    myTriWayJunctionFinder = TriWayJunctionFinder(labelledImage)
    triWayJunctions = myTriWayJunctionFinder.GetTriWayJunctionPositions()
    baseTriWayJunctionFilename = dataFolder + simpleJunctionBaseName
    np.save(baseTriWayJunctionFilename, triWayJunctions)
    folderContent.AddDataToFilenameDict(baseTriWayJunctionFilename, "baseTriWayJunctions")

def checkFolderContentsJunctionPositioning(folderContent, dataBaseFolder,
                                   finalJunctionBaseName="correctedTriWayJunctions.npy",
                                   guardCellJunctionBaseName="guardCellJunctionPositions.npy",
                                   finalCorrespondingJunctionName="correctedCorresponding.json",
                                   save=True, globalVerbosity=3):
        if folderContent.IsKeyInFilenameDict("finalJunctionFilename"):
            triWayJunctions = folderContent.LoadKeyUsingFilenameDict("finalJunctionFilename")
        elif folderContent.IsKeyInFilenameDict("baseTriWayJunctions"):
            triWayJunctions = folderContent.LoadKeyUsingFilenameDict("baseTriWayJunctions")
        else:
            extractAndSaveContourAndBaseJunctions(folderContent, dataBaseFolder,
                                               globalVerbosity=globalVerbosity)
            triWayJunctions = folderContent.LoadKeyUsingFilenameDict("baseTriWayJunctions")
        originalImage = folderContent.LoadKeyUsingFilenameDict("originalImageFilename")
        labelledImage = folderContent.LoadKeyUsingFilenameDict("labelledImageFilename")
        cellContourDict = folderContent.LoadKeyUsingFilenameDict("cellContours", convertDictKeysToInt=True, convertDictValuesToNpArray=True)
        skeletonImage = None
        if folderContent.IsKeyInFilenameDict("guardCellJunctionPositions"):
            guardCellJunctionPositions = folderContent.LoadKeyUsingFilenameDict("guardCellJunctionPositions")
        else:
            guardCellJunctionPositions = None
        tissueName = folderContent.GetTissueName()
        title = "Select missing triWayJunctions for cells of {}".format(tissueName)
        tissueFolderExtension = folderContent.GetFolder()
        myMatplotlibManualJunctionCorrector = MatplotlibManualJunctionCorrector(originalImage, labelledImage, cellContourDict,
                                                                skeletonImage, triWayJunctions=triWayJunctions,
                                                                guardCellJunctionPositions=guardCellJunctionPositions,
                                                                title=title)
        if save:
            dataFolder = dataBaseFolder + tissueFolderExtension
            finalJunctionFilename = dataFolder + finalJunctionBaseName
            guardCellJunctionFilename = dataFolder + guardCellJunctionBaseName
            myMatplotlibManualJunctionCorrector.SaveJunctions(finalJunctionFilename)
            myMatplotlibManualJunctionCorrector.SaveGuardCellJunctionPositions(guardCellJunctionFilename)
            folderContent.AddDataToFilenameDict(finalJunctionFilename, "finalJunctionFilename")
            print(finalJunctionFilename, "finalJunctionFilename", folderContent.IsKeyInFilenameDict("finalJunctionFilename"))
            folderContent.AddDataToFilenameDict(guardCellJunctionFilename, "guardCellJunctionPositions")
            correspondingCells = TriWayJunctionFinder().findCorrespondingCells(myMatplotlibManualJunctionCorrector.GetTriWayJunctions(), labelledImage)
            correspondingCellsFilename = dataFolder + finalCorrespondingJunctionName
            with open(correspondingCellsFilename, "w") as fh:
                correspondingCells = [cells.tolist() for cells in correspondingCells]
                data = json.dump(correspondingCells, fh)
            folderContent.AddDataToFilenameDict(correspondingCellsFilename, "correspondingCellsFilename")
            if globalVerbosity >= 3:
                print("Added key {}, {}, and {} of tissue {} to filenameDict with value {}, {}, and {} and saved corresponding data.".format("finalJunctionFilename", "guardCellJunctionBaseName", "guardCellJunctionPositions", tissueName, finalJunctionFilename, guardCellJunctionFilename, correspondingCellsFilename))

def visualiseJunctions(allFolderContentsFilename="Images/Eng2021Cotyledons.json",
                       selectedReplicateId="20170327 WT S1", timePoint="72h"):
    myMultiFolderContent = MultiFolderContent(allFolderContentsFilename)
    folderContent = myMultiFolderContent.GetFolderContentOfReplicateAtTimePoint(selectedReplicateId, timePoint)
    originalImage = folderContent.LoadKeyUsingFilenameDict("originalImageFilename")
    labelledImage = folderContent.LoadKeyUsingFilenameDict("labelledImageFilename")
    cellContour = folderContent.LoadKeyUsingFilenameDict("cellContours", convertDictKeysToInt=True, convertDictValuesToNpArray=True)
    triWayJunctions = folderContent.LoadKeyUsingFilenameDict("finalJunctionFilename")
    tissueName = folderContent.GetTissueName()
    title = "Select missing triWayJunctions for cells of {}".format(tissueName)
    myMatplotlibManualJunctionCorrector = MatplotlibManualJunctionCorrector(originalImage, labelledImage, cellContour,
                                                            skeletonImage=None, triWayJunctions=triWayJunctions,
                                                            title=title)

def main():
    # visualiseJunctions()
    allFolderContentsFilename = "Images/Eng2021Cotyledons.json"
    selectedReplicateId = "20170501 WT S1"
    timePoint = "96h"
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    for folderContent in multiFolderContent:
        if folderContent.GetReplicateId() == selectedReplicateId and folderContent.GetTimePoint() == timePoint:
            checkFolderContentsJunctionPositioning(folderContent, None, save=False)

if __name__== "__main__":
    main()
