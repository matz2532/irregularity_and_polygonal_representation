import numpy as np
import pandas as pd
import re
import shutil
import sys

sys.path.insert(0, "./Code/DataStructures/")
sys.path.insert(0, "./Code/SAM_division_prediction/")

from CellTypeSelector import CellTypeSelector, extractLeafCellTypesOf, extractLeafCellTypeFromTablessOf
from CellInSAMCenterDecider import CellInSAMCenterDecider
from ConvertTextToLabels import ConvertTextToLabels
from FolderContent import FolderContent
from MGXContourFromPlyFileReader import MGXContourFromPlyFileReader
from MGXGraphFromPlyFileReader import MGXGraphFromPlyFileReader
from MultiFolderContent import MultiFolderContent
from pathlib import Path

class MGXToFolderContentConverter (object):

    # file to read name extensions
    geometricTableBaseName="_geometricData full.csv"
    originalImagePatternExtension="*PM*_MIP*.tif"
    plyCellGraphExtension="_cellGraph"
    plyContourNameExtension="_outlines halved.ply"
    plyJunctionNameExtension="_only junctions.ply"
    polygonGeometricTableBaseName="_geometricData poly.csv"
    # filename name extensions
    adjacencyListNameExtension             ="_adjacencyList.json"
    cellTypesBaseName                      ="_cellTypes.json"
    contourNameExtension                   ="_cellContour.json"
    guardCellJunctionPositionsNameExtension="_guardCellJunctionPositions.json"
    orderedJunctionsNameExtension          ="_orderedJunctionsPerCell.json"
    # filename keys
    adjacencyListFilenameKey             ="labelledImageAdjacencyList"
    cellTypeFilenameKey                  ="cellType"
    contoursFilenameKey                  ="cellContours"
    filamentImageFilenameKey             ="microtubuleImageFilename"
    geometricDataFilenameKey             ="geometricData"
    guardCellJunctionPositionsFilenameKey="guardCellJunctionPositions"
    labelledImageFilenameKey             ="labelledImageFilename"
    orderedJunctionsPerCellFilenameKey   ="orderedJunctionsPerCellFilename"
    originalImageFielnameKey             ="originalImageFilename"
    polygonGeometricDataFilenameKey      ="polygonalGeometricData"
    # cell selection parameters
    minimumRequiredArea=None # in this case square micro meter
    cellTypeNamesToKeep: str or list = "pavement cell" # ["pavement cell", "guard cell", "small cells"]
    guardCellTypeKey="guard cell"
    #
    defaultTimePoint="120h"
    #
    tissuePropertySeperator="_"
    #
    removeSmallCells=True

    def __init__(self, tissuePathFolder, extract3DContours=False, useCellTypeTable=False, keepAllCells=False, extractPeripheralCellsFromJunctions=True,
                 timePointInPath=False, runOnInit=True, **kwargs):
        if type(tissuePathFolder) == str:
            tissuePathFolder = Path(tissuePathFolder)
        self.tissuePathFolder = tissuePathFolder
        self.extract3DContours = extract3DContours
        self.SetDefaultParameter(kwargs)
        if runOnInit:
            self.folderContent = self.createFolderContentFrom(self.tissuePathFolder, useCellTypeTable=useCellTypeTable,
                                                              keepAllCells=keepAllCells,
                                                              extractPeripheralCellsFromJunctions=extractPeripheralCellsFromJunctions,
                                                              timePointInPath=timePointInPath, defaultTimePoint=self.defaultTimePoint)

    def SetDefaultParameter(self, kwargs):
        if "cellTypeNamesToKeep" in kwargs:
            self.cellTypeNamesToKeep = kwargs["cellTypeNamesToKeep"]
        if "geometricTableBaseName" in kwargs:
            self.geometricTableBaseName = kwargs["geometricTableBaseName"]
        if "guardCellTypeKey" in kwargs:
            self.guardCellTypeKey = kwargs["guardCellTypeKey"]
        if "originalImagePatternExtension" in kwargs:
            self.originalImagePatternExtension = kwargs["originalImagePatternExtension"]
        if "plyCellGraphExtension" in kwargs:
            self.plyCellGraphExtension = kwargs["plyCellGraphExtension"]
        if "plyContourNameExtension" in kwargs:
            self.plyContourNameExtension = kwargs["plyContourNameExtension"]
        if "plyJunctionNameExtension" in kwargs:
            self.plyJunctionNameExtension = kwargs["plyJunctionNameExtension"]
        if "polygonGeometricTableBaseName" in kwargs:
            self.polygonGeometricTableBaseName = kwargs["polygonGeometricTableBaseName"]
        if "defaultTimePoint" in kwargs:
            self.defaultTimePoint = kwargs["defaultTimePoint"]
        if "removeSmallCells" in kwargs:
            self.removeSmallCells = kwargs["removeSmallCells"]
        if "tissuePropertySeperator" in kwargs:
            self.tissuePropertySeperator = kwargs["tissuePropertySeperator"]

    def GetFolderContent(self):
        return self.folderContent

    def createFolderContentFrom(self, tissuePathFolder, timePointInPath=False, defaultTimePoint="120h", useCellTypeTable=False,
                                keepAllCells=False, extractPeripheralCellsFromJunctions=True, tissuePropertySeperator=None):
        if tissuePropertySeperator is None:
            tissuePropertySeperator = self.tissuePropertySeperator
        folderContent = self.createBaseFolderContentFromPath(tissuePathFolder, timePointInPath=timePointInPath, defaultTimePoint=defaultTimePoint, tissuePropertySeperator=tissuePropertySeperator)
        # add original images (membrane and filament, here microtubule, intensity images)
        self.addReferenceImagesInfoTo(folderContent, tissuePathFolder)
        self.addGeometricInfoTo(folderContent)
        # add adjacency list tissue of pavement cells
        tissueGraph = self.addGraphAdjacencyList(folderContent, tissuePathFolder)
        # add cell types
        cellTypesDict = self.addCellTypeInfoTo(folderContent, tissueGraph, useCellTypeTable=useCellTypeTable)
        # add contours and junction positions of pavement cell type and guard cell junction positions
        if keepAllCells:
            cellTypesDictToPropagate = None
        else:
            cellTypesDictToPropagate = cellTypesDict
        mgxJunctionReader = self.addContourInfoTo(folderContent, cellTypesDict=cellTypesDictToPropagate,
                                                  extractPeripheralCellsFromJunctions=extractPeripheralCellsFromJunctions,
                                                  plyExtension=self.plyJunctionNameExtension, fileResultsNameExtension=self.orderedJunctionsNameExtension,
                                                  filenameKey=self.orderedJunctionsPerCellFilenameKey)
        self.addContourInfoTo(folderContent, cellTypesDict=cellTypesDictToPropagate,
                              plyExtension=self.plyContourNameExtension, fileResultsNameExtension=self.contourNameExtension,
                              filenameKey=self.contoursFilenameKey)
        cellLabelsMissingJunctions = self.removeCellsWithoutJunctions(folderContent)
        if keepAllCells:
            if extractPeripheralCellsFromJunctions:
                cellTypesDict = self.addAndDistinguishPeripheralCellsTo(cellTypesDict, mgxJunctionReader)
        self.saveCellTypes(cellTypesDict, cellLabelsMissingJunctions)
        if not useCellTypeTable:
            guardCellLabels = cellTypesDict[self.guardCellTypeKey]
            junctionPositionsOfAllCells = mgxJunctionReader.GetCellsContourPositions()
            self.addGuardCellJunctionPositionTo(folderContent, junctionPositionsOfAllCells, guardCellLabels)
        return folderContent

    def createBaseFolderContentFromPath(self, tissuePathFolder, timePointInPath=True, defaultTimePoint="", tissuePropertySeperator="_"):
        timePoint = defaultTimePoint
        if timePointInPath:
            scenarioName, replicateName, timePoint = tissuePathFolder.parts[-3:]
            self.tissueName = replicateName + tissuePropertySeperator + timePoint
        else:
            scenarioName, replicateName = tissuePathFolder.parts[-2:]
            self.tissueName = replicateName
        self.tissueBaseFilename = str(Path(tissuePathFolder).joinpath(self.tissueName))
        folderContent = {"genotype": scenarioName, "replicateId": replicateName, "timePoint": timePoint}
        folderContent = FolderContent(folderContent)
        return folderContent

    def addReferenceImagesInfoTo(self, folderContent, tissuePathFolder):
        foundFilamentFilename = self.findFilenameBasedOnPattern(tissuePathFolder, pattern=self.originalImagePatternExtension)
        if not foundFilamentFilename is None:
            folderContent.AddDataToFilenameDict(foundFilamentFilename, self.filamentImageFilenameKey)

    def addGeometricInfoTo(self, folderContent):
        geometricTableFilename = self.tissueBaseFilename + self.geometricTableBaseName
        folderContent.AddDataToFilenameDict(geometricTableFilename, self.geometricDataFilenameKey)
        geometricPolygonTableFilename = self.tissueBaseFilename + self.polygonGeometricTableBaseName
        folderContent.AddDataToFilenameDict(geometricPolygonTableFilename, self.polygonGeometricDataFilenameKey)

    def findFilenameBasedOnPattern(self, givenPath, pattern, printOutNotCorrectFound=True):
        foundFilenames = list(givenPath.glob(pattern))
        if len(foundFilenames) == 0:
            if printOutNotCorrectFound:
                print(f"In the folder {givenPath} there was no original image with a pattern of {pattern}")
            return None
        else:
            if  len(foundFilenames) > 1:
                print(f"In the folder {givenPath} there were more than one original image with a pattern of {self.originalImagePatternExtension}, of the found filenames the first was selected {foundFilenames}")
            return foundFilenames[0]

    def addCellTypeInfoTo(self, folderContent, tissueGraph, useCellTypeTable=False, save=False):
        if useCellTypeTable:
            cellTypesDict = extractLeafCellTypeFromTablessOf(self.tissueBaseFilename,
                                                             cellTypesResultsBaseName=self.cellTypesBaseName, fullGeometricTableBaseName=self.geometricTableBaseName,
                                                             saveCellTypes=save, saveControlHeatmap=save, seperateSmallCells=self.removeSmallCells)
        else:
            cellTypesDict = extractLeafCellTypesOf(self.tissueBaseFilename, tissueGraph=tissueGraph,
                                                   cellTypesResultsBaseName=self.cellTypesBaseName, fullGeometricTableBaseName=self.geometricTableBaseName,
                                                   saveCellTypes=save, saveControlHeatmap=save, seperateSmallCells=self.removeSmallCells)
        cellTypesFileName = self.tissueBaseFilename + self.cellTypesBaseName
        folderContent.AddDataToFilenameDict(cellTypesFileName, self.cellTypeFilenameKey)
        return cellTypesDict

    def addContourInfoTo(self, folderContent, cellTypesDict=None, extractPeripheralCellsFromJunctions=False,
                         plyExtension=".ply", fileResultsNameExtension="_exampleExtension.json", filenameKey="exampleFilenameKey",
                         allContoursExtraExtension="_withAllCells.json",
                         save=True):
        plyFilename = self.tissueBaseFilename + plyExtension
        geometricData = folderContent.LoadKeyUsingFilenameDict(self.geometricDataFilenameKey, skipfooter=4)
        contourReader = MGXContourFromPlyFileReader(plyFilename, extract3DContours=self.extract3DContours, geometricData=geometricData)
        if cellTypesDict is None:
            onlyKeepCellLabels = None
        else:
            if extractPeripheralCellsFromJunctions:
                cellTypesDict = self.addAndDistinguishPeripheralCellsTo(cellTypesDict, contourReader)
            if isinstance(self.cellTypeNamesToKeep, str):
                onlyKeepCellLabels = cellTypesDict[self.cellTypeNamesToKeep]
            else:
                onlyKeepCellLabels = np.concatenate([cellTypesDict[cellTypeName] for cellTypeName in self.cellTypeNamesToKeep])
            if not self.minimumRequiredArea is None:
                geometricData = folderContent.LoadKeyUsingFilenameDict(self.geometricDataFilenameKey, skipfooter=4)
                cellSizesOfCellsToKeep = [geometricData.iloc[np.where(geometricData.iloc[:, 0] == cell)[0], 2].values[0] for cell in onlyKeepCellLabels]
            else:
                cellSizesOfCellsToKeep = None
        if save:
            allContoursExtension = Path(fileResultsNameExtension).stem + allContoursExtraExtension
            allContoursFilename = contourReader.SaveCellsContoursPositions(tissueBaseFilename=self.tissueBaseFilename,
                                                                           resultsNameExtension=allContoursExtension)
            allContoursFilenameKey = filenameKey + Path(allContoursExtraExtension).stem
            folderContent.AddDataToFilenameDict(allContoursFilename, allContoursFilenameKey)
            contoursFilename = contourReader.SaveCellsContoursPositions(tissueBaseFilename=self.tissueBaseFilename,
                                                                        resultsNameExtension=fileResultsNameExtension,
                                                                        onlyKeepCellLabels=onlyKeepCellLabels,
                                                                        cellSizesOfCellsToKeep=cellSizesOfCellsToKeep)
            folderContent.AddDataToFilenameDict(contoursFilename, filenameKey)
        return contourReader

    def addAndDistinguishPeripheralCellsTo(self, cellTypesDict, contourReader):
        peripheralCellLabels = contourReader.ExtractPeripheralCellLabelsFromJunctions()
        cellSelector = CellTypeSelector(runOnInit=False)
        cellSelector.SetCellType(cellTypesDict)
        if len(peripheralCellLabels) > 0:
            cellSelector.UpdateCellTypeWith({"peripheral cells": peripheralCellLabels}, keepingCellsInType=self.guardCellTypeKey)
            cellTypesDict = cellSelector.GetCellType()
        return cellTypesDict

    def removeCellsWithoutJunctions(self, folderContent):
        cellsContours = folderContent.LoadKeyUsingFilenameDict(self.contoursFilenameKey)
        cellsJunctions = folderContent.LoadKeyUsingFilenameDict(self.orderedJunctionsPerCellFilenameKey)
        cellLabelsHavingContour = np.array(list(cellsContours.keys()))
        cellLabelsHavingJunctions = list(cellsJunctions.keys())
        cellLabelsMissingJunctions = np.setdiff1d(cellLabelsHavingContour, cellLabelsHavingJunctions)
        if len(cellLabelsMissingJunctions) > 0:
            for cellLabel in cellLabelsMissingJunctions:
                cellsContours.pop(cellLabel)
            contoursFilename = folderContent.GetFilenameDict()[self.contoursFilenameKey]
            folderContent.SaveDataFilesTo(cellsContours, contoursFilename)
        return cellLabelsMissingJunctions

    def saveCellTypes(self, cellTypesDict, cellLabelsMissingJunctions: list = []):
        cellSelector = CellTypeSelector(tissueBaseFilename=self.tissueBaseFilename, runOnInit=False)
        cellSelector.SetCellType(cellTypesDict)
        if len(cellLabelsMissingJunctions) > 0:
            cellSelector.UpdateCellTypeWith({"cells without contour": cellLabelsMissingJunctions}, keepingCellsInType=self.guardCellTypeKey)
        cellSelector.SaveCellTypes(baseNameExtension=self.cellTypesBaseName)
        cellSelector.SaveHeatmapOfCellTypes(geometricTableBaseName=self.geometricTableBaseName)

    def addGuardCellJunctionPositionTo(self, folderContent, junctionPositionsOfAllCells, guardCellLabels):
        if len(guardCellLabels) == 0:
            return
        guardCellJunctionPositions = []
        for cellLabel, junctionPositions in junctionPositionsOfAllCells.items():
            if cellLabel in guardCellLabels:
                guardCellJunctionPositions.append(junctionPositions)
        guardCellJunctionPositions = np.concatenate(guardCellJunctionPositions, axis=0)
        duplicateCoordinates = []
        for i, coordinates in enumerate(guardCellJunctionPositions[:-1]):
            for j, otherCoordinates in enumerate(guardCellJunctionPositions[i+1:]):
                if np.all(coordinates == otherCoordinates):
                    duplicateCoordinates.append(i + 1 + j)
        guardCellJunctionPositions = np.delete(guardCellJunctionPositions, duplicateCoordinates, axis=0)
        guardCellJunctionPositionsFilename = self.tissueBaseFilename + self.guardCellJunctionPositionsNameExtension
        folderContent.SaveDataFilesTo(guardCellJunctionPositions, guardCellJunctionPositionsFilename)
        folderContent.AddDataToFilenameDict(guardCellJunctionPositionsFilename, self.guardCellJunctionPositionsFilenameKey)

    def addGraphAdjacencyList(self, folderContent, tissuePathFolder):
        junctionExtensionAsPath = Path(self.plyJunctionNameExtension)
        plyGraphFilename = self.tissueBaseFilename + junctionExtensionAsPath.stem + self.plyCellGraphExtension + junctionExtensionAsPath.suffix
        graphCreator = MGXGraphFromPlyFileReader(plyGraphFilename)
        adjacencyListFilename = graphCreator.SaveGraphsAdjacencyList(tissueBaseFilename=self.tissueBaseFilename,
                                                                     resultsNameExtension=self.adjacencyListNameExtension)
        folderContent.AddDataToFilenameDict(adjacencyListFilename, self.adjacencyListFilenameKey)
        tissueGraph = graphCreator.GetGraph()
        return tissueGraph

def mainCreateAllFullCotyledons(saveFolderContentsUnder=None, cotyledonBaseFolder="Images/full cotyledons/",
                                plyContourNameExtension="_outlines halved.ply",
                                allTissueIdentifiers=[["WT", "20200220 WT S1"],
                                                     ["WT", "20200221 WT S2"],
                                                     ["WT", "20200221 WT S3"],
                                                     ["WT", "20200221 WT S5"],],
                                cellTypeNamesToKeep=None, overwrite=True, **kwargs):
    if saveFolderContentsUnder is None:
        saveFolderContentsUnder = cotyledonBaseFolder + Path(cotyledonBaseFolder).stem + "_multiFolderContent_temp.pkl"
    if overwrite:
        multiFolderContent = MultiFolderContent()
        multiFolderContent.SetAllFolderContentsFilename(saveFolderContentsUnder)
    else:
        multiFolderContent = MultiFolderContent(saveFolderContentsUnder)
    if cellTypeNamesToKeep is not None:
        kwargs["cellTypeNamesToKeep"] = cellTypeNamesToKeep
    if "removeSmallCellsPerGenotype" in kwargs and "removeSmallCells" not in kwargs:
        removeSmallCellsPerGenotype = kwargs["removeSmallCellsPerGenotype"]
    else:
        removeSmallCellsPerGenotype = None
    for tissueIdentifier in allTissueIdentifiers:
        assert len(tissueIdentifier) == 2 or len(tissueIdentifier) == 3, f"The tissue identifier {tissueIdentifier} contains neither 2 nor 3 tissueIdentifier, but {len(tissueIdentifier)}, which is not yet implemented."
        if len(tissueIdentifier) == 2:
            genotype, tissueReplicateId = tissueIdentifier
        else:
            genotype, tissueReplicateId, timePoint = tissueIdentifier
            kwargs["defaultTimePoint"] = timePoint
        tissuePathFolder = Path(f"{cotyledonBaseFolder}{genotype}/{tissueReplicateId}/")
        if removeSmallCellsPerGenotype is not None:
            assert genotype in removeSmallCellsPerGenotype, f"When removing small cells per genotype the {genotype=} needs to be present, but only the following genotypes are present {list(removeSmallCellsPerGenotype.keys())}"
            kwargs["removeSmallCells"] = removeSmallCellsPerGenotype[genotype]
        myMGXToFolderContentConverter = MGXToFolderContentConverter(tissuePathFolder, plyContourNameExtension=plyContourNameExtension, **kwargs)
        folderContent = myMGXToFolderContentConverter.GetFolderContent()
        multiFolderContent.AppendFolderContent(folderContent)
    multiFolderContent.UpdateFolderContents()

def mainCreateProcessedDataOf(imageFolder="Images/Matz2022SAM/", addToExisiting=False, extract3DContours=True,
                              plyContourNameExtension="_full outlines.ply",
                              geometricTableBaseName="_geometricData.csv",
                              polygonGeometricTableBaseName="_geometricData poly.csv",
                              useCellTypeTable=True, timePointInPath=True,
                              checkTimePointsForRemovalOfSmallCells=False, timePointsForWhichNotRemoveSmallCells=["1DAI", "1.5DAI", "2DAI"],
                              redoForTissueInfo=None):
    """
    redoForTissueInfo needs to be a numpy array, where for each entry the first sub-entry is the replicate name and the second is the time point
                      example: np.array([["P1", "T2"], ["P2", "T1"], ["P2", "T1"]]))
    """
    from MultiFolderContent import MultiFolderContent
    scenarioBaseFolder = Path(imageFolder)
    saveFolderContentsUnder = imageFolder + scenarioBaseFolder.parts[-1] + ".pkl"
    resetFolderContent = not addToExisiting and redoForTissueInfo is None
    multiFolderContent = MultiFolderContent(saveFolderContentsUnder, resetFolderContent=resetFolderContent)
    for replicatePath in scenarioBaseFolder.glob("*/*"):
        if not replicatePath.is_dir():
            continue
        replicateName = replicatePath.parts[-1]
        for timePointTissuePath in replicatePath.glob("*"):
            if not timePointTissuePath.is_dir():
                continue
            removeSmallCells = False
            timePoint = timePointTissuePath.parts[-1]
            if checkTimePointsForRemovalOfSmallCells:
                if not timePoint in timePointsForWhichNotRemoveSmallCells:
                    removeSmallCells = True
            if not redoForTissueInfo is None:
                if not np.any(np.sum((replicateName, timePoint) == redoForTissueInfo, axis=1) == 2):
                    continue
            myMGXToFolderContentConverter = MGXToFolderContentConverter(timePointTissuePath,
                                                                        extract3DContours=extract3DContours,
                                                                        useCellTypeTable=useCellTypeTable,
                                                                        timePointInPath=timePointInPath,
                                                                        plyContourNameExtension=plyContourNameExtension,
                                                                        geometricTableBaseName=geometricTableBaseName,
                                                                        polygonGeometricTableBaseName=polygonGeometricTableBaseName,
                                                                        removeSmallCells=removeSmallCells)
            folderContent = myMGXToFolderContentConverter.GetFolderContent()
            if redoForTissueInfo is None:
                multiFolderContent.AppendFolderContent(folderContent)
            else:
                multiFolderContent.ExchangeFolderContent(folderContent)
    multiFolderContent.UpdateFolderContents()

def addGeometricDataFilenameAndKey(allFolderContentsFilename="Images/first full cotyledons/full cotyledons.pkl",
                                   baseFolderKey="orderedJunctionsPerCellFilename", timePointInPath=True, seperator="_",
                                   geometricTableBaseName="{}_geometricData.csv", geometricDataFilenameKey="geometricData",
                                   polygonGeometricTableBaseName="{}_geometricData poly.csv", polygonGeometricDataFilenameKey="polygonalGeometricData"):
    from MultiFolderContent import MultiFolderContent
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    assert len(list(multiFolderContent)) > 0, f"The multiFolderContent is empty initialized from {allFolderContentsFilename=}"
    for folderContent in multiFolderContent:
        baseFolder = folderContent.GetFilenameDict()[baseFolderKey]
        baseFolderPath = Path(baseFolder)
        if not baseFolderPath.is_dir():
            baseFolderPath = baseFolderPath.parent
        scenarioName, replicateName, timePoint = folderContent.GetTissueInfos()
        if timePointInPath:
            replicateName = replicateName + seperator + timePoint
        else:
            replicateName = replicateName
        geometricTableFilename = str(Path(baseFolderPath).joinpath(geometricTableBaseName.format(replicateName)))
        assert Path(geometricTableFilename).is_file(), f"The {geometricTableFilename=} does not exist and could not be added to the following folder content:\n{folderContent}"
        folderContent.AddDataToFilenameDict(geometricTableFilename, geometricDataFilenameKey)
        geometricPolygonTableFilename = str(Path(baseFolderPath).joinpath(polygonGeometricTableBaseName.format(replicateName)))
        assert Path(geometricPolygonTableFilename).is_file(), f"The {geometricPolygonTableFilename=} does not exist and could not be added to the following folder content:\n{folderContent}"
        folderContent.AddDataToFilenameDict(geometricPolygonTableFilename, polygonGeometricDataFilenameKey)
    multiFolderContent.UpdateFolderContents()

# next step select stomata cells and check with guardCellsToRemoveBaseName + "final.csv"
def createGeometricdataTableWithRemovedStomata(baseFolder = "Images/full cotyledons/",
                                               tissueIdentifier = [["WT", '20200220 WT S1', '120h'],
                                                                   ["WT", '20200220 WT S2', '120h'],
                                                                   ["WT", '20200220 WT S3', '120h'],
                                                                   ["WT", '20200220 WT S4', '120h'],
                                                                    ],
                                               geometricTableBaseName = "_geometricData full.csv",
                                               geometricDataWithoutStomataBaseName = "_geometricData no stomata.csv",
                                               guardCellsToRemoveBaseName = "_proposed guard cells.txt",
                                               skipfooter = 4,
                                               copyInsteadOfThrowingErrorIfNoStomata: bool = True):

    for g, r, t in tissueIdentifier:
        tissueBaseName = str(Path(baseFolder).joinpath(g, r, r))
        geometricTableFilename = tissueBaseName + geometricTableBaseName
        guardCellsToRemoveFilename = tissueBaseName + guardCellsToRemoveBaseName
        geometricDataWithoutStomataFilename = tissueBaseName + geometricDataWithoutStomataBaseName
        if Path(guardCellsToRemoveFilename).is_file():
            fullGeometricDf = pd.read_csv(geometricTableFilename, skipfooter=skipfooter, engine="python")
            guardCellLabels = ConvertTextToLabels(guardCellsToRemoveFilename).labels
            isNotGuardCell = np.isin(fullGeometricDf.iloc[:, 0], guardCellLabels, invert=True)
            idxOfCellsToKeep = np.where(isNotGuardCell)[0]
            reducedGeomDf = fullGeometricDf.iloc[idxOfCellsToKeep, :]
            reducedGeomDf.to_csv(geometricDataWithoutStomataFilename, index=False)
            with open(geometricDataWithoutStomataFilename, "a") as fh:
                with open(geometricTableFilename, "r") as geometricDataFh:
                    lastLines = geometricDataFh.readlines()[-skipfooter:]
                for line in lastLines:
                    fh.write(line)
        else:
            if not Path(geometricDataWithoutStomataFilename).is_file():
                if copyInsteadOfThrowingErrorIfNoStomata:
                    shutil.copy(geometricTableFilename, geometricDataWithoutStomataFilename)
                else:
                    print(f"The file geometric data file without stomata {geometricDataWithoutStomataFilename} was not present and\ncould not be created as {guardCellsToRemoveFilename=} is not present for {g}, {r}, {t}")

def saveCentralCellsAsCellType(centerCellsDict,
                               imageFolder="Images/Matz2022SAM/", centerRadius=20,
                               geometryBaseName="{}_{}_geometricData.csv", seperator="_",
                               selectedCellsFilenameSuffix="_CELL_TYPE.csv", centralCellId=8,
                               ):
    scenarioBaseFolder = Path(imageFolder)
    for replicatePath in scenarioBaseFolder.glob("*/*"):
        if not replicatePath.is_dir():
            continue
        replicateName = replicatePath.parts[-1]
        scenarioName = replicatePath.parts[-2]
        tissueInfo = (scenarioName, replicateName)
        centerCellsOfTimePoints = centerCellsDict[tissueInfo]
        for timePointTissuePath in replicatePath.glob("*"):
            if not timePointTissuePath.is_dir():
                continue
            timePoint = timePointTissuePath.parts[-1]
            geometryFilename = timePointTissuePath.joinpath(geometryBaseName.format(replicateName, timePoint))
            centerDefiningCells = centerCellsOfTimePoints[timePoint]
            centerDecider = CellInSAMCenterDecider(geometryFilename, centerDefiningCells, centerRadius=centerRadius)
            centralCells = centerDecider.GetCentralCells()
            replicateAtTimePointName = replicateName + seperator + timePoint
            nrOfCentralCells = len(centralCells)
            cellTypeIds = np.repeat(centralCellId, nrOfCentralCells)
            cellTypeName = np.repeat("central cell", nrOfCentralCells)
            # just add those cells who you like to calculate on, i.e. not adding non-central cells, only guard cells are filtered out (need a cell id of 16)
            cellTypeDf = pd.DataFrame({"Label":centralCells, "Cell type id":cellTypeIds, "Cell type name":cellTypeName})
            centralCellsFilename = timePointTissuePath.joinpath(replicateAtTimePointName + selectedCellsFilenameSuffix)
            cellTypeDf.to_csv(centralCellsFilename, index=False)

def mainInitalizeSAMDataAddingContoursAndJunctions(dataBaseFolder="Images/Matz2022SAM/",
                                                   centerDefiningCellsDict = {('WT inflorescence meristem', 'P1'): {"T0": [618, 467, 570], "T1": [5048, 5305], "T2": [5849, 5601], "T3": [6178, 6155, 6164], "T4": [6288, 6240]},
                                                                              ('WT inflorescence meristem', 'P2'): {"T0": [392], "T1": [553, 779, 527], "T2": [525], "T3": [1135], "T4": [1664, 1657]},
                                                                              ('WT inflorescence meristem', 'P5'): {"T0": [38], "T1": [585, 968, 982], "T2": [927, 1017], "T3": [1136], "T4": [1618, 1575, 1445]},
                                                                              ('WT inflorescence meristem', 'P6'): {"T0": [861], "T1": [1334, 1634, 1651], "T2": [1735, 1762, 1803], "T3": [2109, 2176], "T4": [2381]},
                                                                              ('WT inflorescence meristem', 'P8'): {"T0": [3241, 2869, 3044], "T1": [3421, 3657], "T2": [2676, 2805, 2876], "T3": [2898, 2997, 3013], "T4": [358, 189]},
                                                                              },
                                                   overwrite=True):
    saveCentralCellsAsCellType(centerDefiningCellsDict, imageFolder=dataBaseFolder)
    mainCreateProcessedDataOf(imageFolder=dataBaseFolder, addToExisiting=not overwrite)
