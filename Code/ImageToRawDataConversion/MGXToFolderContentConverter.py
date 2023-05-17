import json
import numpy as np
import pandas as pd
import pickle
import re
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
    filamentImagePatternExtension="*MT*MIP_BOREALIS*.tif"
    geometricTableBaseName="_geometricData full.csv"
    originalImagePatternExtension="*PM*_MIP*.tif"
    plyCellGraphExtension="_cellGraph"
    plyContourNameExtension="_outlines halfed.ply"
    plyJunctionNameExtension="_only junctions.ply"
    polygonGeometricTableBaseName="_geometricData poly.csv"
    # filename name extensions
    adjacencyListNameExtension             ="_adjacencyList.json"
    cellTypesBaseName                      ="_cellTypes.json"
    contourNameExtension                   ="_cellContour.pkl"
    guardCellJunctionPositionsNameExtension="_guardCellJunctionPositions.npy"
    orderedJunctionsNameExtension          ="_orderedJunctionsPerCell.pkl"
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
    pavementCellTypeKey="pavement cell"
    guardCellTypeKey="guard cell"
    #
    tissuePropertySeperator="_"
    #
    removeSmallCells=True

    def __init__(self, tissuePathFolder, extract3DContours=False, useCellTypeTable=False, timePointInPath=False, runOnInit=True, **kwargs):
        if type(tissuePathFolder) == str:
            tissuePathFolder = Path(tissuePathFolder)
        self.tissuePathFolder = tissuePathFolder
        self.extract3DContours = extract3DContours
        self.SetDefaultParameter(kwargs)
        if runOnInit:
            self.folderContent = self.createFolderContentFrom(self.tissuePathFolder, timePointInPath=timePointInPath, useCellTypeTable=useCellTypeTable)

    def SetDefaultParameter(self, kwargs):
        if "filamentImagePatternExtension" in kwargs:
            self.filamentImagePatternExtension = kwargs["filamentImagePatternExtension"]
        if "originalImagePatternExtension" in kwargs:
            self.originalImagePatternExtension = kwargs["originalImagePatternExtension"]
        if "plyCellGraphExtension" in kwargs:
            self.plyCellGraphExtension = kwargs["plyCellGraphExtension"]
        if "plyContourNameExtension" in kwargs:
            self.plyContourNameExtension = kwargs["plyContourNameExtension"]
        if "plyJunctionNameExtension" in kwargs:
            self.plyJunctionNameExtension = kwargs["plyJunctionNameExtension"]
        if "geometricTableBaseName" in kwargs:
            self.geometricTableBaseName = kwargs["geometricTableBaseName"]
        if "polygonGeometricTableBaseName" in kwargs:
            self.polygonGeometricTableBaseName = kwargs["polygonGeometricTableBaseName"]
        if "tissuePropertySeperator" in kwargs:
            self.tissuePropertySeperator = kwargs["tissuePropertySeperator"]
        if "removeSmallCells" in kwargs:
            self.removeSmallCells = kwargs["removeSmallCells"]

    def GetFolderContent(self):
        return self.folderContent

    def createFolderContentFrom(self, tissuePathFolder, timePointInPath=False, defaultTimePoint="120h", useCellTypeTable=False, tissuePropertySeperator=None):
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
        mgxJunctionReader = self.addContourInfoTo(folderContent, cellTypesDict=cellTypesDict,
                                                  extractPeripheralCellsFromJunctions=True,
                                                  plyExtension=self.plyJunctionNameExtension, fileResultsNameExtension=self.orderedJunctionsNameExtension,
                                                  filenameKey=self.orderedJunctionsPerCellFilenameKey)
        mgxContourReader = self.addContourInfoTo(folderContent, cellTypesDict=cellTypesDict,
                                                 plyExtension=self.plyContourNameExtension, fileResultsNameExtension=self.contourNameExtension,
                                                 filenameKey=self.contoursFilenameKey)
        junctionZeroingPoint = mgxJunctionReader.GetZeroingPoint()
        contourZeroingPoint = mgxContourReader.GetZeroingPoint()
        if np.linalg.norm(contourZeroingPoint - junctionZeroingPoint) > 0.0001:
            furtherReduceJunctionsBy = junctionZeroingPoint - contourZeroingPoint
            # load junctions and subtract value from each point
        # self.addRecreatedLabelledImageInfoTo(folderContent, tissuePathFolder, mgxContourReader)
        self.removeCellsWithOutJunctions(folderContent, mgxContourReader, mgxJunctionReader, cellTypesDict)
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
        foundOriginalFilename = self.findFilenameBasedOnPattern(tissuePathFolder, pattern=self.filamentImagePatternExtension)
        if not foundOriginalFilename is None:
            folderContent.AddDataToFilenameDict(foundOriginalFilename, self.originalImageFielnameKey)
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
                         plyExtension=".ply", fileResultsNameExtension="_exampleExtension.pkl", filenameKey="exampleFilenameKey",
                         save=True):
        plyFilename = self.tissueBaseFilename + plyExtension
        contourReader = MGXContourFromPlyFileReader(plyFilename, extract3DContours=self.extract3DContours)
        if extractPeripheralCellsFromJunctions:
            peripheralCellLabels = contourReader.ExtractPeripheralCellLabelsFromJunctions()
            cellSelector = CellTypeSelector(runOnInit=False)
            cellSelector.SetCellType(cellTypesDict)
            if len(peripheralCellLabels) > 0:
                cellSelector.UpdateCellTypeWith({"peripheral cells":peripheralCellLabels}, keepingCellsInType=self.guardCellTypeKey)
                cellTypesDict = cellSelector.GetCellType()
        if cellTypesDict is None:
            onlyKeepCellLabels = None
        else:
            onlyKeepCellLabels = cellTypesDict[self.pavementCellTypeKey]
            if not self.minimumRequiredArea is None:
                geometricData = folderContent.LoadKeyUsingFilenameDict(self.geometricDataFilenameKey, skipfooter=4)
                cellSizesOfCellsToKeep = [geometricData.iloc[np.where(geometricData.iloc[:, 0] == cell)[0], 2].values[0] for cell in onlyKeepCellLabels]
            else:
                cellSizesOfCellsToKeep = None
        if save:
            contoursFilename = contourReader.SaveCellsContoursPositions(tissueBaseFilename=self.tissueBaseFilename,
                                                                        resultsNameExtension=fileResultsNameExtension,
                                                                        onlyKeepCellLabels=onlyKeepCellLabels,
                                                                        cellSizesOfCellsToKeep=cellSizesOfCellsToKeep)
            folderContent.AddDataToFilenameDict(contoursFilename, filenameKey)
        return contourReader

    def removeCellsWithOutJunctions(self, folderContent, mgxContourReader, mgxJunctionReader, cellTypesDict):
        cellsContours = folderContent.LoadKeyUsingFilenameDict(self.contoursFilenameKey)
        cellsJunctions = folderContent.LoadKeyUsingFilenameDict(self.orderedJunctionsPerCellFilenameKey)
        cellLabelsHavingContour = np.array(list(cellsContours.keys()))
        cellLabelsHavingJunctions = list(cellsJunctions.keys())
        cellLabelsMissingJunctions = np.setdiff1d(cellLabelsHavingContour, cellLabelsHavingJunctions)
        if len(cellLabelsMissingJunctions) > 0:
            for cellLabel in cellLabelsMissingJunctions:
                cellsContours.pop(cellLabel)
            cellLabelsHavingContour = np.array(list(cellsContours.keys()))
            cellLabelsHavingJunctions = list(cellsJunctions.keys())
            contoursFilename = folderContent.GetFilenameDict()[self.contoursFilenameKey]
            with open(contoursFilename, "wb") as fh:
                pickle.dump(cellsContours, fh)
        cellSelector = CellTypeSelector(tissueBaseFilename=self.tissueBaseFilename, runOnInit=False)
        cellSelector.SetCellType(cellTypesDict)
        if len(cellLabelsMissingJunctions) > 0:
            cellSelector.UpdateCellTypeWith({"cells without contour":cellLabelsMissingJunctions}, keepingCellsInType=self.guardCellTypeKey)
        cellSelector.SaveCellTypes(baseNameExtension=self.cellTypesBaseName)
        cellSelector.SaveHeatmapOfCellTypes(geometricTableBaseName=self.geometricTableBaseName)

    def addRecreatedLabelledImageInfoTo(self, folderContent, tissuePathFolder, mgxContourReader):
        # need to shift positions to be positive
        cellsContourPositions = mgxContourReader.GetCellsContourPositions()
        if folderContent.IsKeyInFilenameDict(self.originalImageFielnameKey):
            originalImage = folderContent.LoadKeyUsingFilenameDict(self.originalImageFielnameKey)
            imageShape = originalImage.shape[:2]
        else:
            allContourMaxPositions = [np.max(contourPositions, axis=0) for contourPositions in cellsContourPositions.values()]
            allContourMaxPositions = np.concatenate(allContourMaxPositions, axis=0).reshape(len(allContourMaxPositions), 2)
            imageShape = np.max(allContourMaxPositions, axis=0)
        labelledImage = np.zeros(imageShape, dtype=int)
        # fill labelledImage
        from skimage.draw import polygon
        for cellLabel, contourPositions in cellsContourPositions.items():
            rr, cc = polygon(contourPositions[:, 0], contourPositions[:, 1], imageShape)
            labelledImage[rr,cc] = int(cellLabel)
        import matplotlib.pyplot as plt
        plt.imshow(labelledImage)
        plt.show()
        labelledImageFilename = self.tissueBaseFilename + "_" + self.labelledImageFilenameKey + ".npy"
        np.save(labelledImageFilename, labelledImage)
        folderContent.AddDataToFilenameDict(labelledImageFilename, self.labelledImageFilenameKey)

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
        np.save(guardCellJunctionPositionsFilename, guardCellJunctionPositions)
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

def mainCreateAllFullCotyledons(saveFolderContentsUnder=None, cotyledonBaseFolder="Images/full cotyledons/"):
    if saveFolderContentsUnder is None:
        saveFolderContentsUnder = cotyledonBaseFolder + Path(cotyledonBaseFolder).name + "_multiFolderContent_temp.pkl"
    multiFolderContent = MultiFolderContent(saveFolderContentsUnder)
    allTissueReplicateIds = []
    allTissueIdentifers = [
        ["WT", "20200220 WT S1"],
        ["WT", "20200221 WT S2"],
        ["WT", "20200221 WT S3"],
        ["WT", "20200221 WT S5"],
        ["ktn1-2", '20200220 ktn1-2 S1', '120h'],
        ["ktn1-2", '20200220 ktn1-2 S2', '120h'],
        ["ktn1-2", '20200220 ktn1-2 S3', '120h'],
    ]
    for tissueIdentifer in allTissueIdentifers:
        assert len(tissueIdentifer) == 2 or len(tissueIdentifer) == 3, f"The tissue identifier {tissueIdentifer} contains neither 2 nor 3 tissueIdentifer, but {len(tissueIdentifer)}, which is not yet implemented."
        print(f"{tissueIdentifer=}")
        if len(tissueIdentifer) == 2:
            genotype, tissueReplicateId = tissueIdentifer
        else:
            genotype, tissueReplicateId, timePoint = tissueIdentifer
        if genotype == "WT":
            plyContourNameExtension = "_outlines halfed.ply"
        else:
            plyContourNameExtension = "_full outlines.ply"
        tissuePathFolder = Path(f"{cotyledonBaseFolder}{genotype}/{tissueReplicateId}/")
        myMGXToFolderContentConverter = MGXToFolderContentConverter(tissuePathFolder, plyContourNameExtension=plyContourNameExtension)
        folderContent = myMGXToFolderContentConverter.GetFolderContent()
        multiFolderContent.AppendFolderContent(folderContent)
    multiFolderContent.SetAllFolderContentsFilename(saveFolderContentsUnder)
    multiFolderContent.UpdateFolderContents()

def mainCreateProcessedDataOf(scenarioName="first_leaf_LeGloanec2022", multiFolderName="first leaf",
                              imageFolder="Images/", addToExisiting=False, extract3DContours=True,
                              plyContourNameExtension="_full outlines.ply",
                              geometricTableBaseName="_geometricData.csv",
                              polygonGeometricTableBaseName="_geometricData poly.csv",
                              useCellTypeTable=True, timePointInPath=True,
                              multiFolderBaseName="_multiFolderContent.pkl",
                              checkTimePointsForRemovalOfSmallCells=False, redoForTissueInfo=None):
    """
    redoForTissueInfo needs to be a numpy array, where for each entry the first sub-entry is the replicate name and the second is the time point
                      example: np.array([["P1", "T2"], ["P2", "T1"], ["P2", "T1"]]))
    """
    from MultiFolderContent import MultiFolderContent
    scenarioBaseFolder = Path(imageFolder).joinpath(scenarioName)
    saveFolderContentsUnder = imageFolder + multiFolderName + multiFolderBaseName
    resetFolderContent = not addToExisiting and redoForTissueInfo is None
    multiFolderContent = MultiFolderContent(saveFolderContentsUnder, resetFolderContent=resetFolderContent)
    for replicatePath in scenarioBaseFolder.glob("*"):
        if not replicatePath.is_dir():
            continue
        replicateName = replicatePath.parts[-1]
        for timePointTissuePath in replicatePath.glob("*"):
            if not timePointTissuePath.is_dir():
                continue
            removeSmallCells = False
            timePoint = timePointTissuePath.parts[-1]
            if checkTimePointsForRemovalOfSmallCells:
                if not timePoint in ["1DAI", "1.5DAI", "2DAI"]:
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

def addGeometricDataFilenameAndKey(allFolderContentsFilename="Images/first leaf_multiFolderContent.pkl",
                                   baseFolderKey="orderedJunctionsPerCellFilename", timePointInPath=True, seperator="_",
                                   geometricTableBaseName="{}_geometricData.csv", geometricDataFilenameKey="geometricData",
                                   polygonGeometricTableBaseName="{}_geometricData poly.csv", polygonGeometricDataFilenameKey="polygonalGeometricData"):
    from MultiFolderContent import MultiFolderContent
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
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
def createGeometricdataTableWithRemovedStomata():
    geometricTableBaseName = "_geometricData full.csv"
    geometricDataWithoutStomataBaseName = "_geometricData no stomata.csv"
    guardCellsToRemoveBaseName = "_proposed guard cells.txt"
    baseFolder = "Images/full cotyledons/"
    skipfooter = 4
    tissueIdentifer = [["WT", '20200220 WT S1', '120h'],
                       ["WT", '20200220 WT S2', '120h'],
                       ["WT", '20200220 WT S3', '120h'],
                       ["WT", '20200220 WT S4', '120h'],]

    for g, r, t in tissueIdentifer:
        tissueBaseName = str(Path(baseFolder).joinpath(g, r, r))
        geometricTableFilename = tissueBaseName + geometricTableBaseName
        guardCellsToRemoveFilename = tissueBaseName + guardCellsToRemoveBaseName
        if Path(guardCellsToRemoveFilename).is_file():
            geometricDataWithoutStomataFilename = tissueBaseName + geometricDataWithoutStomataBaseName
            fullGometricDf = pd.read_csv(geometricTableFilename, skipfooter=skipfooter, engine="python")
            guradCellLabels = ConvertTextToLabels(guardCellsToRemoveFilename).labels
            isNotGuardCell = np.isin(fullGometricDf.iloc[:, 0], guradCellLabels, invert=True)
            idxOfCellsToKeep = np.where(isNotGuardCell)[0]
            reducedGeomDf = fullGometricDf.iloc[idxOfCellsToKeep, :]
            reducedGeomDf.to_csv(geometricDataWithoutStomataFilename, index=False)
            with open(geometricDataWithoutStomataFilename, "a") as fh:
                with open(geometricTableFilename, "r") as geometricDataFh:
                    lastLines = geometricDataFh.readlines()[-skipfooter:]
                for line in lastLines:
                    fh.write(line)
        else:
            print(f"The file {guardCellsToRemoveFilename=} is not present for {g}, {r}, {t}")

def saveCentralCellsAsCellType(centerCellsDict, scenarioName="ktn inflorescence meristem",
                               imageFolder="Images/SAM/", centerRadius=20,
                               geometryBaseName="{}_{}_geometricData.csv", seperator="_",
                               selectedCellsFilenameSuffix="_CELL_TYPE.csv", centralCellId=8, nonCentralCellId=5,
                               ):
    scenarioBaseFolder = Path(imageFolder).joinpath(scenarioName)
    for replicatePath in scenarioBaseFolder.glob("*"):
        if not replicatePath.is_dir():
            continue
        replicateName = replicatePath.parts[-1]
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

def mainInitalizeSAMDataAddingContoursAndJunctions():
    centerDefiningCellsDict = {('WT inflorescence meristem', 'P1'): {"T0": [618, 467, 570], "T1": [5048, 5305], "T2": [5849, 5601], "T3": [6178, 6155, 6164], "T4": [6288, 6240]},
                               ('WT inflorescence meristem', 'P2'): {"T0": [392], "T1": [553, 779, 527], "T2": [525], "T3": [1135], "T4": [1664, 1657]},
                               ('WT inflorescence meristem', 'P5'): {"T0": [38], "T1": [585, 968, 982], "T2": [927, 1017], "T3": [1136], "T4": [1618, 1575, 1445]},
                               ('WT inflorescence meristem', 'P6'): {"T0": [861], "T1": [1334, 1634, 1651], "T2": [1735, 1762, 1803], "T3": [2109, 2176], "T4": [2381]},
                               ('WT inflorescence meristem', 'P8'): {"T0": [3241, 2869, 3044], "T1": [3421, 3657], "T2": [2676, 2805, 2876], "T3": [2898, 2997, 3013], "T4": [358, 189]},
                               ('ktn inflorescence meristem', 'ktnP1'): {"T0": [2965, 3144], "T1": [3839, 3959], "T2": [963, 968, 969]},
                               ('ktn inflorescence meristem', 'ktnP2'): {"T0": [23], "T1": [424, 426, 50], "T2": [17, 127, 220]},
                               ('ktn inflorescence meristem', 'ktnP3'): {"T0": [29, 199, 527], "T1": [424, 28, 431], "T2": [391, 438]},
                               }
    saveCentralCellsAsCellType(centerDefiningCellsDict, imageFolder="Images/SAM/", scenarioName="WT inflorescence meristem")
    saveCentralCellsAsCellType(centerDefiningCellsDict, imageFolder="Images/SAM/", scenarioName="ktn inflorescence meristem")
    mainCreateProcessedDataOf(imageFolder="Images/", scenarioName="SAM/WT inflorescence meristem", multiFolderName="SAM", addToExisiting=False)
    mainCreateProcessedDataOf(imageFolder="Images/", scenarioName="SAM/ktn inflorescence meristem", multiFolderName="SAM", addToExisiting=True)

def mainAddGuardCellJunctionPositions():
    multiFolderContentFilename = "Images/first leaf_multiFolderContent.pkl"
    extract3DContours = True
    cellTypeFilenameKey = "cellType"
    cellTypesBaseName = "_cellTypes.json"
    converter = MGXToFolderContentConverter("", extract3DContours=extract3DContours, runOnInit=False)
    folderContents = MultiFolderContent(multiFolderContentFilename)
    updatedAnyContent = False
    for content in folderContents:
        tissueBaseFilename = content.GetFilenameDictKeyValue(cellTypeFilenameKey).replace(cellTypesBaseName, "")
        cellLabelsPerCellType = content.LoadKeyUsingFilenameDict(cellTypeFilenameKey)
        if not "guard cell" in cellLabelsPerCellType:
            print(f"was ignored as no guard cells present in {tissueBaseFilename}")
            continue
        guardCellLabels = cellLabelsPerCellType["guard cell"]
        if len(guardCellLabels) == 0:
            print(f"was ignored as no guard cells present in {tissueBaseFilename}")
        else:
            print(tissueBaseFilename)
            converter.tissueBaseFilename = tissueBaseFilename
            mgxJunctionReader = converter.addContourInfoTo(content, cellTypesDict=cellLabelsPerCellType,
                                                           extractPeripheralCellsFromJunctions=True,
                                                           plyExtension=converter.plyJunctionNameExtension,
                                                           save=False)
            junctionPositionsOfAllCells = mgxJunctionReader.GetCellsContourPositions()
            converter.addGuardCellJunctionPositionTo(content, junctionPositionsOfAllCells, guardCellLabels)
            updatedAnyContent = True
    if updatedAnyContent:
        folderContents.UpdateFolderContents()

if __name__ == '__main__':
    # run both createGeometricdataTableWithRemovedStomata and mainCreateAllFullCotyledons
    # to use proposed cell labels as guard cells instead of removing stomata in MGX files and
    # creating heatmap from the MGX file which does not contain guard cells
    # createGeometricdataTableWithRemovedStomata()
    # mainCreateAllFullCotyledons()
    #
    # mainCreateProcessedDataOf(scenarioName="first_leaf_LeGloanec2022", multiFolderName="first leaf", addToExisiting=False, checkTimePointsForRemovalOfSmallCells=True)
    # mainCreateProcessedDataOf(scenarioName="speechless_Fox2018_MGXfromLeGloanec", multiFolderName="first leaf", addToExisiting=True, checkTimePointsForRemovalOfSmallCells=False)
    # addGeometricDataFilenameAndKey(allFolderContentsFilename="Images/first leaf_multiFolderContent.pkl", checkTimePointsForRemovalOfSmallCells=True)
    # addGeometricDataFilenameAndKey("Images/full cotyledons/full cotyledons.pkl", geometricTableBaseName="{}_geometricData no stomata.csv", timePointInPath=False)
    # addGeometricDataFilenameAndKey("Images/Matz2022SAM.pkl", seperator="", geometricTableBaseName="area{}.csv", polygonGeometricTableBaseName="area{} poly.csv")
    # mainInitalizeSAMDataAddingContoursAndJunctions()
    # mainCreateProcessedDataOf(imageFolder="Images/", scenarioName="SAM/WT inflorescence meristem", multiFolderName="SAM", addToExisiting=False, redoForTissueInfo=np.array([["P2", "T1"]]))
    mainAddGuardCellJunctionPositions()