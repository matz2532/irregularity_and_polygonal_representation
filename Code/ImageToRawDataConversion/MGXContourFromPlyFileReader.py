import numpy as np
import pandas as pd
import pickle
import re
import sys
import warnings

sys.path.insert(0, "./Code/DataStructures/")
from FolderContent import FolderContent
from pathlib import Path

class MGXContourFromPlyFileReader (object):

    # default not run variables
    pointCloud=None
    pointCloudPositions=None
    dictOfCellsContourIndices=None
    dictOfCellsContourPositions=None
    # default filename extension (for saving)
    resultsNameExtension="orderedJunctionsPerCell.pkl"
    # default patterns in ply file
    endHeaderPattern="end_header"
    beforeContourPattern="element vertex (\d+)"
    lineSeperator=" "
    # default indices of indices
    coordinateIndicesOfContours2D=[0, 1]
    coordinateIndicesOfContours3D=[0, 1, 2]
    minValues=None

    def __init__(self, junctionPlyFilename=None, extract3DContours=False):
        if extract3DContours:
            self.coordinateIndicesOfContours = self.coordinateIndicesOfContours3D
        else:
            self.coordinateIndicesOfContours = self.coordinateIndicesOfContours2D
        if not junctionPlyFilename is None:
            self.readContoursFromPlyFile(junctionPlyFilename)

    def GetCellsContourIndices(self):
        return self.dictOfCellsContourIndices

    def GetCellsContourPositions(self):
        return self.dictOfCellsContourPositions

    def GetPointCloud(self):
        return self.pointCloud

    def GetPointCloudPoints(self):
        return self.pointCloudPositions

    def GetZeroingPoint(self):
        return self.minValues

    def ExtractPeripheralCellLabelsFromJunctions(self):
        contourIdxToCellAdjacency = self.createContourIdxToCellNeighborhood(self.dictOfCellsContourIndices)
        peripheralCells = self.determineCellsHavingContourPointsWithFewerCells(contourIdxToCellAdjacency, numberOfCells=3)
        return peripheralCells

    def createContourIdxToCellNeighborhood(self, dictOfCellsContourIndices):
        contourIdxToCellAdjacency = {}
        for cellLabel, contourIndicesOfCell in dictOfCellsContourIndices.items():
            for contourIdx in contourIndicesOfCell:
                if not contourIdx in contourIdxToCellAdjacency:
                    contourIdxToCellAdjacency[contourIdx] = [cellLabel]
                else:
                    contourIdxToCellAdjacency[contourIdx].append(cellLabel)
        return contourIdxToCellAdjacency

    def determineCellsHavingContourPointsWithFewerCells(self, contourIdxToCellAdjacency, numberOfCells=3):
        cellsBoarderingContoursWithFewer = []
        for contourIdx, neighboringCells in contourIdxToCellAdjacency.items():
            if len(neighboringCells) < numberOfCells:
                cellsBoarderingContoursWithFewer.extend(neighboringCells)
        cellsBoarderingContoursWithFewer = pd.unique(cellsBoarderingContoursWithFewer)
        return cellsBoarderingContoursWithFewer

    def SaveCellsContoursPositions(self, filenameToSave=None,
                                   tissueBaseFilename=None,
                                   resultsNameExtension=None,
                                   nameIdx=-1, onlyKeepCellLabels=None,
                                   cellSizesOfCellsToKeep=None):
        if filenameToSave is None:
            if resultsNameExtension is None:
                resultsNameExtension = self.resultsNameExtension
            if not tissueBaseFilename is None:
                filenameToSave = tissueBaseFilename + resultsNameExtension
            else:
                parentOfFilename = Path(self.junctionPlyFilename).parent
                if not nameIdx is None:
                    extraName = parentOfFilename.parts[nameIdx]
                    if resultsNameExtension[0] != "_":
                        extraName = extraName + "_"
                else:
                    extraName = ""
                filenameToSave = parentOfFilename.joinpath(extraName + resultsNameExtension)
        if not onlyKeepCellLabels is None:
            cellLabelsWithContourPositions = list(self.dictOfCellsContourPositions.keys())
            isCellToKeepPresent = np.isin(onlyKeepCellLabels, cellLabelsWithContourPositions)
            if not cellSizesOfCellsToKeep is None and not np.all(isCellToKeepPresent):
                notPresentCells = np.array(onlyKeepCellLabels)[np.invert(isCellToKeepPresent)]
                areasOfNotPresentCells = np.array(cellSizesOfCellsToKeep)[np.invert(isCellToKeepPresent)]
                smallCellThreshold = np.percentile(np.array(cellSizesOfCellsToKeep)[isCellToKeepPresent], 10)
                isToSmall = areasOfNotPresentCells < smallCellThreshold
                onlyKeepCellLabels = np.array(cellLabelsWithContourPositions)[isCellToKeepPresent]
                if not np.all(isToSmall):
                    warnings.warn(f"The following cell/s from {self.junctionPlyFilename=} don't have contour positions\n{list(notPresentCells)} and\nof those these cells {list(notPresentCells[np.invert(isToSmall)])} with areas {list(areasOfNotPresentCells[np.invert(isToSmall)])} dont fall bellow the area threshold of {smallCellThreshold}\nand could not be saved under:\n{filenameToSave}\nonly the following are present:\n{list(np.sort(cellLabelsWithContourPositions))}\nThe cells could be to small or in terms of polygonal cells (only junctions) are no valid polygons with just two junctions.")
            else:
                if not np.all(isCellToKeepPresent):
                    print(f"WARNING: The following cell/s from {self.junctionPlyFilename=} don't have contour positions\n{list(np.sort( np.array(onlyKeepCellLabels)[np.invert(isCellToKeepPresent)] ))}\nand could not be saved under:\n{filenameToSave}\nonly the following are present:\n{list(np.sort(cellLabelsWithContourPositions))}\nThe cells could be to small or in terms of polygonal cells (only junctions) are no valid polygons with just two junctions.")
                    onlyKeepCellLabels = list(np.array(onlyKeepCellLabels)[isCellToKeepPresent])
            dictOfCellsContourPositions = {}
            for cellLabel in onlyKeepCellLabels:
                dictOfCellsContourPositions[cellLabel] = self.dictOfCellsContourPositions[cellLabel]
        else:
            dictOfCellsContourPositions = self.dictOfCellsContourPositions
        FolderContent().SaveDataFilesTo(dictOfCellsContourPositions, filenameToSave)
        return filenameToSave


    def readContoursFromPlyFile(self, junctionPlyFilename, sep=None):
        assert Path(junctionPlyFilename).is_file(), f"The ply.-file {junctionPlyFilename} does not exist and it's cell contours/junctions can not be extracted."
        self.junctionPlyFilename = junctionPlyFilename
        self.dictOfCellsContourIndices, self.pointCloudPositions = self.extractCellsContourIndicesAndPoints(junctionPlyFilename, sep=sep)
        self.makeContourPositionsPositive(self.pointCloudPositions)
        self.dictOfCellsContourPositions = self.extractContourPositionsOfCells(self.pointCloudPositions, self.dictOfCellsContourIndices)

    def extractCellsContourIndicesAndPoints(self, junctionPlyFilename, sep=None):
        if sep is None:
            sep = self.lineSeperator
        dictOfCellsContourIndices = {}
        pointCloudPositions = []
        lineCountAfterHeader = 0
        linesBeforeContourIndicesOfCells = None
        isAfterHeader = False
        with open(junctionPlyFilename, "r") as fh:
            line = fh.readline()
            while line:
                if isAfterHeader:
                    line = line.strip()
                    splitLine = line.split(sep)
                    if lineCountAfterHeader >= linesBeforeContourIndicesOfCells:
                        nrOfContours = int(splitLine.pop(0))
                        label = int(splitLine.pop(-1))
                        contourIndices = [int(i) for i in splitLine]
                        dictOfCellsContourIndices[label] = contourIndices
                    else:
                        splitLine = [float(i) for i in splitLine]
                        pointCloudPositions.append(splitLine)
                        if len(pointCloudPositions) > 0:
                            assert len(splitLine) == len(pointCloudPositions[0]), f"At line {lineCountAfterHeader=} of {junctionPlyFilename=} with the {line=} there are not equal lengths {len(splitLine)} != {len(pointCloudPositions[0])}"
                    lineCountAfterHeader += 1
                else:
                    line = line.strip()
                    isLineHeader = re.search(self.endHeaderPattern, line)
                    searchForBeforeContourIndicesOfCells = re.search(self.beforeContourPattern, line)
                    if isLineHeader:
                        isAfterHeader = True
                    if searchForBeforeContourIndicesOfCells:
                        linesBeforeContourIndicesOfCells = int(searchForBeforeContourIndicesOfCells.group(1))
                line = fh.readline()
        pointCloudPositions = pd.DataFrame(pointCloudPositions)
        return dictOfCellsContourIndices, pointCloudPositions

    def makeContourPositionsPositive(self, pointCloudPositions):
        self.minValues = pointCloudPositions.min(axis=0).to_numpy()
        pointCloudPositions -= self.minValues
        return pointCloudPositions

    def extractContourPositionsOfCells(self, allContourPositions, dictOfCellsContourIndices):
        dictOfCellsContourPositions = {}
        for label, contourIndices in dictOfCellsContourIndices.items():
            pointCloudPositions = allContourPositions.iloc[contourIndices, self.coordinateIndicesOfContours].to_numpy()
            dictOfCellsContourPositions[label] = pointCloudPositions
        return dictOfCellsContourPositions

def main():
    import matplotlib.pyplot as plt
    tissueReplicateId = "20200220 WT S1"
    baseName = f"Images/full cotyledons/WT/{tissueReplicateId}/{tissueReplicateId}"
    extractCellContours = True
    save = False
    if extractCellContours:
        junctionPlyFilename = baseName + "_only pavement cells.ply"
        resultsNameExtension = "cellContour.pkl"
    else:
        junctionPlyFilename = baseName + "_only junctions.ply"
        resultsNameExtension = None
    myMGXContourFromPlyFileReader = MGXContourFromPlyFileReader(junctionPlyFilename)
    if save:
        filename = myMGXContourFromPlyFileReader.SaveCellsContoursPositions(resultsNameExtension=resultsNameExtension)
    cellContours = myMGXContourFromPlyFileReader.GetCellsContourPositions()
    if extractCellContours:
        highlightedPoints = baseName + "_orderedJunctionsPerCell.pkl"
        with open(highlightedPoints, "rb") as fh:
            highlightedPoints = pickle.load(fh)
    else:
        highlightedPoints = None
    fig, ax = plt.subplots(1, 1)
    for cellLabel, cellsContour in cellContours.items():
        x, y = cellsContour[:, 0], cellsContour[:, 1]
        x = np.concatenate([x, [x[0]]])
        y = np.concatenate([y, [y[0]]])
        ax.plot(x, y)
        if not highlightedPoints is None:
            if cellLabel in  highlightedPoints:
                ax.scatter(highlightedPoints[cellLabel][:, 0], highlightedPoints[cellLabel][:, 1])
    plt.show()

def correctlyAddOutlines():
    outlinePlyExtension = "_just outlines.ply"
    contourNameExtension = "_cellContour.pkl"
    baseImageFolder = "Images"
    allContourPlyBaseFilenames = []
    folderExtension = "first_leaf_LeGloanec2022"
    folderExtension = "full cotyledons"
    useFullCotyledonFormating = True
    # for plantName in ["01", "02", "03"]:
    for tissueName, plantName in zip(["ktn1-2"],
                                     ["20200220 ktn1-2 S2"]):
        for timePoint in ["120h", ]:
            # Images\full cotyledons\ktn1-2\20200220 ktn1-2 S2\20202020 ktn1-2 S2_just outlines
            # Images\full cotyledons\ktn1-2\20200220 ktn1-2 S2\20200220 ktn1-2 S2_just outlines.ply
            # for timePoint in ["1.5DAI", "2DAI", "2.5DAI", "3DAI", "3.5DAI", "4DAI", "4.5DAI", "5DAI", "5.5DAI", ]:
            if useFullCotyledonFormating:
                baseNameFormater = "{}"
                baseName = str(Path(baseImageFolder).joinpath(folderExtension, tissueName, plantName, baseNameFormater.format(plantName)))
            else:
                baseNameFormater = "{}_{}"
                baseName = str(Path(baseImageFolder).joinpath(folderExtension, plantName, timePoint, baseNameFormater.format(plantName, timePoint)))
            allContourPlyBaseFilenames.append(baseName)
    for tissueBaseFilename in allContourPlyBaseFilenames:
        print(tissueBaseFilename)
        junctionPlyFilename = tissueBaseFilename + outlinePlyExtension
        print(junctionPlyFilename)
        if not Path(junctionPlyFilename).is_file():
            print(f"The file {junctionPlyFilename} does not exist")
            continue
        contourFilename = tissueBaseFilename + contourNameExtension
        myMGXContourFromPlyFileReader = MGXContourFromPlyFileReader(junctionPlyFilename, extract3DContours=True)
        originalContours = pickle.load(open(contourFilename, "rb"))
        onlyKeepCellLabels = list(originalContours.keys())
        myMGXContourFromPlyFileReader.SaveCellsContoursPositions(filenameToSave=contourFilename, onlyKeepCellLabels=onlyKeepCellLabels)
        sys.exit()

if __name__ == '__main__':
    correctlyAddOutlines()
    # main()
