import json
import networkx as nx
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, "./Code/SAM_division_prediction/")

from GraphCreatorFromAdjacencyList import GraphCreatorFromAdjacencyList
from pathlib import Path

class CellTypeSelector (object):

    # indices of columns
    cellLabelColIdxOfGeometricData=0
    valueColIdx=1
    # cell type definition
    firstCellType="pavement cell"
    secondCellType="guard cell"
    # filename extensions (used for loading)
    geometricTableBaseName="_geometricData no stomata.csv"
    topologyBaseName="_cell neighborhood 2D full.csv"
    # default table loading parameter
    sep=","
    skipFooter=4
    # default values for loading cell types from table
    stomataTypeId=16
    cellLabelColIdx=0
    cellTypeIdColIdx=1
    # value to change wheter warnings are printed out (the smaller the less prints verbosity=0 means that no prints happen)
    verbosity=1

    def __init__(self, tissueBaseFilename=None, tissuesGraphFilename=None, tableOfFirstCellTypeFilename=None,
                 cellTypeTableFilename=None, runOnInit=True, seperateSmallCells=False, **kwargs):
        self.tissueBaseFilename = tissueBaseFilename
        self.tissuesGraphFilename = tissuesGraphFilename
        self.tableOfFirstCellTypeFilename = tableOfFirstCellTypeFilename
        self.SetDefaultParameter(kwargs)
        if runOnInit:
            tableOfFirstCellType = self.loadGeometricData(tableOfFirstCellTypeFilename, self.geometricTableBaseName)
            if not cellTypeTableFilename is None:
                if Path(cellTypeTableFilename).is_file():
                    cellTypesTable = pd.read_csv(cellTypeTableFilename)
                    self.cellTypeDict = self.selectCellTypesFromTable(tableOfFirstCellType, cellTypesTable, seperateSmallCells=seperateSmallCells)
                    self.graphOfTissue = None
                else:
                    if self.verbosity > 0:
                        print(f"The cellTypeDict was initialised empty as the cell type table filename {cellTypeTableFilename} does not exist.")
                    self.cellTypeDict = {self.firstCellType: tableOfFirstCellType.iloc[:, self.cellLabelColIdxOfGeometricData].to_numpy()}
                    self.graphOfTissue = None
            else:
                self.graphOfTissue = self.loadTissueGraph()
                self.cellTypeDict = self.createCellTypes(self.graphOfTissue, tableOfFirstCellType)
        else:
            self.graphOfTissue = None


    def SetDefaultParameter(self, kwargs):
        if "cellLabelColIdxOfGeometricData" in kwargs:
            self.cellLabelColIdxOfGeometricData = kwargs["cellLabelColIdxOfGeometricData"]
        if "valueColIdx" in kwargs:
            self.valueColIdx = kwargs["valueColIdx"]
        if "firstCellType" in kwargs:
            self.firstCellType = kwargs["firstCellType"]
        if "secondCellType" in kwargs:
            self.secondCellType = kwargs["secondCellType"]
        if "geometricTableBaseName" in kwargs:
            self.geometricTableBaseName = kwargs["geometricTableBaseName"]
        if "topologyBaseName" in kwargs:
            self.topologyBaseName = kwargs["topologyBaseName"]
        if "sep" in kwargs:
            self.sep = kwargs["sep"]
        if "skipFooter" in kwargs:
            self.skipFooter = kwargs["skipFooter"]
        if "verbosity" in kwargs:
            self.skipFooter = kwargs["verbosity"]

    def GetCellType(self):
        return self.cellTypeDict

    def SetCellType(self, cellTypeDict):
        # values should be of type np.array otherwise unintended erros could occur
        self.cellTypeDict = cellTypeDict

    def SaveCellTypes(self, filenameToSave=None, baseNameExtension="_cellTypes.json"):
        if filenameToSave is None:
            if self.tissueBaseFilename is None:
                filenameToSave = baseNameExtension
            else:
                filenameToSave = self.tissueBaseFilename + baseNameExtension
        jsonCompatableData = {str(k): v if isinstance(v, list) else v.tolist() for k, v in self.cellTypeDict.items()}
        with open(filenameToSave, "w") as fh:
            json.dump(jsonCompatableData, fh)

    def SaveHeatmapOfCellTypes(self, geometricTableOrFilename=None, filenameToSave=None,
                               geometricTableBaseName=None, heatmapOfCellTypeBaseName="heatmapOfCellType.csv"):
        geometricData = self.loadGeometricData(geometricTableOrFilename, geometricTableBaseName)
        cellLabels = geometricData.iloc[:, self.cellLabelColIdxOfGeometricData]
        for i, (cellTypeName, cellLabelsOfCurrentType) in enumerate(self.cellTypeDict.items()):
            isCellOfType = np.isin(cellLabels, cellLabelsOfCurrentType)
            geometricData.iloc[isCellOfType, self.valueColIdx] = i
            geometricData.iloc[isCellOfType, self.valueColIdx+1] = cellTypeName
        if filenameToSave is None:
            if self.tissueBaseFilename is None:
                filenameToSave = heatmapOfCellTypeBaseName
            else:
                filenameToSave = self.tissueBaseFilename + "_" + heatmapOfCellTypeBaseName
        geometricData.to_csv(filenameToSave, index=False)

    def UpdateCellTypeWith(self, cellLabelsOfCellTypeDict, keepingCellsInType=None):
        allCellLabelsToChange = np.concatenate([v for v in cellLabelsOfCellTypeDict.values()])
        for cellTypeName, cellLabelsOfCurrentType in self.cellTypeDict.items():
            keepCellLabel = np.isin(cellLabelsOfCurrentType, allCellLabelsToChange, invert=True)
            if not np.all(keepCellLabel) and not cellTypeName == keepingCellsInType:
                if type(cellLabelsOfCurrentType) == list:
                    cellLabelsOfCurrentType = np.array(cellLabelsOfCurrentType)
                self.cellTypeDict[cellTypeName] = cellLabelsOfCurrentType[keepCellLabel]
        for cellTypeName, cellLabelsOfCurrentType in cellLabelsOfCellTypeDict.items():
            self.cellTypeDict[cellTypeName] = np.array(cellLabelsOfCurrentType, dtype=int)

    def UpdateCellTypeWithDifferentBetween(self, geometricTableOrFilename=None, substractWithGeometricTableOrFilename=None,
                                           cellTypeName="newCellType", geometricTableBaseName=None, substractWithGeometricTableBaseName=None):
        assert not geometricTableOrFilename is None or not substractWithGeometricTableOrFilename is None or not geometricTableBaseName is None or not substractWithGeometricTableBaseName is None, "At least one of the following arguments needs to be not None: geometricTableOrFilename, substractWithGeometricTableOrFilename, geometricTableBaseName, substractWithGeometricTableBaseName"
        geometricData = self.loadGeometricData(geometricTableOrFilename, geometricTableBaseName)
        geometricDataToSubstractWith = self.loadGeometricData(substractWithGeometricTableOrFilename, substractWithGeometricTableBaseName)
        allCellLabels = geometricData.iloc[:, self.cellLabelColIdxOfGeometricData].to_numpy()
        allCellLabelsToSubstract = geometricDataToSubstractWith.iloc[:, self.cellLabelColIdxOfGeometricData].to_numpy()
        cellLabelsOfNewType = allCellLabelsallCellLabels[np.isin(allCellLabels, allCellLabelsToSubstract, invert=True)]
        cellLabelsOfCellTypeDict = {cellTypeName: cellLabelsOfNewType}
        self.UpdateCellTypeWith(cellLabelsOfCellTypeDict)

    def UpdateCellTypeUsingGeometricDataValues(self, geometricTableOrFilename=None, geometricTableBaseName=None, cellTypeName="newCellType",
                                               newCellTypeHasSmallerThanValue=None, newCellTypeHasHigherThanValue=None, valueColIdx=None):
        assert not newCellTypeHasSmallerThanValue is None or not newCellTypeHasHigherThanValue is None, "At least one of the arguments needs to be a numeric value: newCellTypeHasSmallerThanValue or newCellTypeHasHigherThanValue"
        geometricData = self.loadGeometricData(geometricTableOrFilename, geometricTableBaseName)
        allCellLabels = geometricData.iloc[:, self.cellLabelColIdxOfGeometricData].to_numpy()
        if valueColIdx is None:
            valueColIdx = self.valueColIdx
        valuesOfCells = geometricData.iloc[:, valueColIdx].to_numpy()
        if not newCellTypeHasSmallerThanValue is None:
            isCellOfNewType = valuesOfCells < newCellTypeHasSmallerThanValue
        else:
            isCellOfNewType = valuesOfCells > newCellTypeHasHigherThanValue
        cellLabelsOfNewType = allCellLabels[isCellOfNewType]
        cellLabelsOfCellTypeDict = {cellTypeName: cellLabelsOfNewType}
        self.UpdateCellTypeWith(cellLabelsOfCellTypeDict)

    def UpdateCellTypeUsingTextFile(self, textFilename=None, textBaseName=None, cellTypeName="newCellType", sep=","):
        assert not textFilename is None or not textBaseName is None, "Either the textFilename or textBaseName need to be NOT None."
        if textFilename is None:
            textFilename = self.tissueBaseFilename + textBaseName
        with open(textFilename, "r") as fh:
            file = fh.readlines()
        cellLabelsOfNewType = []
        for line in file:
            currentCellLabels = line.split(sep)
            cellLabelsOfNewType.extend(currentCellLabels)
        cellLabelsOfNewType = np.array(cellLabelsOfNewType, dtype=int)
        cellLabelsOfCellTypeDict = {cellTypeName: cellLabelsOfNewType}
        self.UpdateCellTypeWith(cellLabelsOfCellTypeDict)

    def loadTissueGraph(self):
        if not self.tissuesGraphFilename is None:
            tissuesGraphFilename = self.tissuesGraphFilename
        else:
            tissuesGraphFilename = self.tissueBaseFilename + self.topologyBaseName
        if isinstance(tissuesGraphFilename, (str, Path)):
            graphOfTissue = GraphCreatorFromAdjacencyList(tissuesGraphFilename).GetGraph()
        else:
            graphOfTissue = tissuesGraphFilename
        return graphOfTissue

    def selectCellTypesFromTable(self, geometricData, cellTypesTable, thirdCellType="small cells", seperateSmallCells=False):
        allCellLabels = geometricData.iloc[:, self.cellLabelColIdxOfGeometricData]
        cellTypeDict = {}
        stomataCellLabels = self.selectStomataCellsFromTable(cellTypesTable)
        isNonStomata = np.isin(allCellLabels, stomataCellLabels, invert=True)
        nonStomataGeometricData = geometricData.iloc[isNonStomata, :]
        if seperateSmallCells:
            # select pavement cells >= small cell threshold
            smallCellThreshold = np.percentile(nonStomataGeometricData.iloc[:, self.valueColIdx], 25)
        else:
            smallCellThreshold = nonStomataGeometricData.iloc[:, self.valueColIdx].min() - 1
        isSmallCell = nonStomataGeometricData.iloc[:, self.valueColIdx] < smallCellThreshold
        nonStomataCellLabels = nonStomataGeometricData.iloc[:, self.cellLabelColIdxOfGeometricData].to_numpy()
        cellTypeDict[self.firstCellType] = nonStomataCellLabels[np.invert(isSmallCell)]
        cellTypeDict[self.secondCellType] = np.array(stomataCellLabels)
        cellTypeDict[thirdCellType] = nonStomataCellLabels[isSmallCell]
        return cellTypeDict

    def selectStomataCellsFromTable(self, cellTypesTable):
        stomataCellLabels = []
        if cellTypesTable is None:
            return stomataCellLabels
        for i in range(len(cellTypesTable)):
            cellLabel = cellTypesTable.iloc[i, self.cellLabelColIdx]
            cellType = cellTypesTable.iloc[i, self.cellTypeIdColIdx]
            if cellType == self.stomataTypeId:
                stomataCellLabels.append(cellLabel)
        return stomataCellLabels

    def createCellTypes(self, graphOfTissue, tableOfFirstCellType):
        cellTypeDict = {}
        cellsOfFirstType = tableOfFirstCellType.iloc[:, 0].to_numpy(dtype=int)
        cellTypeDict[self.firstCellType] = cellsOfFirstType
        allCells = np.array(list(graphOfTissue), dtype=int)
        cellsOfSecondType = allCells[np.isin(allCells, cellsOfFirstType, invert=True)]
        cellTypeDict[self.secondCellType] = cellsOfSecondType
        return cellTypeDict

    def loadGeometricData(self, geometricTableOrFilename, geometricTableBaseName=None, sep=None, skipFooter=None):
        if geometricTableOrFilename is None:
            if geometricTableBaseName is None:
                geometricTableBaseName = self.geometricTableBaseName
            geometricTableOrFilename = self.tissueBaseFilename + geometricTableBaseName
        if sep is None:
            sep = self.sep
        if skipFooter is None:
            skipFooter = self.skipFooter
        if isinstance(geometricTableOrFilename, (str, Path)):
            geometricData = pd.read_csv(geometricTableOrFilename, sep=sep,
                                        skipfooter=skipFooter, engine="python")
        else:
            geometricData = geometricTableOrFilename
        return geometricData

def extractLeafCellTypesOf(tissueBaseFilename, tissueGraph=None, fullGeometricTableBaseName="_geometricData full.csv",
                           withoutStomataGeometricTableBaseName="_geometricData no stomata.csv",
                           cellTypesResultsBaseName="_cellTypes.json", extractedPeripheralsExtension="_final",
                           seperateSmallCells=False, saveCellTypes=True, saveControlHeatmap=True, skipfooter=4):
    # is needed to convert from MGX selected cells of text copied from terminal to cell list
    areaValues = pd.read_csv(tissueBaseFilename + withoutStomataGeometricTableBaseName, skipfooter=skipfooter, engine="python")
    if seperateSmallCells:
        # select cut of for which cells smaller than this value are selected as small cells
        smallCellValues = np.percentile(areaValues.iloc[:, 1], 25)
    else:
        smallCellValues = areaValues.iloc[:, 1].min() - 1
    myCellTypeSelector = CellTypeSelector(tissueBaseFilename, tissuesGraphFilename=tissueGraph)
    myCellTypeSelector.UpdateCellTypeUsingGeometricDataValues(cellTypeName="small cells", geometricTableBaseName=withoutStomataGeometricTableBaseName,
                                                              newCellTypeHasSmallerThanValue=smallCellValues)
    if saveCellTypes:
        myCellTypeSelector.SaveCellTypes(baseNameExtension=cellTypesResultsBaseName)
    if saveCellTypes:
        myCellTypeSelector.SaveHeatmapOfCellTypes(geometricTableBaseName=fullGeometricTableBaseName)
    cellTypeDict = myCellTypeSelector.GetCellType()
    return cellTypeDict

def extractLeafCellTypeFromTablessOf(tissueBaseFilename, fullGeometricTableBaseName="_geometricData.csv",
                           cellTypeTableBaseName="_CELL_TYPE.csv", cellTypesResultsBaseName="_cellTypes.json",
                           seperateSmallCells=False, saveCellTypes=True, saveControlHeatmap=True, verbosity=1):
    cellTypeTableFilename = tissueBaseFilename + cellTypeTableBaseName
    myCellTypeSelector = CellTypeSelector(tissueBaseFilename, cellTypeTableFilename=cellTypeTableFilename,
                                          geometricTableBaseName=fullGeometricTableBaseName,
                                          seperateSmallCells=seperateSmallCells)
    if saveCellTypes:
        myCellTypeSelector.SaveCellTypes(baseNameExtension=cellTypesResultsBaseName)
    if saveCellTypes:
        myCellTypeSelector.SaveHeatmapOfCellTypes(geometricTableBaseName=fullGeometricTableBaseName)
    cellTypeDict = myCellTypeSelector.GetCellType()
    return cellTypeDict

def mainSaveTypesOfCellsInLeavesOfAndCreateSelectedCells(baseFolder="Images/full cotyledons/",
                scenarioName="WT", searchPattern="/*/",
                geometricTableBaseName="_geometricData no stomata.csv",
                fullGeometricTableBaseName="_geometricData full.csv",
                peripheralCellsBaseName="_peripheralCells.txt",
                selectedCellsBaseName="_selectedNonPeripheralPavementCells.csv"):
    tissuePaths = Path(baseFolder).glob(f"{scenarioName}{searchPattern}")
    selectedCellsDf = []
    for pathOfTissue in tissuePaths:
        if pathOfTissue.suffix:
            continue
        # extract peripheral cells, saving results as labels seperated by "," under "_peripheralCells_final.txt"
        # and a heat map to control selection under "_peripheralCells_control.txt"
        tissueBaseFilename = pathOfTissue.joinpath(pathOfTissue.stem)
        cellTypeDict = extractLeafCellTypesOf(tissueBaseFilename, peripheralCellsBaseName=peripheralCellsBaseName,
                                              fullGeometricTableBaseName=fullGeometricTableBaseName,
                                              withoutStomataGeometricTableBaseName=geometricTableBaseName,
                                              cellTypesResultsBaseName="cellTypes.json")
        pavementCellIds = cellTypeDict["pavement cell"]
        nrOfPavementCells = len(pavementCellIds)
        tissueReplicateId = np.repeat(pathOfTissue.name, nrOfPavementCells)
        timePoints = np.repeat(120, nrOfPavementCells)
        selectedCellsDf.append(np.array([tissueReplicateId, timePoints, pavementCellIds]).T)
    selectedCellsDf = np.concatenate(selectedCellsDf, axis=0)
    selectedCellsDf = pd.DataFrame(selectedCellsDf, columns=["tissue replicate id", "time point", "cell"])
    selectedCellsDfFilename = Path(baseFolder).joinpath(scenarioName, scenarioName + selectedCellsBaseName)
    selectedCellsDf.to_csv(selectedCellsDfFilename, index=False)

if __name__ == '__main__':
    mainSaveTypesOfCellsInLeavesOfAndCreateSelectedCells()
