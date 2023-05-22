import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.ndimage as ndimage
import sys

sys.path.insert(0, "./Code/Analyze/")
sys.path.insert(0, "./Code/DataStructures/")
sys.path.insert(0, "./Code/MeasureCreator/")
from BasePlotter import BasePlotter
from MultiFolderContent import MultiFolderContent
from PatchCreator import PatchCreator
from pathlib import Path
from plottingUtils import createColorMapper
from PolygonalRegularityCalculator import PolygonalRegularityCalculator

class CellIrregualrityBasedSelector (BasePlotter):

    axesLength=None
    cellOutlineKey="zeroedCellOutline",
    cellOutlineFilenameKey="cellContours"
    orderedJunctionsKey="orderedJunctionsPerCellFilename"
    defaultOrderedJunctionsEdgeColor="#0072b2a0"
    resultsTable=None
    xLim=None

    def __init__(self, resultsTableFilename, allFolderContentsFilename, labelNameConverterDict=None):
        self.resultsTable = pd.read_csv(resultsTableFilename)
        self.multiFolderContent = MultiFolderContent(allFolderContentsFilename)
        if not labelNameConverterDict is None:
            self.SetLabelNameConverterDict(labelNameConverterDict)

    def SetDefaultOrderedJunctionsEdgeColor(self, defaultOrderedJunctionsEdgeColor):
        self.defaultOrderedJunctionsEdgeColor = defaultOrderedJunctionsEdgeColor

    def SetResultsTable(self, resultsTable):
        self.resultsTable = resultsTable

    def GetAxesLength(self):
        return self.axesLength

    def GetXLim(self):
        return self.xLim

    def PlotOutlinesAndPolygonOf(self, measureColumn, cellPropertyOverview=None, ax=None, showPlot=True,
                                 figSizeFactor=4, baseXOffset=2, dynamicallyRoundYTicks=None, roundToDecimals=2,
                                 useMapper=True, useLinearScale=False, lengthPerCell=2, startXLim=0,
                                 forceValueLimTo=None, useRegularXTicks=False, setXTicks=None, usePropertiesValues=False,
                                 ignoreOutline=False, saveToFilename=None, useGivenXLim=None, useAxesLength=None,
                                 fontSize=20, edgeLineWidth=3, setXLabel=True, vertexSize=None, includeMean=True):
        if usePropertiesValues:
            if cellPropertyOverview is None:
                cellPropertyOverview = self.cellPropertyOverview
            selectedValues = [prop.GetCellularProperty(measureColumn) for prop in cellPropertyOverview.values()]
        else:
            assert not self.resultsTable is None, f"There are no {measureColumn=} given, but the resultsTable {self.resultsTable=} is also empty."
            cellPropertyOverview = self.extractCorrespondingCellsPropertiesOf(measureColumn=measureColumn, includeMean=includeMean)
            selectedValues = self.resultsTable[measureColumn]
        if forceValueLimTo is None:
            valueLim = (np.min(selectedValues), np.max(selectedValues))
        else:
            valueLim = forceValueLimTo
        nrOfCells = len(cellPropertyOverview)
        if useLinearScale:
            self.axesLength = lengthPerCell * nrOfCells
            if not useAxesLength is None:
                self.axesLength = useAxesLength
            valueToXOffsetMapper = self.createFunctionToMapValueToOffsetAlongAxes(valueLim, self.axesLength, startXLim)
        plt.rcParams["font.size"] = fontSize
        plt.rcParams["font.family"] = "Arial"
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=[figSizeFactor*nrOfCells, figSizeFactor], constrained_layout=True)
        xTickPos = []
        xTickName = []
        xOffset = 0
        if useMapper:
            measureValues = [cellProperty.GetCellularProperty(measureColumn) for cellProperty in cellPropertyOverview.values()]
            mapper = createColorMapper(valueLim)
        for i, (positionProp, cellProperty) in enumerate(cellPropertyOverview.items()):
            title = positionProp
            measureValue = cellProperty.GetCellularProperty(measureColumn)
            if useMapper:
                orderedJunctionsEdgeColor = mapper.to_rgba(measureValue)
            else:
                orderedJunctionsEdgeColor = self.defaultOrderedJunctionsEdgeColor
            if useLinearScale:
                xOffset = valueToXOffsetMapper(measureValue)
            self.plotCellularOutlineAndPolygon(ax, cellProperty, xOffset, orderedJunctionsEdgeColor, ignoreOutline=ignoreOutline,
                                               edgeLineWidth=edgeLineWidth, vertexSize=vertexSize)
            xTickPos.append(xOffset)
            xTickName.append(cellProperty.GetCellularProperty(measureColumn))
            xOffset += baseXOffset
        if not dynamicallyRoundYTicks is None:
            roundToDecimals = self.determineRoundingDecimalPlace(xTickName, dynamicallyRoundYTicks)
        xTickName = np.round(xTickName, roundToDecimals)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        if useGivenXLim is None:
            self.setLimitsForOutlines(ax, cellPropertyOverview, ignoreOutline=ignoreOutline, xOffset=np.max(xTickPos))
        else:
            ax.set_xlim(useGivenXLim)
        if useRegularXTicks:
            ax.set_xticks(xTickPos, xTickName)
        else:
            if not setXTicks is None:
                argMaxPos = np.argmax(xTickPos)
                if xTickPos[argMaxPos] == 0:
                    argMaxPos -= 1
                xTickPosConversion = xTickPos[argMaxPos] / xTickName[argMaxPos]
                xTickPos, xTickName = [xTickPosConversion * tickName for tickName in setXTicks], setXTicks
            ax.set_xticks(xTickPos)
            ax.set_xticklabels(xTickName)
        ax.set_yticks([])
        if setXLabel:
            ax.set_xlabel(self.GetLabelOfMeasure(measureColumn))
        if useMapper:
            colorBar = plt.colorbar(mapper)
        if not saveToFilename is None:
            Path(saveToFilename).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(saveToFilename, bbox_inches="tight", dpi=300)
            plt.close()
        else:
            if showPlot:
                plt.show()
        return cellPropertyOverview

    def extractCorrespondingCellsPropertiesOf(self, measureColumn, includeMean=True):
        selectedValues = self.resultsTable[measureColumn]
        argmin, argmax = np.argmin(selectedValues), np.argmax(selectedValues)
        argmean = self.closestArgmean(selectedValues)
        cellPropertyOfArgmin = self.extractFolderContentOfTableIdx(argmin, measureColumn)
        cellPropertyOfArgmax = self.extractFolderContentOfTableIdx(argmax, measureColumn)
        cellPropertyOfArgmean = self.extractFolderContentOfTableIdx(argmean, measureColumn)
        if includeMean:
            return {"min":cellPropertyOfArgmin, "mean":cellPropertyOfArgmean, "max":cellPropertyOfArgmax}
        else:
            return {"min":cellPropertyOfArgmin, "max":cellPropertyOfArgmax}

    def closestArgmean(self, values):
        mean = np.mean(values)
        values = np.array(values)
        argmean = np.argmin(np.abs(values - mean))
        return argmean

    def extractFolderContentOfTableIdx(self, idx, measureColumn, genotypeColumn="genotype",
                                       replicateIdColumn="replicate id", timePointColumn="time point",
                                       cellLabelColumn="cell label", idxKey="idx"):
        genotype = self.resultsTable[genotypeColumn].iloc[idx]
        replicateId = self.resultsTable[replicateIdColumn].iloc[idx]
        timePoint = self.resultsTable[timePointColumn].iloc[idx]
        cellLabel = self.resultsTable[cellLabelColumn].iloc[idx]
        folderContent = self.multiFolderContent.GetFolderContentOfReplicateAtTimePoint(replicateId, timePoint)
        cellProperty = CellProperty(genotype, replicateId, timePoint, cellLabel, folderContent)
        cellOutline = self.loadCellLabelFromWithFilenameKey(cellLabel, folderContent, self.cellOutlineFilenameKey)
        cellProperty = self.addMovedAndScaledPointArray(cellProperty, cellOutline, self.cellOutlineKey)
        junctionPoints = self.loadCellLabelFromWithFilenameKey(cellLabel, folderContent, self.orderedJunctionsKey)
        cellProperty = self.addMovedAndScaledPointArray(cellProperty, junctionPoints, self.orderedJunctionsKey,
                                                        shift=cellProperty.GetCellularProperty("shiftedBy"),
                                                        scaledDownByValue=cellProperty.GetCellularProperty("scaledDownBy"))
        measureValue = self.resultsTable[measureColumn].iloc[idx]
        cellProperty.AddCellularProperty(measureColumn, measureValue)
        cellProperty.AddCellularProperty(idxKey, idx)
        return cellProperty

    def loadCellLabelFromWithFilenameKey(self, cellLabel, folderContent, filenameKey):
        cellOutlines = folderContent.LoadKeyUsingFilenameDict(filenameKey, convertDictKeysToInt=True, convertDictValuesToNpArray=True)
        assert cellLabel in cellOutlines, f"The {cellLabel=} is not present in the cellOutlines of {folderContent}\nThe keys {list(cellOutlines.keys())} exist."
        return cellOutlines[cellLabel]

    def addMovedAndScaledPointArray(self, cellProperty, positionArray, positionArrayKey, shift=None, scaledDownByValue=None):
        if shift is None:
            # outlineImage = np.zeros(np.max(positionArray, axis=0)+1)
            # for x,y in positionArray:
            #     outlineImage[x,y]=1
            # shift = np.asarray(ndimage.center_of_mass(outlineImage))
            shift = np.mean(positionArray, axis=0)
            cellProperty.AddCellularProperty("shiftedBy", shift)
        positionArray = positionArray.astype(float) - shift
        if scaledDownByValue is None:
            scaledDownByValue = 0.5 * np.abs(np.max(positionArray[:, 0]) - np.min(positionArray[:, 0]))
            cellProperty.AddCellularProperty("scaledDownBy", scaledDownByValue)
        positionArray /= scaledDownByValue
        cellProperty.AddCellularProperty(positionArrayKey, positionArray)
        return cellProperty

    def createFunctionToMapValueToOffsetAlongAxes(self, valueLimits, axesLength, startLim):
        m = axesLength / (valueLimits[1] - valueLimits[0])
        n = startLim
        return lambda x: m*(x-valueLimits[0]) + n

    def plotCellularOutlineAndPolygon(self, ax, cellProperty, xOffset=0, orderedJunctionsEdgeColor="red", edgeLineWidth=3,
                                      ignoreOutline=False, vertexSize=None, decreasePolygonLineWidth=1):
        cellLabel = cellProperty.GetCellLabel()
        if not ignoreOutline:
            outline = cellProperty.GetCellularProperty(self.cellOutlineKey)
            outline[:, 0] += xOffset
            PatchCreator().PlotPatchesFromOutlineDictOn(ax, {cellLabel:outline}, flipOutlineCoordinates=False, edgecolor="#000000", linewidth=edgeLineWidth)
        orderedJunctionPosition = cellProperty.GetCellularProperty(self.orderedJunctionsKey)
        orderedJunctionPosition[:, 0] += xOffset
        PatchCreator().PlotPatchesFromOutlineDictOn(ax, {cellLabel:orderedJunctionPosition}, flipOutlineCoordinates=False, edgecolor=orderedJunctionsEdgeColor, linewidth=edgeLineWidth-decreasePolygonLineWidth)
        ax.scatter(orderedJunctionPosition[:, 0], orderedJunctionPosition[:, 1], color="#009e73ff", s=vertexSize)

    def determineRoundingDecimalPlace(self, values, includeSpacesAfter=2):
        decimalPlace = 0
        for i in values:
            if i != 0:
                currentRounding = np.ceil(- np.log10(i))
                if currentRounding > decimalPlace:
                    decimalPlace = currentRounding
        decimalPlace = int(decimalPlace + includeSpacesAfter - 1)
        if decimalPlace < 0:
            decimalPlace = 0
        return decimalPlace

    def setLimitsForOutlines(self, ax, cellPropertyOverview, increaseAxesLimitByFactor=.15, ignoreOutline=False, xOffset=0):
        xMin, xMax, yMin, yMax = np.inf, -np.inf, np.inf, -np.inf
        for cellProperty in cellPropertyOverview.values():
            if ignoreOutline:
                outline = cellProperty.GetCellularProperty(self.orderedJunctionsKey)
            else:
                outline = cellProperty.GetCellularProperty(self.cellOutlineKey)
            currentXMin, currentYMin = np.min(outline, axis=0)
            currentXMax, currentYMax = np.max(outline, axis=0)
            if currentXMin < xMin:
                xMin = currentXMin
            if currentXMax > xMax:
                xMax = currentXMax
            if currentYMin < yMin:
                yMin = currentYMin
            if currentYMax > yMax:
                yMax = currentYMax
        self.xLim = [xMin, xMax]
        yLim = [yMin, yMax]
        if not increaseAxesLimitByFactor is None and increaseAxesLimitByFactor != 0:
            self.xLim = [self.xLim[0]-increaseAxesLimitByFactor * np.abs(self.xLim[0]), self.xLim[1]+increaseAxesLimitByFactor * np.abs(self.xLim[1])]
            yLim = [yLim[0]-increaseAxesLimitByFactor * np.abs(yLim[0]), yLim[1]+increaseAxesLimitByFactor * np.abs(yLim[1])]
        if np.abs(xOffset) > 0:
            self.xLim[1] += xOffset
        ax.set_xlim(self.xLim)
        ax.set_ylim(yLim)

class CellProperty:

    def __init__(self, genotype=None, replicateId=None, timePoint=None, cellLabel=None, folderContent=None):
        self.genotype, self.replicateId, self.timePoint, self.cellLabel = genotype, replicateId, timePoint, cellLabel
        self.folderContent = folderContent
        self.cellularProperty = {}

    def __str__(self):
        text = f"The cell label {self.cellLabel} of genotype {self.genotype} from {self.replicateId} at {self.timePoint} and\n contains the keys of the cellular properties {list(self.cellularProperty.keys())}."
        return text

    def GetCellLabel(self):
        return self.cellLabel

    def GetCellularProperty(self, propertyKey):
        assert propertyKey in self.cellularProperty, f"The {propertyKey=} does not exist in the cellular property dictionary. The keys {list(self.cellularProperty.keys())} exist."
        return copy.deepcopy(self.cellularProperty[propertyKey])

    def AddCellularProperty(self, propertyKey, propertyValue):
        self.cellularProperty[propertyKey] = propertyValue


def selectCellsTableOf(valueForGrouping, ofTable, columnsToGroupPerCell=["genotype", "replicate id", "cell id"]):
    isRowMatching = np.full(len(ofTable), True)
    for value, column in zip(valueForGrouping, columnsToGroupPerCell):
        isRowMatching = (ofTable[column] == value) & isRowMatching
    return ofTable[isRowMatching]


def sortByMaxDiffInFirstAndNoDiffInSecond(plotterClass, allRegularityMeasures, columnsToGroupPerCell=["genotype", "replicate id", "cell id"]):
    resultsTable = plotterClass.resultsTable
    allScorings, cellInfos = [], []
    for cellGroupId, individualCellsTable in resultsTable.groupby(columnsToGroupPerCell):
        plotterClass.SetResultsTable(individualCellsTable)
        minMaxMeanCellProperties = plotterClass.extractCorrespondingCellsPropertiesOf(allRegularityMeasures[0], includeMean=False)
        for currentCellProperty in minMaxMeanCellProperties.values():
            idx = currentCellProperty.GetCellularProperty("idx")
            regularityValue = individualCellsTable.iloc[idx, :][allRegularityMeasures[1]]
            currentCellProperty.AddCellularProperty(allRegularityMeasures[1], regularityValue)
        minFirstRegularityValue = np.abs(minMaxMeanCellProperties["min"].GetCellularProperty(allRegularityMeasures[0]))
        maxFirstRegularityValue = np.abs(minMaxMeanCellProperties["max"].GetCellularProperty(allRegularityMeasures[0]))
        if minFirstRegularityValue < maxFirstRegularityValue:
            firstRegDifference = maxFirstRegularityValue - minFirstRegularityValue
        else:
            firstRegDifference = minFirstRegularityValue - maxFirstRegularityValue
        firstRegularityValues = [v.GetCellularProperty(allRegularityMeasures[0]) for v in minMaxMeanCellProperties.values()]
        giniOfFirstRegs = PolygonalRegularityCalculator().calcGiniCoefficient(firstRegularityValues)

        secondRegularityValues = [v.GetCellularProperty(allRegularityMeasures[1]) for v in minMaxMeanCellProperties.values()]
        giniOfSecondRegs = PolygonalRegularityCalculator().calcGiniCoefficient(secondRegularityValues)
        differentButNotValue = giniOfFirstRegs / giniOfSecondRegs
        allScorings.append(differentButNotValue)
        cellInfos.append(cellGroupId)
    allScorings = np.asarray(allScorings)
    cellInfos = np.asarray(cellInfos, dtype=object)
    argsortOfScoring = np.argsort(allScorings)[::-1]
    return allScorings[argsortOfScoring], cellInfos[argsortOfScoring]


def plotScaleComparison(plotterClass, individualCellsTable, allRegularityMeasures, resultsFolder="", cellGroupId=[], includeMean=True, showPlot=True, savePlot=False, printOutSavedFilename=True):
    import matplotlib.pyplot as plt
    plotKwargs = dict(useLinearScale=True, edgeLineWidth=6, vertexSize=200, saveToFilename=None, useMapper=False, showPlot=False)
    nrOfMeasures = len(allRegularityMeasures)
    fig, ax = plt.subplots(nrOfMeasures, 1, figsize=[8, 8], constrained_layout=True)
    plotterClass.SetResultsTable(individualCellsTable)
    if allRegularityMeasures[0] == "lengthGiniCoeff" and allRegularityMeasures[1] == "angleGiniCoeff":
        comparisonName = "length vs angle irreg "
    elif allRegularityMeasures[0] == "angleGiniCoeff" and allRegularityMeasures[1] == "lengthGiniCoeff":
        comparisonName = "angle vs length irreg "
    else:
        comparisonName = ""
    cellProperties = None
    for i, regularityMeasureName in enumerate(allRegularityMeasures):
        if not cellProperties is None:
            for currentCellProperty in cellProperties.values():
                idx = currentCellProperty.GetCellularProperty("idx")
                regularityValue = individualCellsTable.iloc[idx, :][regularityMeasureName]
                currentCellProperty.AddCellularProperty(regularityMeasureName, regularityValue)
        cellProperties = plotterClass.PlotOutlinesAndPolygonOf(regularityMeasureName, ax=ax[i],
                                                               usePropertiesValues=not cellProperties is None,
                                                               cellPropertyOverview=cellProperties, includeMean=includeMean, **plotKwargs)
    if savePlot:
        filenameToSave = f"{resultsFolder}regularityResults/Scales/comparing scales of {comparisonName}{'_'.join([str(i) for i in cellGroupId])}.png"
        if printOutSavedFilename:
            print(filenameToSave)
        plt.savefig(filenameToSave, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        if showPlot:
            plt.show()


def visualiseRealExampleScales():
    dataBaseFolder = "Images/Eng2021Cotyledons/"
    resultsFolder = "Results/"
    folderContentsName = "Eng2021Cotyledons.pkl"
    tableBaseName = "combinedMeasures_Eng2021Cotyledons.csv"
    allFolderContentsFilename = dataBaseFolder + folderContentsName
    resultsTableFilename = resultsFolder + tableBaseName
    polygonalCellRegularityScalePlotter = CellIrregualrityBasedSelector(resultsTableFilename, allFolderContentsFilename)
    resultsTable = polygonalCellRegularityScalePlotter.resultsTable
    allRegularityMeasures = ["lengthGiniCoeff", "angleGiniCoeff"]
    selectedCellsInfos = ["col-0", "20170501 WT S1", 14]
    selectedTable = selectCellsTableOf(selectedCellsInfos, resultsTable)
    plotScaleComparison(polygonalCellRegularityScalePlotter, selectedTable, allRegularityMeasures, resultsFolder=resultsFolder,
                        cellGroupId=selectedCellsInfos, includeMean=False, savePlot=True)
    allRegularityMeasures = ["angleGiniCoeff", "lengthGiniCoeff"]
    selectedCellsInfos = ['col-0', '20170501 WT S2', 5]
    selectedTable = selectCellsTableOf(selectedCellsInfos, resultsTable)
    plotScaleComparison(polygonalCellRegularityScalePlotter, selectedTable, allRegularityMeasures, resultsFolder=resultsFolder,
                        cellGroupId=selectedCellsInfos, includeMean=False, savePlot=True)


def mainGoThroughOneByOne():
    import matplotlib.pyplot as plt
    dataBaseFolder = "Images/Eng2021Cotyledons/"
    resultsFolder = "Results/"
    folderContentsName = "Eng2021Cotyledons.pkl"
    tableBaseName = "combinedMeasures_Eng2021Cotyledons.csv"
    defaultOrderedJunctionsEdgeColor = "#0072b2ff"
    allFolderContentsFilename = dataBaseFolder + folderContentsName
    resultsTableFilename = resultsFolder + tableBaseName
    allRegularityMeasures = ["lengthGiniCoeff", "angleGiniCoeff", ]

    polygonalCellRegularityScalePlotter = CellIrregualrityBasedSelector(resultsTableFilename, allFolderContentsFilename)
    resultsTable = polygonalCellRegularityScalePlotter.resultsTable
    columnsToGroupPerCell = ["genotype", "replicate id", "cell id"]
    # 4.033264824122963 ['col-0' '20170501 WT S2' 16] strange
    scorings, cellInfos = sortByMaxDiffInFirstAndNoDiffInSecond(polygonalCellRegularityScalePlotter, allRegularityMeasures, columnsToGroupPerCell=columnsToGroupPerCell)
    """
    use example: 25.894623275122758 ['col-0' '20170501 WT S2' 5] for angle increasing, but side length being the same

    """
    for score, cellGroupId in zip(scorings, cellInfos):
        selectedTable = selectCellsTableOf(cellGroupId, resultsTable)
        if cellGroupId[0] == "col-0":
            print(score, cellGroupId)
            plotScaleComparison(polygonalCellRegularityScalePlotter, selectedTable, allRegularityMeasures, resultsFolder=resultsFolder, cellGroupId=cellGroupId, includeMean=False)
    """ sensable examples
    2.019779355818002 ['col-0' '20170501 WT S1' 26]
    5.563333589431028 ['ktn1-2' '20180618 ktn1-2 S1' 11]
    4.1132040973738135 ['Oryzalin' '20171106 oryzalin S1' 19]
    2.6529246828429924 ['col-0' '20170327 WT S1' 15]

    5.563333589431028 ['ktn1-2' '20180618 ktn1-2 S1' 11]
    5.297734557543829 ['ktn1-2' '20180618 ktn1-2 S2' 27]
    """
    sys.exit()
    polygonalCellRegularityScalePlotter.SetDefaultOrderedJunctionsEdgeColor(defaultOrderedJunctionsEdgeColor)
    for cellGroupId, individualCellsTable in resultsTable.groupby(columnsToGroupPerCell):
        plotScaleComparison(polygonalCellRegularityScalePlotter, individualCellsTable, allRegularityMeasures, resultsFolder=resultsFolder, cellGroupId=cellGroupId)


def main():
    dataBaseFolder = "Images/Eng2021Cotyledons/"
    resultsFolder = "Results/"
    folderContentsName = "Eng2021Cotyledons.pkl"
    tableBaseName = "combinedMeasures_Eng2021Cotyledons.csv"
    measure = "lengthRegularity"
    allFolderContentsFilename = dataBaseFolder + folderContentsName
    resultsTableFilename = resultsFolder + tableBaseName
    myCellIrregualrityBasedSelector = CellIrregualrityBasedSelector(resultsTableFilename, allFolderContentsFilename)
    myCellIrregualrityBasedSelector.PlotOutlinesAndPolygonOf(measure)


if __name__ == '__main__':
    # main()
    # mainGoThroughOneByOne()
    visualiseRealExampleScales()
