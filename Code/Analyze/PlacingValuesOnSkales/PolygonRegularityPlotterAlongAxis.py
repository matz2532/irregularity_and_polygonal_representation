import json
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, "./Code/MeasureCreator/")
from PolygonalRegularityCalculator import PolygonalRegularityCalculator
from CellIrregualrityBasedSelector import CellIrregualrityBasedSelector, CellProperty
from OtherMeasuresCreator import OtherMeasuresCreator

class PolygonRegularityPlotterAlongAxis (CellIrregualrityBasedSelector):

    orderedJunctionsKey="polygon vertices position"
    lobynessKey="lobyness"
    relativeCompletenessKey="relativeCompleteness"

    def __init__(self, jsonPolygonVerticesFilename):
        self.dictOfPolygonsVertices = self.loadPolygon(jsonPolygonVerticesFilename)
        self.cellPropertyOverview = self.createPolygonRepresentationsWithPositionAndRegularityValues()
    
    def PlotCellOutlinesAlongTwoAxis(self, xMeasureName, yMeasureName, cellPropertyOverview=None, 
                                     polygonSize=0.2,
                                     fontSize=None, showPlot=True, filenameToSave=None):
        if cellPropertyOverview is None:
            cellPropertyOverview = self.cellPropertyOverview
        self.SetRcParams(fontSize=fontSize)
        allXValues = [cell.GetCellularProperty(xMeasureName) for cell in self.cellPropertyOverview.values()]
        allYValues = [cell.GetCellularProperty(yMeasureName) for cell in self.cellPropertyOverview.values()]
        xLim = [np.min(allXValues), np.max(allXValues)]
        yLim = [np.min(allYValues), np.max(allYValues)]
        polygonSize = [polygonSize * (xLim[1] - xLim[0]),
                       polygonSize * (yLim[1] - yLim[0])]
        fig, ax = plt.subplots(constrained_layout=True)
        ax.scatter(allXValues, allYValues, c="black", s=2)
        for i, cellProperty in enumerate(cellPropertyOverview.values()):
            centerPosition = [allXValues[i], allYValues[i]]
            self.moveAndScaleCellPosition(cellProperty, centerPosition, polygonSize)
            self.plotCellularOutlineAndPolygon(ax, cellProperty, ignoreOutline=True, vertexSize=5, orderedJunctionsEdgeColor=self.defaultOrderedJunctionsEdgeColor)
        ax.set_xlabel(self.GetLabelOfMeasure(xMeasureName))
        ax.set_ylabel(self.GetLabelOfMeasure(yMeasureName))
        self.SaveOrShowFigure(filenameToSave=filenameToSave, showPlot=showPlot)

    def moveAndScaleCellPosition(self, cellProperty, position, scalePolygonToSize):
        positionArray = cellProperty.GetCellularProperty(self.orderedJunctionsKey)
        polygonSize = np.max(positionArray, axis=0) - np.min(positionArray, axis=0)
        # For the scaling to work like this, the polygon needs to be centered around zero
        scalingFactor = scalePolygonToSize / polygonSize
        positionArray -= np.mean(positionArray, axis=0) # reset polygon center position
        positionArray *= scalingFactor
        positionArray += position
        cellProperty.AddCellularProperty(self.orderedJunctionsKey, positionArray)
    
    def loadPolygon(self, filename):
        with open(filename, "r") as fh:
            file = json.load(fh)
        return file
    
    def createPolygonRepresentationsWithPositionAndRegularityValues(self, addOtherMeasures=False):
        polygonPropertyOverview = {}
        for polygonName, polygonVertices in self.dictOfPolygonsVertices.items():
            polygonVertices = np.array(polygonVertices)
            polygonProperties = CellProperty(cellLabel=polygonName)
            polygonProperties = self.addMovedAndScaledPointArray(polygonProperties, polygonVertices, self.orderedJunctionsKey)
            regularityMetrics = self.calcRegularityMetrics(polygonVertices)
            self.addMeasuresToPolygon(polygonProperties, regularityMetrics)
            if addOtherMeasures:
                otherMeasures = self.calcOtherMeasures(polygonVertices)
                self.addMeasuresToPolygon(polygonProperties, otherMeasures)
            polygonPropertyOverview[polygonName] = polygonProperties
        return polygonPropertyOverview

    def calcRegularityMetrics(self, polygonPositions):
        return PolygonalRegularityCalculator().CalcRegularityDict(polygonPositions)

    def addMeasuresToPolygon(self, cellProperty: CellProperty, measureDict: dict):
        for measureName, measureValue in measureDict.items():
            cellProperty.AddCellularProperty(measureName, measureValue)

    def calcOtherMeasures(self, polygonPositions):
        polygonPositionList = list(polygonPositions)
        lobyness = OtherMeasuresCreator().calcLobyness(polygonPositionList)
        relativeCompleteness = OtherMeasuresCreator().calcRelativeCompletenessFromPointList(polygonPositionList)
        return {self.lobynessKey: lobyness, self.relativeCompletenessKey: relativeCompleteness}

def visualizeCellsAlongAxisMain():
    dataBaseFolder = "Images/Eng2021Cotyledons/"
    resultsFolder = "Results/"
    folderContentsName = "Eng2021Cotyledons.json"
    tableBaseName = "combinedMeasures_Eng2021Cotyledons.csv"
    defaultOrderedJunctionsEdgeColor = "#0072b2ff"
    allFolderContentsFilename = dataBaseFolder + folderContentsName
    resultsTableFilename = resultsFolder + tableBaseName
    allRegularityMeasures = ["lengthGiniCoeff", "angleGiniCoeff"]
    for regularityMeasure in allRegularityMeasures:
        filenameToSave = f"{resultsFolder}regularityResults/Scales/MinMeanMaxScale_{regularityMeasure}.png"
        polygonalCellRegularityScalePlotter = CellIrregualrityBasedSelector(resultsTableFilename, allFolderContentsFilename)
        polygonalCellRegularityScalePlotter.SetDefaultOrderedJunctionsEdgeColor(defaultOrderedJunctionsEdgeColor)
        polygonalCellRegularityScalePlotter.PlotOutlinesAndPolygonOf(regularityMeasure, useLinearScale=True, edgeLineWidth=6, vertexSize=200,
                                                                     saveToFilename=filenameToSave, useMapper=False)

def visualizeArtificialPolygonsAlongAxisMain(showPlot: bool = False, resultsFolder: str = "Results/", filename: str = "Images/ArtificialPolygons/complexPolygons.json",
                                             allRegularityMeasures: list = ["lengthGiniCoeff", "angleGiniCoeff"]):
    for regularityMeasure in allRegularityMeasures:
        filenameToSave = f"{resultsFolder}regularityResults/Scales/ArtificialPolygons Scale of {regularityMeasure}.png"
        myPolygonRegularityPlotterAlongAxis = PolygonRegularityPlotterAlongAxis(filename)
        if showPlot:
            filenameToSave = None
        myPolygonRegularityPlotterAlongAxis.PlotOutlinesAndPolygonOf(regularityMeasure, usePropertiesValues=True, ignoreOutline=True,
                                                                     useLinearScale=True, edgeLineWidth=5, vertexSize=40,
                                                                     fontSize=38, setXTicks=[0, 0.1, 0.2, 0.3, 0.4], setXLabel=False,
                                                                     saveToFilename=filenameToSave)
        
def visualizeArtificialPolygonsAlongTwoAxisMain(showPlot: bool = False, resultsFolder: str = "Results/",
                                                filename = "Images/ArtificialPolygons/complexPolygons.json"):

    allRegularityMeasures = [["lengthGiniCoeff", "angleGiniCoeff"],
                             ["lengthGiniCoeff", "lobyness"],
                             ["lengthGiniCoeff", "relativeCompleteness"],
                             ["angleGiniCoeff", "lobyness"],
                             ["angleGiniCoeff", "relativeCompleteness"],
                             ]
    defaultOrderedJunctionsEdgeColor = "#0072b2ff"
    labelNameConverterDict = {"lengthGiniCoeff": "Gini coefficient of length", "angleGiniCoeff": "Gini coefficient of angle",
                              "lobyness": "Lobyness", "relativeCompleteness": "Relative completeness"}
    myPolygonRegularityPlotterAlongAxis = PolygonRegularityPlotterAlongAxis(filename)
    myPolygonRegularityPlotterAlongAxis.SetDefaultOrderedJunctionsEdgeColor(defaultOrderedJunctionsEdgeColor)
    myPolygonRegularityPlotterAlongAxis.SetLabelNameConverterDict(labelNameConverterDict)
    lobynessAndRelCompletenessOfPolygons = [{'lobyness': 1.0, 'relativeCompleteness': 0.8811688311688312},
                                            {'lobyness': 1.1297757309528387, 'relativeCompleteness': 0.6626582278481012},
                                            {'lobyness': 1.0, 'relativeCompleteness': 0.9299645390070922},
                                            {'lobyness': 1.8126503921288961, 'relativeCompleteness': 0.19985783674920032},
                                            {'lobyness': 1.2065072398688503, 'relativeCompleteness': 0.5993506493506493}]
    for i, cellProperty in enumerate(myPolygonRegularityPlotterAlongAxis.cellPropertyOverview.values()):
        myPolygonRegularityPlotterAlongAxis.addMeasuresToPolygon(cellProperty, lobynessAndRelCompletenessOfPolygons[i])
    for xMeasureName, yMeasureName in allRegularityMeasures:
        if showPlot:
            filenameToSave = None
        else:
            filenameToSave = f"{resultsFolder}regularityResults/Scales/ArtificialPolygons {yMeasureName} vs {xMeasureName}.png"
        myPolygonRegularityPlotterAlongAxis.PlotCellOutlinesAlongTwoAxis(xMeasureName, yMeasureName, fontSize=18, filenameToSave=filenameToSave)

if __name__ == '__main__':
    visualizeArtificialPolygonsAlongAxisMain()
    # visualizeArtificialPolygonsAlongTwoAxisMain()
