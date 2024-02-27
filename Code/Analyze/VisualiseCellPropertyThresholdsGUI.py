import matplotlib
import pandas as pd

matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *

from PatchCreator import PatchCreator

class VisualiseCellPropertyThresholdsGUI (PatchCreator):

    # default axes and canvas and slider value
    distributionPlotCanvas=None
    ax1=None
    ax2=None
    sliderValue: float = None
    # table loading parameters
    cellLabelColumnName: str = "Label"
    valueColumnName: str = "Area (µm²)"
    # coloring and naming schemes
    startingPercentileSliderValue: float = 65
    aboveThresholdColor: str = "#00ff00ff"
    bellowThresholdColor: str = "#ff0000ff"
    selectedOrNotColorLabel: list = ["selected", "not selected"]
    valueToCellTypeConverter: dict = {0: "pavement cell", 1: "small cells", 2: "guard cell", 3: "peripheral cells", 4: "cells without contour"}
    cellTypeToValueConverter: dict = {v: k for k, v in valueToCellTypeConverter.items()}


    def __init__(self, cellsContourPositionDict, cellValueDf: pd.DataFrame = None, cellTypes: dict = None,
                 defaultSliderValue=None, useMeanAsDefault=False, yFigSizeInInch=5):
        self.SetData(cellsContourPositionDict, cellValueDf, cellTypes, cellLabelColumnName=self.cellLabelColumnName, valueColumnName=self.valueColumnName)
        if cellTypes is not None:
            self.defaultSliderValue = 0.5
        elif defaultSliderValue is None:
            if useMeanAsDefault:
                self.defaultSliderValue = self.cellValueDf.iloc[:, 1].mean()
            else:
                self.defaultSliderValue = np.percentile(self.cellValueDf.iloc[:, 1], self.startingPercentileSliderValue)
        else:
            self.defaultSliderValue = defaultSliderValue
        self.yFigSizeInInch = yFigSizeInInch
        self.run()

    def SetData(self, cellsContourPositionDict, cellValueDf: pd.DataFrame = None, cellTypes: dict = None, cellLabelColumnName="Label", valueColumnName="Area (µm²)"):
        self.cellsContourPositionDict = cellsContourPositionDict
        self.valueColumnName = valueColumnName
        if cellValueDf is not None:
            self.sliderResolution = 10
            self.cellValueDf = cellValueDf[[cellLabelColumnName, valueColumnName]]
        elif cellTypes is not None:
            self.sliderResolution = 1
            self.cellValueDf = self.convertCellTypesInValueDf(cellTypes, cellLabelColumnName, valueColumnName)
        self.cellsValueDict = dict(zip(self.cellValueDf[cellLabelColumnName], self.cellValueDf[valueColumnName]))
        cellsWithContours = np.array(list(self.cellsContourPositionDict.keys()))
        cellsWithValues = self.cellValueDf[cellLabelColumnName].to_list()
        missingCellsWithValue = np.setdiff1d(cellsWithContours, cellsWithValues)
        assert len(missingCellsWithValue) == 0, f"The cells {missingCellsWithValue} are missing values, only {cellsWithValues} have values."

    def convertCellTypesInValueDf(self, cellTypes: dict, cellLabelColumnName: str, valueColumnName: str, cellTypeNameColumnName: str = "cell type", unassignedDefaultValue: int = -1):
        allCellLabels, cellTypeAsValue, cellTypeNameOfLabel = [], [], []
        for cellTypeName, cellLabelsOfType in cellTypes.items():
            if cellTypeName in self.cellTypeToValueConverter:
                value = self.cellTypeToValueConverter[cellTypeName]
            else:
                value = unassignedDefaultValue
                print(cellTypeName)
            nrOfCellsOfType = len(cellLabelsOfType)
            allCellLabels.extend(list(cellLabelsOfType))
            cellTypeAsValue.extend(list(np.full(nrOfCellsOfType, value)))
            cellTypeNameOfLabel.extend(list(np.full(nrOfCellsOfType, cellTypeName)))
        cellValueDf = pd.DataFrame({cellLabelColumnName: allCellLabels,
                                    valueColumnName: cellTypeAsValue,
                                    cellTypeNameColumnName: cellTypeNameOfLabel})
        return cellValueDf

    def run(self):
        self.root = Tk()
        self.root.title("Visualisation of {} thresholding".format(self.valueColumnName))
        self.initImageRepresentation()
        self.initSlider()
        self.initValueDistribution()
        self.root.mainloop()

    def initImageRepresentation(self, offSet=10):
        xMax, yMax = -np.inf, -np.inf
        for contourPositions in self.cellsContourPositionDict.values():
            currentMax = np.max(contourPositions, axis=0)
            if xMax < currentMax[1]:
                xMax = currentMax[1]
            if yMax < currentMax[0]:
                yMax = currentMax[0]
        xMax += offSet
        yMax += offSet
        self.xLim = (0, xMax)
        self.yLim = (0, yMax)
        xyRatio = xMax / yMax
        fig = Figure(figsize=(self.yFigSizeInInch * xyRatio, self.yFigSizeInInch), dpi=100, constrained_layout=True)
        self.ax1 = fig.add_subplot(111)
        self.drawImageRepresentation()
        self.imageRepresentationCanvas = FigureCanvasTkAgg(fig, self.root)
        self.imageRepresentationCanvas.get_tk_widget().grid(row=0, column=0)

    def drawImageRepresentation(self):
        cellFaceColors = self.determineFaceColorByThrehold()
        self.PlotPatchesFromOutlineDictOn(self.ax1, self.cellsContourPositionDict, edgecolor="black", faceColor=cellFaceColors)#,
        self.ax1.set_xlim(self.xLim)
        self.ax1.set_ylim(self.yLim)
        self.addLegend(cellFaceColors, correspondingLabel=self.selectedOrNotColorLabel)

    def determineFaceColorByThrehold(self):
        if self.sliderValue:
            currentThreshold = self.sliderValue.get()
        else:
            currentThreshold = self.defaultSliderValue
        cellFaceColors = [self.aboveThresholdColor if self.cellsValueDict[cellLabel] > currentThreshold else self.bellowThresholdColor for cellLabel in self.cellsContourPositionDict.keys()]
        return cellFaceColors

    def addLegend(self, allCellFaceColors: list or np.ndarray, correspondingLabel: list or np.ndarray = None):
        uniqueCellFaceColors = np.unique(allCellFaceColors)
        for i, c in enumerate(uniqueCellFaceColors):
            labelOfColor = str(i)
            if correspondingLabel is not None:
                if i < len(correspondingLabel):
                    labelOfColor = correspondingLabel[i]
            self.ax1.scatter(-10, -10, c=c, label=labelOfColor, marker="s")
        self.ax1.legend()

    def initSlider(self):
        self.sliderFrame = Frame(self.root, relief = SUNKEN, borderwidth = 1)
        self.sliderValue = DoubleVar()
        label = Label(self.sliderFrame, text = 'Slider Descriptor')
        min, max = self.cellValueDf.iloc[:, 1].min(), self.cellValueDf.iloc[:, 1].max()
        sli = Scale(self.sliderFrame, from_ = max+1, to = min-1, orient = VERTICAL,
                    variable = self.sliderValue, command = self.sliderChanged, resolution=self.sliderResolution)
        self.sliderValue.set(self.defaultSliderValue)
        label.grid(row=0, column=0)
        sli.grid(row=1, column=0, rowspan=1, sticky="nsew")
        self.sliderFrame.grid(row=0, column=1, rowspan=1, sticky="nsew")

    def sliderChanged(self, event):
        if self.ax2:
            self.ax2.clear()
            self.drawValueDistribution()
            self.distributionPlotCanvas.draw()
        if self.ax1:
            self.ax1.clear()
            self.drawImageRepresentation()
            self.imageRepresentationCanvas.draw()

    def initValueDistribution(self):
        fig = Figure(figsize=(3.5, self.yFigSizeInInch), dpi=100, constrained_layout=True)
        self.ax2 = fig.add_subplot(111)
        self.drawValueDistribution()
        self.distributionPlotCanvas = FigureCanvasTkAgg(fig, self.root)
        self.distributionPlotCanvas.get_tk_widget().grid(row=0, column=2)

    def drawValueDistribution(self):
        self.ax2.boxplot(self.cellValueDf.iloc[:, 1])
        self.ax2.set_ylabel(self.valueColumnName)
        self.ax2.set_title("boxplot of {}".format(self.valueColumnName))
        sliderValue = self.sliderValue.get()
        self.ax2.axhline(y=sliderValue, xmin=0.05, xmax=0.95, linestyle="--", alpha=0.6, color="black")


def main():
    import sys
    sys.path.insert(0, "./Code/DataStructures/")
    from MultiFolderContent import MultiFolderContent
    allFolderContentsFilename = "Images/Smit2023Cotyledons/Smit2023Cotyledons.json"
    excludeViaSize = False
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    # folderContent = list(multiFolderContent)[0]
    for folderContent in multiFolderContent:
        cellsContourPositionDict = folderContent.LoadKeyUsingFilenameDict("cellContours")
        if excludeViaSize:
            cellValueDf = folderContent.LoadKeyUsingFilenameDict("geometricData", skipfooter=4)
            cellTypes = None
        else:
            cellValueDf = None
            cellTypes = folderContent.LoadKeyUsingFilenameDict("cellType")
        myVisualiseCellPropertyThresholdsGUI = VisualiseCellPropertyThresholdsGUI(cellsContourPositionDict, cellValueDf=cellValueDf, cellTypes=cellTypes)

if __name__ == '__main__':
    main()
