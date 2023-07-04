import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pickle
import sys

sys.path.insert(0, "./Code/ImageToRawDataConversion/")
from PolygonalRegularityCalculator import PolygonalRegularityCalculator

class PerTissuePolygonRegularityCalculator (PolygonalRegularityCalculator):

    verbose=1
    baseResolution=1 # baseline resolution in case non is given in length units [e.g. microM] per pixel
    ignoreSegmentOrAngleExtension="_ignoringSomeSegmentOrAngle"
    labelledImageToContourDifference=0 # is -1, when using GraVis contours
    # default values for regularity dictionaries
    lengthRegularityOfCellsInTissue: dict = None
    lengthVariatonOfCellsInTissue: dict = None
    lengthGiniCoeffOfCellsInTissue: dict = None
    angleRegularityOfCellsInTissue: dict = None
    angleVariatonOfCellsInTissue: dict = None
    angleGiniCoeffOfCellsInTissue: dict = None

    def __init__(self, labelledImgArray, allContours, orderedJunctionsOfCellsInTissue, runOnInit=True,
                 resolution=1, contourOffset=np.asarray([-1, -1]), ignoreSegmentOrAngleExtension=None,
                 ignoreSegmentsDict=None, ignoreAnglesDict=None):
        # The labelledImgArray is structured that zero is the background, one is the tissue skeleton (e.g. all cell outlines),
        # and all values >= 2 represent cells
        # The keys in allContours start from 1, but represent a cell of +1 in the labelledImgArray
        # e.g. cell 1 of allContours is cell 2 in labelledImgArray
        self.resolution = resolution
        self.contourOffset = contourOffset
        if not labelledImgArray is None and not allContours is None and not orderedJunctionsOfCellsInTissue is None:
            self.SetCalculator(labelledImgArray, allContours, orderedJunctionsOfCellsInTissue,
                               resolution=self.resolution, ignoreSegmentOrAngleExtension=ignoreSegmentOrAngleExtension,
                               ignoreSegmentsDict=ignoreSegmentsDict, ignoreAnglesDict=ignoreAnglesDict)
            if runOnInit:
                self.CalcPolygonalComplexityForTissue()

    def CalcPolygonalComplexityForTissue(self):
        self.lengthGiniCoeffOfCellsInTissue = {}
        self.angleGiniCoeffOfCellsInTissue = {}
        for cell in self.cellIds:
            orderedJunctions = self.orderedJunctionsOfCellsInTissue[cell]
            if len(orderedJunctions) < 3:
                continue
            if not orderedJunctions is None:
                isSegmentIgnored = self.ignoreSegmentsDict[cell] if not self.ignoreSegmentsDict is None else None
                isAngleIgnored = self.ignoreAnglesDict[cell] if not self.ignoreAnglesDict is None else None
                lengthGiniCoeff, angleGiniCoeff = self.CalcGiniCoefficients(orderedJunctions, isSegmentIgnored=isSegmentIgnored,
                                                                            isAngleIgnored=isAngleIgnored, currentCell=cell)
                self.orderedJunctionsOfCellsInTissue[cell] = orderedJunctions
                self.lengthGiniCoeffOfCellsInTissue[cell] = lengthGiniCoeff
                self.angleGiniCoeffOfCellsInTissue[cell] = angleGiniCoeff

    def CalcPolygonalComplexityForTissueFull(self):
        self.lengthRegularityOfCellsInTissue = {}
        self.lengthVariatonOfCellsInTissue = {}
        self.lengthGiniCoeffOfCellsInTissue = {}
        self.angleRegularityOfCellsInTissue = {}
        self.angleVariatonOfCellsInTissue = {}
        self.angleGiniCoeffOfCellsInTissue = {}
        for cell in self.cellIds:
            orderedJunctions = self.orderedJunctionsOfCellsInTissue[cell]
            if len(orderedJunctions) < 3:
                continue
            if not orderedJunctions is None:
                isSegmentIgnored = self.ignoreSegmentsDict[cell] if not self.ignoreSegmentsDict is None else None
                isAngleIgnored = self.ignoreAnglesDict[cell] if not self.ignoreAnglesDict is None else None
                lengthRegularity, lengthVariaton, lengthGiniCoeff = self.CalcAllSideLengthRegularitiesOfPolygonWith(orderedJunctions, isSegmentIgnored)
                angleRegularity, angleVariaton, angleGiniCoeff = self.CalcAllInternalAngleRegularitiesOfPolygonWith(orderedJunctions, isAngleIgnored, currentCell=cell)
                self.orderedJunctionsOfCellsInTissue[cell] = orderedJunctions
                self.lengthRegularityOfCellsInTissue[cell] = lengthRegularity
                self.lengthVariatonOfCellsInTissue[cell] = lengthVariaton
                self.lengthGiniCoeffOfCellsInTissue[cell] = lengthGiniCoeff
                self.angleRegularityOfCellsInTissue[cell] = angleRegularity
                self.angleVariatonOfCellsInTissue[cell] = angleVariaton
                self.angleGiniCoeffOfCellsInTissue[cell] = angleGiniCoeff

    def VisualiseCellsComplexity(self, cell, inLabelledImg=False):
        if cell in self.angleGiniCoeffOfCellsInTissue:
            complexity = self.angleGiniCoeffOfCellsInTissue[cell]
            orderedJuntions = self.orderedJunctionsOfCellsInTissue[cell]
            contour = self.allContours[cell]
            if inLabelledImg:
                print("Not yet implemented inLabelledImg=True")
            else:
                meanEdgeLength = np.mean(self.calcPolygonSideLengths(orderedJuntions))
                title = "Cell {} with polygon complexity {}\nmean edge length {}".format(cell, np.round(complexity, 2), np.round(meanEdgeLength))
                img  = self.createZeroedImgOf(contour, orderedJuntions)
                plt.imshow(img)
                plt.axis('off')
                plt.title(title)
                plt.show()
        else:
            if self.verbose >= 1:
                print("The cell's {} polygon complexity was was not visualised as it does not have a complexity value.".format(cell))

    def createZeroedImgOf(self, contour, orderedJuntions):
        min = np.min(contour, axis=0)
        contour -= min
        orderedJuntions -= min
        imgShape = np.max(contour, axis=0)+1
        img = np.zeros(imgShape)
        for p in contour:
            img[p[0], p[1]] = 125
        for p in orderedJuntions:
            img[p[0], p[1]] = 255
        return img

    def ExtractMeanFromDict(self, dictionary, resolution=1):
        return np.mean(list(dictionary.values())) * resolution

    def ExtractstdFromDict(self, dictionary, resolution=1):
        return np.std(list(dictionary.values())) * resolution

    def GetAngleRegularityOfCellsInTissue(self):
        return self.angleRegularityOfCellsInTissue

    def GetAngleVariatonOfCellsInTissue(self):
        return self.angleVariatonOfCellsInTissue

    def GetAngleGiniCoeffOfCellsInTissue(self):
        return self.angleGiniCoeffOfCellsInTissue

    def GetLengthRegularityOfCellsInTissue(self):
        return self.lengthRegularityOfCellsInTissue

    def GetLengthVariatonOfCellsInTissue(self):
        return self.lengthVariatonOfCellsInTissue

    def GetLengthGiniCoeffOfCellsInTissue(self):
        return self.lengthGiniCoeffOfCellsInTissue

    def GetRegularityMeasuresDict(self):
        angleRegularityKey = "angleRegularity"
        if not self.ignoreAnglesDict is None:
            angleRegularityKey += self.ignoreSegmentOrAngleExtension
        angleVariatonKey = "angleVariaton"
        if not self.ignoreAnglesDict is None:
            angleVariatonKey += self.ignoreSegmentOrAngleExtension
        angleGiniCoeffKey = "angleGiniCoeff"
        if not self.ignoreAnglesDict is None:
            angleGiniCoeffKey += self.ignoreSegmentOrAngleExtension
        lengthRegularityKey = "lengthRegularity"
        if not self.ignoreSegmentsDict is None:
            lengthRegularityKey += self.ignoreSegmentOrAngleExtension
        lengthVariatonKey = "lengthVariaton"
        if not self.ignoreSegmentsDict is None:
            lengthVariatonKey += self.ignoreSegmentOrAngleExtension
        lengthGiniCoeffKey = "lengthGiniCoeff"
        if not self.ignoreSegmentsDict is None:
            lengthGiniCoeffKey += self.ignoreSegmentOrAngleExtension
        regularityMeasureDicts = {}
        angleRegularity = self.GetAngleRegularityOfCellsInTissue()
        if angleRegularity is not None:
            regularityMeasureDicts[angleRegularityKey] = angleRegularity
        angleVariaton = self.GetAngleVariatonOfCellsInTissue()
        if angleVariaton is not None:
            regularityMeasureDicts[angleVariatonKey] = angleVariaton
        angleGiniCoeff = self.GetAngleGiniCoeffOfCellsInTissue()
        if angleGiniCoeff is not None:
            regularityMeasureDicts[angleGiniCoeffKey] = angleGiniCoeff
        lengthRegularity = self.GetLengthRegularityOfCellsInTissue()
        if lengthRegularity is not None:
            regularityMeasureDicts[lengthRegularityKey] = lengthRegularity
        lengthVariaton = self.GetLengthVariatonOfCellsInTissue()
        if lengthVariaton is not None:
            regularityMeasureDicts[lengthVariatonKey] = lengthVariaton
        lengthGiniCoeff = self.GetLengthGiniCoeffOfCellsInTissue()
        if lengthGiniCoeff is not None:
            regularityMeasureDicts[lengthGiniCoeffKey] = lengthGiniCoeff
        return regularityMeasureDicts

    def SetCalculator(self, labelledImgArray, allContours, orderedJunctionsOfCellsInTissue, resolution=None, ignoreSegmentOrAngleExtension=None, ignoreSegmentsDict=None, ignoreAnglesDict=None):
        self.SetCalculatorWithoutChecking(allContours, orderedJunctionsOfCellsInTissue, resolution=resolution,
                                         ignoreSegmentOrAngleExtension=ignoreSegmentOrAngleExtension,
                                         ignoreSegmentsDict=ignoreSegmentsDict, ignoreAnglesDict=ignoreAnglesDict)
        labelledImageCellIds = np.unique(labelledImgArray) - self.labelledImageToContourDifference
        isCellFromLabelledImgPresentInContour = np.isin(self.cellIds, labelledImageCellIds)
        assert np.all(isCellFromLabelledImgPresentInContour), "The contour id/s {} are not present in the labelled, potential cells are {}.".format(self.cellIds[np.invert(isCellFromLabelledImgPresentInContour)], labelledImageCellIds)

    def SetCalculatorWithoutChecking(self, allContours, orderedJunctionsOfCellsInTissue, resolution=None,
                                     ignoreSegmentOrAngleExtension=None, ignoreSegmentsDict=None, ignoreAnglesDict=None):
        self.allContours = allContours
        self.orderedJunctionsOfCellsInTissue = orderedJunctionsOfCellsInTissue
        self.cellIds = np.asarray(list(self.allContours.keys()))
        if not resolution is None:
            self.resolution = resolution
        else:
            self.resolution = self.baseResolution
        if not ignoreSegmentOrAngleExtension is None:
            self.ignoreSegmentOrAngleExtension = ignoreSegmentOrAngleExtension
        self.ignoreSegmentsDict = ignoreSegmentsDict
        self.ignoreAnglesDict = ignoreAnglesDict

def extractPolygonalComplexityOf(folder):
    contourFilenames = folder + "cellContours.pkl"
    labelledImageFilename = folder + "labelledImage.npy"
    labelledImgArray = np.load(labelledImageFilename)
    with open(contourFilenames, "rb") as fh:
        allContours = pickle.load(fh)
    triWayJunctionsFilename = folder + "correctTriWayJunctions.pkl"
    with open(triWayJunctionsFilename, "rb") as fh:
        triWayJunctions = pickle.load(fh)
    triWayJunctions = triWayJunctions.astype(int)
    myPolygonalRegularityCalculator = PerTissuePolygonRegularityCalculator(labelledImgArray, allContours, triWayJunctions)
    return myPolygonalRegularityCalculator

def mainTestOneTissue():
    folder = "../Bezier_curve_complexity/Data/Eng 2021 time lines/col-0/20170501 WT S2/0h/"
    myPerTissuePolygonRegularityCalculator = extractPolygonalComplexityOf(folder)
    myPerTissuePolygonRegularityCalculator.VisualiseCellsComplexity(1)

def main():
    baseFolder = "../Bezier_curve_complexity/Data/Eng 2021 time lines/col-0/20170501 WT S2/"
    timeExtensions = ["0h/", "24h/", "48h/", "72h/", "96h/"]
    resolution = 0.221
    meanPolygonalDifferences = []
    stdPolygonalDifferences = []
    for extension in timeExtensions:
        folder = baseFolder + extension
        myPerTissuePolygonRegularityCalculator = extractPolygonalComplexityOf(folder)
        meanPolygonalDifferences.append(np.round(myPerTissuePolygonRegularityCalculator.GetMeanTissueComplexity(resolution), 2))
        stdPolygonalDifferences.append(np.round(myPerTissuePolygonRegularityCalculator.GetStdTissueComplexity(resolution), 2))
    timePoints = [t.replace("/", "") for t in timeExtensions]
    meanPlusMinusStd = ["{}+-{}".format(mean, std) for mean, std in zip(meanPolygonalDifferences, stdPolygonalDifferences)]
    print(dict(zip(timePoints, meanPlusMinusStd)))

if __name__ == '__main__':
    main()
