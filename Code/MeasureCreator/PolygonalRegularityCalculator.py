import numpy as np
import warnings

from itertools import combinations
from scipy.stats import variation
from shapely import geometry
from shapely.geometry import LineString

class PolygonalRegularityCalculator (object):

    resolution=1 # in length units [e.g. microM] per pixel
    verbosity=1 # set verbosity to 0 to ignore warnings about using external angles instead of internal angles
    currentCell=None

    def SetResolution(self, resolution):
        self.resolution = resolution

    def CalcAllSideLengthRegularitiesOfPolygonWith(self, vertexList, isSegmentIgnored=None):
        # calculates the deviation of the side length of the given polygon
        sideLengths = self.calcPolygonSideLengths(vertexList)
        if not isSegmentIgnored is None:
            sideLengths = sideLengths[np.invert(isSegmentIgnored)]
            if len(sideLengths) == 0:
                if self.verbosity > 0:
                    warnings.warn(f"In the {vertexList=} all segments were ignored and None is returned for all metrics.")
                    return None, None, None
        lengthRegularity = self.calcRegularity(sideLengths)
        lengthVariaton = self.calcCoefficientOfVariation(sideLengths)
        lengthGiniCoeff = self.calcGiniCoefficient(sideLengths)
        return lengthRegularity, lengthVariaton, lengthGiniCoeff

    def CalcAllInternalAngleRegularitiesOfPolygonWith(self, vertexList, isAngleIgnored=None, currentCell=None):
        # calculates the deviation of the internal angle of the given polygon
        # whoese vertices are given in a clockwise order
        # (if given counter clockwise, the outer angles are calculated and an warning is printed)
        self.currentCell=currentCell
        internalPolygonAngles = self.calcAllInternalPolygonAngles(vertexList)
        if not isAngleIgnored is None:
            internalPolygonAngles = internalPolygonAngles[np.invert(isAngleIgnored)]
            if len(internalPolygonAngles) == 0:
                if self.verbosity > 0:
                    warnings.warn(f"In the {vertexList=} all angles were ignored and None is returned for all metrics.")
                    return None, None, None
        angleRegularity = self.calcRegularity(internalPolygonAngles)
        angleVariaton = self.calcCoefficientOfVariation(internalPolygonAngles)
        angleGiniCoeff = self.calcGiniCoefficient(internalPolygonAngles)
        return angleRegularity, angleVariaton, angleGiniCoeff

    def CalcRegularityDict(self, vertexList, isSegmentIgnored=None):
        lengthRegularity, lengthVariaton, lengthGiniCoeff = self.CalcAllSideLengthRegularitiesOfPolygonWith(vertexList, isSegmentIgnored)
        angleRegularity, angleVariaton, angleGiniCoeff = self.CalcAllInternalAngleRegularitiesOfPolygonWith(vertexList, isSegmentIgnored)
        metricsDict = {"lengthRegularity":lengthRegularity, "lengthVariaton":lengthVariaton, "lengthGiniCoeff":lengthGiniCoeff,
                       "angleRegularity":angleRegularity, "angleVariaton":angleVariaton, "angleGiniCoeff":angleGiniCoeff}
        return metricsDict

    def calcPolygonSideLengths(self, vertexList):
        if isinstance(vertexList, list):
            vertexList = np.asarray(vertexList)
        nrOfVertices = len(vertexList)
        sideLengths = np.zeros(nrOfVertices)
        for i in range(nrOfVertices):
            startP = vertexList[i-1]
            endP = vertexList[i]
            sideLengths[i] = np.linalg.norm(startP - endP)
        if self.resolution != 1:
            sideLengths *= self.resolution
        return sideLengths

    def calcRegularity(self, x):
        unnormalisedRegularity = (np.sum(x / np.max(x)) - 1)
        return unnormalisedRegularity / (len(x) - 1)

    def calcCoefficientOfVariation(self, x):
        coeffOfVariation = variation(x)
        return coeffOfVariation

    def calcGiniCoefficient(self, x):
        # thanks to Martin Thoma, from stackoverflow
        n = len(x)
        diff = np.sum(np.abs(i - j) for i, j in combinations(x, r=2))
        return diff / (2 * n**2 * np.mean(x))

    def calcAllInternalPolygonAngles(self, vertexList, checkOrientationOfVertices=True, errorTolerance=0.0001):
        vertexList = np.asarray(vertexList)
        nrOfVertices = len(vertexList)
        allInternalAngles = np.zeros(nrOfVertices)
        for i in range(nrOfVertices):
            angle = self.angleBetweenPoints(vertexList[i-2], vertexList[i-1], vertexList[i])
            allInternalAngles[i] = angle
        if checkOrientationOfVertices:
            expectedInternalAngleSum = (nrOfVertices-2) * 180
            if np.sum(allInternalAngles) > expectedInternalAngleSum + errorTolerance:
                if self.verbosity > 0:
                    print(f"""Warning your polygons internal angles sum is of {self.currentCell=} higher than expected {np.sum(allInternalAngles)} > {expectedInternalAngleSum + errorTolerance}
                        for the polygon {geometry.Polygon(vertexList)},
                        you are probably given the external angles.
                        In case you want the internal angle reverse the order of vertices.""")
        return allInternalAngles

    def angleBetweenPoints(self, startPoint, midPoint, endPoint, inDeg=True):
        # calculate the angle clockwise (switch v1 and v2 to calculate counter-clockwise)
        v2 = startPoint - midPoint
        v1 = endPoint - midPoint
        if len(v1) == 2:
            angle = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
        elif len(v1) == 3:
            cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(cosine_angle)
        else:
            assert len(v1) == 2 or len(v1) == 3, f"The calculation between the 3 points was aborted as the points are need to have either 2 or 3 coordinates. {startPoint=}, {midPoint=}, {endPoint=}"
        if angle < 0:
            angle += np.pi + np.pi
        if inDeg:
            angle = np.rad2deg(angle)
        return angle

def extractSideLengthsAndInternalAngleOf(folderContensFilename):
    import sys
    sys.path.insert(0, "./Code/DataStructures/")
    from MultiFolderContent import MultiFolderContent
    multiFolderContent = MultiFolderContent(folderContensFilename)
    regularityCalculator = PolygonalRegularityCalculator()
    allSideLengths, allInternalAngles = [], []
    for folderContent in multiFolderContent:
        orderedJunctionsPerCell = folderContent.LoadKeyUsingFilenameDict("orderedJunctionsPerCellFilename")
        for cellId, orderedJunctions in orderedJunctionsPerCell.items():
            sideLengths = regularityCalculator.calcPolygonSideLengths(orderedJunctions)
            internalAngles = regularityCalculator.calcAllInternalPolygonAngles(orderedJunctions)
            allSideLengths.append(sideLengths)
            allInternalAngles.append(internalAngles)
    allSideLengths = np.concatenate(allSideLengths)
    allInternalAngles = np.concatenate(allInternalAngles)
    return allSideLengths, allInternalAngles

def mainVisualiseLengthAndAngleDistribution():
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams["font.size"] = 25
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    allFolderContensFilename = ["Images/Eng2021Cotyledons.pkl", "Images/full cotyledons/full cotyledons.pkl", "Images/Matz2022SAM.pkl"]
    allSideLengths, allInternalAngles = [], []
    for folderContensFilename in allFolderContensFilename:
        sideLengths, internalAngles = extractSideLengthsAndInternalAngleOf(folderContensFilename)
        allSideLengths.append(sideLengths)
        allInternalAngles.append(internalAngles)
    allSideLengths = np.concatenate(allSideLengths)
    allInternalAngles = np.concatenate(allInternalAngles)
    fig, ax = plt.subplots(1, 3)
    sns.violinplot(y=allSideLengths, ax=ax[0])
    sns.violinplot(y=np.log2(allSideLengths), ax=ax[1])
    sns.violinplot(y=allInternalAngles, ax=ax[2])
    ax[0].set_ylabel("Side length [µm]")
    ax[0].set_title("Side length of all combinations", fontsize=18)
    ax[1].set_ylabel("Log2(side length)")
    ax[1].set_title("Log2 norm. side length of all combinations", fontsize=18)
    ax[2].set_ylabel("Internal angles [°]")
    ax[2].set_title("Internal angles of all combinations", fontsize=18)
    plt.show()

def mainCheck():
    myPolygonalRegularityCalculator = PolygonalRegularityCalculator()
    clockwiseVertexList = [(10,10, 10),(0,10,0), (7,7,0), (10,0,0)]
    # clockwiseVertexList = [(10,10), (0,10), (7,7), (10,0)]
    counterClockwiseVertexList = [(10,10), (10,0), (7,7),(0,10)]
    # clockwiseVertexList = [(10,10),(0,10), (7,7), (8,4), (10,0)]
    # counterClockwiseVertexList = [(10,10), (10,0), (8,4), (7,7),(0,10)]
    clockwiseInternalAngles = myPolygonalRegularityCalculator.calcAllInternalPolygonAngles(clockwiseVertexList)
    clockwiseSideLengths = myPolygonalRegularityCalculator.calcPolygonSideLengths(clockwiseVertexList)
    print("polygon clockwise:", geometry.Polygon(clockwiseVertexList))
    print("area", geometry.Polygon(clockwiseVertexList).area)
    print("angles", clockwiseInternalAngles, "sum", np.sum(clockwiseInternalAngles))
    print("side lengths", clockwiseSideLengths)
    print("counter clock example")
    counterclockwiseInternalAngles = myPolygonalRegularityCalculator.calcAllInternalPolygonAngles(counterClockwiseVertexList)
    print("polygon counter clockwise:", geometry.Polygon(counterClockwiseVertexList))
    print("area", geometry.Polygon(counterClockwiseVertexList).area)
    print("angles", counterclockwiseInternalAngles, "sum", np.sum(counterclockwiseInternalAngles))

if __name__ == '__main__':
    # mainCheck()
    mainVisualiseLengthAndAngleDistribution()
