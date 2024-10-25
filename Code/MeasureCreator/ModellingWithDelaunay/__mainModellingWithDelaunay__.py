import numpy as np
import sys
import warnings

sys.path.insert(0, "./Code/DataStructures/")

from GraphCreatorFromDelaunayTriangulation import pointsAdjacencyGraphFromDelaunayTriangulation, faceAdjacencyGraphFromDelaunayTriangulation
from FolderContent import FolderContent
from MultiFolderContent import MultiFolderContent
from scipy.spatial import Delaunay

def createAndAnalyseDelaunayTriangulatedTissueFrom(tissue: FolderContent, repetitions: int = 1, startingSeed: int = 42):
    tissueProperties = extractTissueProperties(tissue)
    allTriangulatedTissues = []
    for i in range(repetitions):
        seed = startingSeed + i
        triangulatedTissue = delaunayTriangulatedTissue(tissueProperties, tissue, seed)
        allTriangulatedTissues.append(triangulatedTissue)
    allTriangulatedTissues = MultiFolderContent(allTriangulatedTissues)
    visualizeTissueProperties(allTriangulatedTissues)

def delaunayTriangulatedTissue(tissueProperties, tissue, seed: int or None = None):
    if seed is not None:
        np.random.set_state(seed)
    perimeterPoints = determineRandomisedPerimeter(tissueProperties["numberOfCellsAtPerimeter"], tissueProperties["perimeterInMicrons"])
    newCellCenters = placePointsInsidePerimeter(tissueProperties["numberOfCells"], perimeterPoints)
    triangulation = applyDelaunayTriangulationTo(newCellCenters, perimeterPoints)
    triangulation = removeExcessPointsOrEdges(triangulation, None)
    triangulatedTissue: FolderContent = parameterizeDelaunayDerivedTissue(triangulation, tissueProperties, tissue)
    return triangulatedTissue

def extractTissueProperties(tissue: FolderContent):
    tissueProperties = {}
    tissueProperties["numberOfCells"] = getNumberOfCells(tissue)
    tissueProperties["numberOfCellsAtPerimeter"] = None
    tissueProperties["numberOfJunctions"] = None
    tissueProperties["perimeterPoints"] = None
    tissueProperties["perimeterInMicrons"] = None
    tissueProperties["tissueAreaInMicrons^2"] = None

    return tissueProperties

def getNumberOfCells(tissue: FolderContent, keyForFileWithCellDict: str = "areaMeasuresPerCell", nestedKeyName: str or None = "labelledImageArea"):
    cellDict = tissue.LoadKeyUsingFilenameDict(keyForFileWithCellDict)
    if nestedKeyName is not None:
        assert nestedKeyName in cellDict, f"Invalid {nestedKeyName=} for the tissue {tissue.GetTissueName()}, it's data from the key={keyForFileWithCellDict}"
        cellDict = cellDict[nestedKeyName]
    assert type(cellDict) == dict, f"For the tissue {tissue.GetTissueName()}, it's data from the key={keyForFileWithCellDict} was no dictionary (with cells representing the key), {type(cellDict)} != dict"
    numberOfCells = len(cellDict)
    assert numberOfCells != 0, f"The tissue {tissue.GetTissueName()} seemed to contain no cells please check the corresponding cell dictionary from the key={keyForFileWithCellDict}"
    if numberOfCells < 10:
        warnings.warn(f"Please double check the tissue {tissue.GetTissueName()} as it seem to contain less than 10 cell ({numberOfCells=}) from the key={keyForFileWithCellDict} in file={tissue.GetFilenameDictKeyValue(keyForFileWithCellDict)}")
    return numberOfCells

def determineRandomisedPerimeter(numberOfPerimeterPoints: int, perimeterLength: float):
    # have different modes, but for now just do a circle
    randomisedPerimeterPoints = np.zeros((numberOfPerimeterPoints, 2))
    return randomisedPerimeterPoints

def placePointsInsidePerimeter(numberOfPoints: int, perimeterPoints: np.array):
    # could have different modes, but for now randomly select points and
    # kick them out when they are not inside the perimeter until target number of points is reached
    # do I need a buffer zone to avoid points right at perimeter (probably, should be something like half of mean cell perimeter)
    points = np.zeros((numberOfPoints, 2))
    return points

def applyDelaunayTriangulationTo(points, perimeterPoints):
    pointsWithPerimeter = None
    # unsure of how to incorporate perimeter points, probably just pool points and
    # double check that the no non-perimeter point is at perimeter
    tri = Delaunay(pointsWithPerimeter)
    return tri

def removeExcessPointsOrEdges(
        tri,
        maxNumberOfEdges: int,
        scalingFactorBetweenEdgesAndPoints: float = 0
        # determines how many cells (original points used in Delaunay Triangulation) to merge
        # to reduce the number of edges,
        # zero means ignore number of points | one means don't remove any points | values in between are some kind of ratio
):
    # optional for now
    if scalingFactorBetweenEdgesAndPoints == 1 or maxNumberOfEdges is None:
        return tri
    # remove edges, "merging" adjacent points, needs an edge/points selection method (maybe very small once?)
    return tri

def parameterizeDelaunayDerivedTissue(triangulation, tissueProperties: dict, originalTissue: FolderContent):
    triangulatedTissue = {}
    #calculate properties used in the visualization
    cellularConnectivityNetwork = pointsAdjacencyGraphFromDelaunayTriangulation(triangulation)
    junctionConnectivityNetwork = faceAdjacencyGraphFromDelaunayTriangulation(triangulation)
    return FolderContent(triangulatedTissue)

def visualizeTissueProperties(tissue: FolderContent or MultiFolderContent):
    # most open point: visualize artificial tissue, show irregularities, compare basic tissue properties especially network derived once
    pass

def main():
    dataSetname = "Eng2021Cotyledons"  # "Smit2023Cotyledons" #
    filename = f"Images/{dataSetname}/{dataSetname}.json"
    mfc = MultiFolderContent(filename)
    tissueContent = list(mfc)[0]
    createAndAnalyseDelaunayTriangulatedTissueFrom(tissueContent)

if __name__ == '__main__':
    main()
