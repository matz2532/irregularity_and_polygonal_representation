import numpy as np
import sys
import warnings

sys.path.insert(0, "./Code/DataStructures/")
sys.path.insert(0, "./Code/MeasureCreator/")

from AdjacencyGraphHelper import extractAdjacencyGraph, extractOrderedPeripheralNodes
from GraphCreatorFromDelaunayTriangulation import pointsAdjacencyGraphFromDelaunayTriangulation, faceAdjacencyGraphFromDelaunayTriangulation
from FolderContent import FolderContent
from MultiFolderContent import MultiFolderContent
from scipy.spatial import Delaunay

verbosity = 1
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
    previousVerbosity = tissue.verbose
    tissue.verbose = 0 # does not give a message, when resolution is not set, but rather just returns none
    resolution: int or None = tissue.GetResolution()

    # use adjacencyListFilenameKey="labelledImageAdjacencyList" for Eng data and
    # neighborDistancesFilenameKey="neighborDistance" for MGX data derived tissues (i.e. Matz and Smit data)
    # (also implement and test extraction of adjacency graph)
    fullAdjacencyGraph = extractAdjacencyGraph(tissue, adjacencyListFilenameKey="labelledImageAdjacencyList")
    orderedPerimeterCells = extractOrderedPeripheralNodes(fullAdjacencyGraph)
    junctionPositions = tissue.LoadKeyUsingFilenameDict("orderedJunctionsPerCellFilename")
    tissue.verbose = previousVerbosity

    tissueProperties["numberOfCells"] = getNumberOfCells(tissue)
    tissueProperties["numberOfCellsAtPerimeter"] = len(orderedPerimeterCells)
    orderedPerimeterPositions = extractOrderedPerimeterPoints(orderedPerimeterCells, junctionPositions)
    tissueProperties["perimeterPoints"] = len(orderedPerimeterPositions)
    tissueProperties["perimeterInMicrons"] = getPerimeterDistance(orderedPerimeterPositions, resolution)
    tissueProperties["numberOfJunctions"] = getNumberOfJunctions(tissue)
    tissueProperties["tissueAreaInMicrons^2"] = getTissueArea(tissue)
    return tissueProperties

def findSharedPoints(pointsFrom1, pointsFrom2, tolerance: float = np.sqrt(5)):
    sharedPoints, indicesOfSharedPoints = [], []
    for i, p1 in enumerate(pointsFrom1):
        # print(pointsFrom2 - p1)
        distanceToP1 = np.linalg.norm(pointsFrom2 - p1, axis=1)
        # print(np.min(distanceToP1))
        isShared = distanceToP1 <= tolerance
        if np.any(isShared):
            sharedPoints.append(p1)
            indicesOfSharedPoints.append((i, np.where(isShared)[0][0]))
    return sharedPoints, indicesOfSharedPoints

def findSharedEdges(orderedPerimeterCells, junctionPositionsOfCells, returnIndicesToo: bool = True):
    sharedJunctionsOfEdges, indicesSharedJunctionsOfEdges = {}, {}
    for i, currentPerimeterCell in enumerate(orderedPerimeterCells):
        previousPerimeterCell = orderedPerimeterCells[i - 1]
        if previousPerimeterCell in junctionPositionsOfCells and currentPerimeterCell in junctionPositionsOfCells:
            sharedJunctions, indicesOfSharedJunctions = findSharedPoints(junctionPositionsOfCells[previousPerimeterCell], junctionPositionsOfCells[currentPerimeterCell])
        else:
            if verbosity > 0:
                print(f"Extracting ordered perimeter points between {previousPerimeterCell} and {currentPerimeterCell} resulted in no shared points even though they are neighbors. {(previousPerimeterCell in junctionPositionsOfCells)=} {(currentPerimeterCell in junctionPositionsOfCells)=}")
            sharedJunctions, indicesOfSharedJunctions = None, None
        edge = (previousPerimeterCell, currentPerimeterCell)
        sharedJunctionsOfEdges[edge] = sharedJunctions
        indicesSharedJunctionsOfEdges[edge] = indicesOfSharedJunctions
    if returnIndicesToo:
        return sharedJunctionsOfEdges, indicesSharedJunctionsOfEdges
    return sharedJunctionsOfEdges
def extractOrderedPerimeterPoints(orderedPerimeterCells, junctionPositionsOfCells):
    orderedPerimeterPositions = [] # min number is len(orderedPerimeterCells), but can be higher as well
    # check two shared junctions of adjacent cells, determine, which appears only twice
    # if there are junctions only appearing once add them continuously until reaching one with more than one appearance
    sharedJunctionsOfEdges, indicesSharedJunctionsOfEdges = findSharedEdges(orderedPerimeterCells, junctionPositionsOfCells)
    return orderedPerimeterPositions

def getPerimeterDistance(orderedPerimeterPositions, resolution):
    nextPerimeterPosition = np.concatenate([orderedPerimeterPositions[1:], [orderedPerimeterPositions[0]]])
    distanceBetweenPoints = np.linalg.norm(orderedPerimeterPositions-nextPerimeterPosition, axis=1)
    if resolution == 1:
        return distanceBetweenPoints.sum()
    else:
        return distanceBetweenPoints.sum() * resolution

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

def getNumberOfJunctions(tissue: FolderContent, keyForJunctionPositions: str = "finalJunctionFilename"):
    junctionPositions = tissue.LoadKeyUsingFilenameDict(keyForJunctionPositions)
    return len(junctionPositions)

def getTissueArea(tissue: FolderContent, keyForFileWithAreaDict: str = "areaMeasuresPerCell", nestedKeyName: str or None = "labelledImageArea", resolutionFactor: float or None = None):
    areaPerCellDict = tissue.LoadKeyUsingFilenameDict(keyForFileWithAreaDict)
    if nestedKeyName is not None:
        assert nestedKeyName in areaPerCellDict, f"Invalid {nestedKeyName=} for the tissue {tissue.GetTissueName()}, it's data from the key={keyForFileWithAreaDict}"
        areaPerCellDict = areaPerCellDict[nestedKeyName]
    assert type(areaPerCellDict) == dict, f"For the tissue {tissue.GetTissueName()}, it's data from the key={keyForFileWithAreaDict} was no dictionary (with cells representing the key), {type(areaPerCellDict)} != dict"
    numberOfCells = len(areaPerCellDict)
    assert numberOfCells != 0, f"The tissue {tissue.GetTissueName()} seemed to contain no cells please check the corresponding cell dictionary from the key={keyForFileWithAreaDict}"
    if numberOfCells < 10:
        warnings.warn(f"Please double check the tissue {tissue.GetTissueName()} as it seem to contain less than 10 cell ({numberOfCells=}) from the key={keyForFileWithAreaDict} in file={tissue.GetFilenameDictKeyValue(keyForFileWithAreaDict)}")
    tissueArea = 0
    for cellArea in areaPerCellDict.values():
        tissueArea += cellArea
        print(cellArea)
    if tissueArea < 10:
        warnings.warn(f"Please double check the tissue {tissue.GetTissueName()} as it seem to be smaller than {tissueArea} in size from the key={keyForFileWithAreaDict} in file={tissue.GetFilenameDictKeyValue(keyForFileWithAreaDict)}")
    if resolutionFactor is not None:
        tissueArea *= resolutionFactor
    return tissueArea

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
