#region Imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

sys.path.insert(0, "./Code/DataStructures/")

from __mainCompareDelaunayTriWithBioPoly__ import plotDelaunayTriangulationWithFaceMidPoints
from __mainModellingWithDelaunay__ import placePointsInsidePerimeter
from GraphCreatorFromDelaunayTriangulation import faceAdjacencyGraphFromDelaunayTriangulation
from FolderContent import FolderContent
from MultiFolderContent import MultiFolderContent
from scipy import ndimage
from scipy.spatial import Delaunay
#endregion

#region MainCode
def randomizeTissueUsingDelaunayTriangulation(tissue: FolderContent, seed=42, visualizeStepsInBetween=False):
    """
    Get number and area of cells from tissue
    Assume circular area for randomized tissue (based on summed area of cells)
    randomly place same number of points as cells in circular area
    apply Delaunay triangulation getting getting graph representation of faces
    """
    centerOfCells = {}
    orderedJunctionsPerCell = tissue.LoadKeyUsingFilenameDict("orderedJunctionsPerCellFilename")
    areaMeasuresPerCell = tissue.LoadKeyUsingFilenameDict("areaMeasuresPerCell", convertDictKeysToInt=False)["originalPolygonArea"]
    numberOfCells = len(areaMeasuresPerCell)
    totalAreaOfCells = np.sum(list(areaMeasuresPerCell.values()))
    randomPointsInShape = randomlyPlacedPointsInCircle(numberOfCells, totalAreaOfCells, seed)
    tri = Delaunay(randomPointsInShape)
    delaunayFaceGraph = faceAdjacencyGraphFromDelaunayTriangulation(tri, randomPointsInShape)
    if visualizeStepsInBetween:
        plotDelaunayTriangulationWithFaceMidPoints(delaunayFaceGraph, randomPointsInShape, tri)
    randomizationParameters = {"node count": numberOfCells,
                               "rng seed": seed,
                               "shape parameters": 
                                {"shape": "circle", "area": totalAreaOfCells},
                                "points in shape": randomPointsInShape,
                                "face indices": tri.simplices.copy()
                              }
    return delaunayFaceGraph, randomizationParameters

def randomlyPlacedPointsInCircle(numberOfCells, totalAreaOfCells, seed):
    radiusOfAreaInCircle = np.sqrt(totalAreaOfCells / np.pi)
    circlePerimeterPoints = pointsAlongCircle(radiusOfAreaInCircle, 360)
    rng = np.random.default_rng(seed)
    # improve here: spacing of points should have min distance based on smallest distance between cell centers
    randomPointsInShape = placePointsInsidePerimeter(numberOfCells, circlePerimeterPoints, rng=rng)
    return randomPointsInShape

def pointsAlongCircle(radius, numberOfPoints):
    anglesOfPoints = np.linspace(0, 2*np.pi, numberOfPoints, endpoint=False)
    x = radius * np.sin(anglesOfPoints)
    y = radius * np.cos(anglesOfPoints)
    return np.concatenate([x,y]).reshape(2, numberOfPoints).T

def parameterizeDelaunayDerivedTissue(delaunayFaceGraph):
    orderedJunctionsOfFaceGraph = extractOrderedJunctionsOf(delaunayFaceGraph)
    # calculate area
    # calculate Gini coeffs
    pass

def extractOrderedJunctionsOf(delaunayFaceGraph):
    junctionsOfCells = assignJunctionsToCell(nx.get_node_attributes(delaunayFaceGraph, "adjacent cells"))
    # order junctions <- use exisiting functionality
    # exclude cells, whos junctions are not forming a circle

def assignJunctionsToCell(cellsOfJunctions):
    junctionsOfCells = {}
    for junctionId, cellIds in cellsOfJunctions.items():
        for i in cellIds:
            if i in junctionsOfCells:
                junctionsOfCells[i].append(junctionId)
            else:
                junctionsOfCells[i] = [junctionId]
    return junctionsOfCells
#endregion

#region mainCodeExecution
def main():
    dataSetname = "Eng2021Cotyledons"  # "Smit2023Cotyledons" # "Matz2022SAM" # 
    filename = f"Images/{dataSetname}/{dataSetname}.json"
    mfc = MultiFolderContent(filename)
    #tissueContent = list(mfc)[0]
    startRng = 42
    for tissueContent in mfc:
        print(tissueContent.GetTissueName())
        delaunayFaceGraph, randomizationParameters = randomizeTissueUsingDelaunayTriangulation(tissueContent, seed=startRng)
        nx.get_node_attributes(delaunayFaceGraph, "pos")
        propertiesOfRandomizedTissue = parameterizeDelaunayDerivedTissue(delaunayFaceGraph)
        startRng += 1 # change seed for each run to avoid the same node position during randomizattion for each tissue (logging the seed though in parameters)

if __name__ == '__main__':
    main()
#endregion