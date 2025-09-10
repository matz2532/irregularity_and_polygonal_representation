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
    centerOfCells = {}
    orderedJunctionsPerCell = tissue.LoadKeyUsingFilenameDict("orderedJunctionsPerCellFilename")
    areaMeasuresPerCell = tissue.LoadKeyUsingFilenameDict("areaMeasuresPerCell", convertDictKeysToInt=False)["originalPolygonArea"]
    numberOfCells = len(areaMeasuresPerCell)
    totalAreaOfCells = np.sum(list(areaMeasuresPerCell.values()))
    randomPointsInShape = randomlyPlacedPointsInCircle(numberOfCells, totalAreaOfCells, seed)
    allCellCenters = randomPointsInShape
    tri = Delaunay(allCellCenters)
    delaunayFaceGraph = faceAdjacencyGraphFromDelaunayTriangulation(tri, allCellCenters)
    if visualizeStepsInBetween:
        plotDelaunayTriangulationWithFaceMidPoints(delaunayFaceGraph, allCellCenters, tri)
    randomizationParameters = {"node count": numberOfCells,
                               "rng seed": seed,
                               "shape parameters": 
                                {"shape": "circle", "area": totalAreaOfCells}
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
    # find perimeter nodes
    # remove outside node
    # extract cells with junctions
    # calculate area
    # calculate Gini coeffs
    pass
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