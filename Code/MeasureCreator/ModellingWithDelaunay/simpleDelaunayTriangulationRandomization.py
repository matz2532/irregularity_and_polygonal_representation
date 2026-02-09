#region Imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

sys.path.insert(0, "./Code/DataStructures/")

from __mainCompareDelaunayTriWithBioPoly__ import plotDelaunayTriangulationWithFaceMidPoints
from __mainModellingWithDelaunay__ import placePointsInsidePerimeter
from matplotlib.axes import Axes
from GraphCreatorFromDelaunayTriangulation import faceAdjacencyGraphFromDelaunayTriangulation
from FolderContent import FolderContent
from MultiFolderContent import MultiFolderContent
from scipy import ndimage
from scipy.spatial import Delaunay
#endregion

#region LocalGlobalParameters
nodeCountKey = "node count"
rngSeedKey = "rng seed"
shapeParameterKey = "shape parameters"
shapeTypeKey = "shape"
shapeAreaKey = "area"
pointPositionKey = "points in shape"
faceIndicesKey = "face indices"
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
    randomizationParameters = {nodeCountKey: numberOfCells,
                               rngSeedKey: seed,
                               shapeParameterKey: 
                                {shapeTypeKey: "circle", shapeAreaKey: totalAreaOfCells},
                               pointPositionKey: randomPointsInShape,
                               faceIndicesKey: tri.simplices.copy()
                              }
    return delaunayFaceGraph, randomizationParameters

def randomlyPlacedPointsInCircle(numberOfCells, totalAreaOfCells, seed):
    circlePerimeterPoints = calculateCirclePointsFromArea(totalAreaOfCells)
    rng = np.random.default_rng(seed)
    # improve here: spacing of points should have min distance based on smallest distance between cell centers
    randomPointsInShape = placePointsInsidePerimeter(numberOfCells, circlePerimeterPoints, rng=rng)
    return randomPointsInShape

def calculateCirclePointsFromArea(totalAreaOfCells, numberOfPoints=360):
    radiusOfAreaInCircle = np.sqrt(totalAreaOfCells / np.pi)
    circlePerimeterPoints = pointsAlongCircle(radiusOfAreaInCircle, numberOfPoints)
    return circlePerimeterPoints

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

#region VisualizeRandomizationProcedure
def plotStepsOfRandomizationFor(delaunayFaceGraph, randomizationParameters, tissueContent, ax: Axes=None):
    plotRandomPointsInCircle(randomizationParameters, ax)
    plotTriangulationFromPoints(delaunayFaceGraph, randomizationParameters, ax)
    plotTriWayJunctionEstimation(delaunayFaceGraph, randomizationParameters, ax)
    plotRandomizedTissue(delaunayFaceGraph, randomizationParameters, ax)

def plotRandomPointsInCircle(randomizationParameters, ax: Axes=None):
    missingExternalAxes = ax is None
    if missingExternalAxes:
        fig, ax = plt.subplots(figsize=(8,8), constrained_layout=True)
    perimeterOfRanomizedTissue = calculateCirclePointsFromArea(randomizationParameters[shapeParameterKey][shapeAreaKey])
    ax.plot(
        np.concatenate([perimeterOfRanomizedTissue[:, 0], [perimeterOfRanomizedTissue[0, 0]]]), 
        np.concatenate([perimeterOfRanomizedTissue[:, 1], [perimeterOfRanomizedTissue[0, 1]]]))
    randomPoints = randomizationParameters[pointPositionKey]
    ax.scatter(randomPoints[:, 0], randomPoints[:, 1], c="C2")
    plt.axis("off")
    if missingExternalAxes:
        plt.show()

def plotTriangulationFromPoints(delaunayFaceGraph, randomizationParameters, ax: Axes=None):
    missingExternalAxes = ax is None
    if missingExternalAxes:
        fig, ax = plt.subplots(figsize=(8,8), constrained_layout=True)
    randomPoints = randomizationParameters[pointPositionKey] 
    tri = Delaunay(randomPoints)
    ax.triplot(randomPoints[:, 0], randomPoints[:, 1], tri.simplices.copy(), zorder=0)
    ax.scatter(randomPoints[:, 0], randomPoints[:, 1], c="C2", zorder=1)
    nx.draw_networkx_nodes(delaunayFaceGraph, pos=nx.get_node_attributes(delaunayFaceGraph, "pos"), label="tri way junction", ax=ax, node_color="black", node_size=20)
    plt.axis("off")
    if missingExternalAxes:
        plt.show()
        
def plotTriWayJunctionEstimation(delaunayFaceGraph, randomizationParameters, ax: Axes=None):
    missingExternalAxes = ax is None
    if missingExternalAxes:
        fig, ax = plt.subplots(figsize=(8,8), constrained_layout=True)
    randomPoints = randomizationParameters[pointPositionKey]
    tri = Delaunay(randomPoints)
    plotDelaunayTriangulationWithFaceMidPoints(delaunayFaceGraph, randomPoints, tri, ax=ax)
    if missingExternalAxes:
        plt.show()

def plotRandomizedTissue(delaunayFaceGraph, randomizationParameters, ax: Axes=None):
    missingExternalAxes = ax is None
    if missingExternalAxes:
        fig, ax = plt.subplots(figsize=(8,8), constrained_layout=True)
    nx.draw_networkx_edges(delaunayFaceGraph, pos=nx.get_node_attributes(delaunayFaceGraph, "pos"), label="triangulated edges", ax=ax)
    plt.axis("off")
    if missingExternalAxes:
        plt.show()
#endregion

"""
Next things:
prepare plots for figure/representation
fix pointArrayFromGeoms(shape) in innterCircleOfTriangle.py for second tissue -> name: col-0_20170327 WT S1_24h
calculate features
"""

#region mainCodeExecution
def main():
    dataSetname = "Eng2021Cotyledons"  # "Smit2023Cotyledons" # "Matz2022SAM" # 
    filename = f"Images/{dataSetname}/{dataSetname}.json"
    mfc = MultiFolderContent(filename)
    #tissueContent = list(mfc)[0]
    startRng = 42
    visualizeRandomizationStepsForTissue = ["col-0_20170327 WT S1_0h"]
    visualizeRandomizationStepsForRng = [42]
    for tissueContent in mfc:
        print(tissueContent.GetTissueName())
        delaunayFaceGraph, randomizationParameters = randomizeTissueUsingDelaunayTriangulation(tissueContent, seed=startRng)
        visualizeStepsInBetween = startRng in visualizeRandomizationStepsForRng and tissueContent.GetTissueName() in visualizeRandomizationStepsForTissue
        if visualizeStepsInBetween:
            plotStepsOfRandomizationFor(delaunayFaceGraph, randomizationParameters)
        nx.get_node_attributes(delaunayFaceGraph, "pos")
        propertiesOfRandomizedTissue = parameterizeDelaunayDerivedTissue(delaunayFaceGraph)
        startRng += 1 # change seed for each run to avoid the same node position during randomizattion for each tissue (logging the seed though in parameters)
        break

if __name__ == '__main__':
    main()
#endregion