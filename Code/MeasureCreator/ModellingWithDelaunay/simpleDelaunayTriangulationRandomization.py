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
def randomizeTissueUsingDelaunayTriangulation(tissue: FolderContent, seed=42, visualizeStepsInBetween=True):
    centerOfCells = {}
    orderedJunctionsPerCell = tissue.LoadKeyUsingFilenameDict("orderedJunctionsPerCellFilename")
    areaMeasuresPerCell = tissue.LoadKeyUsingFilenameDict("areaMeasuresPerCell", convertDictKeysToInt=False)["originalPolygonArea"]
    numberOfCells = len(areaMeasuresPerCell)
    totalAreaOfCells = np.sum(list(areaMeasuresPerCell.values()))
    radiusOfAreaInCircle = np.sqrt(np.pi / totalAreaOfCells)
    circlePerimeterPoints = pointsAlongCircle(radiusOfAreaInCircle, 360)
    rng = np.random.default_rng(seed)
    randomPointsInShape = placePointsInsidePerimeter(numberOfCells, circlePerimeterPoints, rng=rng)
    allCellCenters = randomPointsInShape
    tri = Delaunay(allCellCenters)
    delaunayFaceGraph = faceAdjacencyGraphFromDelaunayTriangulation(tri, allCellCenters)
    if visualizeStepsInBetween:
        plotDelaunayTriangulationWithFaceMidPoints(delaunayFaceGraph, allCellCenters, tri)
    return delaunayFaceGraph

def pointsAlongCircle(radius, numberOfPoints):
    anglesOfPoints = np.linspace(0, 2*np.pi, numberOfPoints, endpoint=False)
    x = radius * np.sin(anglesOfPoints)
    y = radius * np.cos(anglesOfPoints)
    return np.concatenate([x,y]).reshape(2, numberOfPoints).T
#endregion

#region mainCodeExecution
def main():
    dataSetname = "Eng2021Cotyledons"  # "Smit2023Cotyledons" # 
    filename = f"Images/{dataSetname}/{dataSetname}.json"
    mfc = MultiFolderContent(filename)
    #tissueContent = list(mfc)[0]
    for tissueContent in mfc:
        print(tissueContent.GetTissueName())
        delaunayFaceGraph = randomizeTissueUsingDelaunayTriangulation(tissueContent)
        nx.get_node_attributes(delaunayFaceGraph, "pos")

if __name__ == '__main__':
    main()
#endregion