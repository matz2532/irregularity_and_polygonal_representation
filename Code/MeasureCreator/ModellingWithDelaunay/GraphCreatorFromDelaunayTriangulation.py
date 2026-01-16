import innerCircleOfTriangle
import networkx as nx
import numpy as np

from scipy.spatial import Delaunay
from shapely.geometry import Polygon

def pointsAdjacencyGraphFromDelaunayTriangulation(tri):
    graph = adjacencyGraphFromArray(tri.simplices)
    positions = dict(zip(np.unique(tri.simplices), tri.points))
    nx.set_node_attributes(graph, positions, "pos")
    return graph

def faceAdjacencyGraphFromDelaunayTriangulation(tri, cellCenters, ax=None):
    graph = adjacencyGraphFromArray(tri.neighbors, False)
    faceMidPoints = extractFaceMidPoint(cellCenters, tri, ax=ax)
    graph.remove_node(-1)
    positions = dict(zip(graph.nodes, faceMidPoints))
    nx.set_node_attributes(graph, positions, "pos")
    cellsFormingTriangle = {i: vertexIndicesOfTriangle for i, vertexIndicesOfTriangle in enumerate(tri.simplices)}
    nx.set_node_attributes(graph, cellsFormingTriangle, "adjacent cells")
    return graph

def adjacencyGraphFromArray(array, pointBack=True):
    indices = np.unique(array)
    adjacencyGraph = {}
    for pointIndex in indices:
        presentInRows = np.where(array == pointIndex)[0]
        if pointBack:
            adjacencyGraph[pointIndex] = np.unique(array[presentInRows]).tolist()
        else:
            adjacencyGraph[pointIndex] = presentInRows
        if pointIndex in adjacencyGraph[pointIndex]:
            adjacencyGraph[pointIndex].remove(pointIndex)
    graph = nx.Graph(adjacencyGraph)
    return graph

import matplotlib.pyplot as plt
import matplotlib
def pointsAlongCircle(radius, numberOfPoints):
    anglesOfPoints = np.linspace(0, 2*np.pi, numberOfPoints, endpoint=False)
    x = radius * np.sin(anglesOfPoints)
    y = radius * np.cos(anglesOfPoints)
    return np.concatenate([x,y]).reshape(2, numberOfPoints).T
def extractFaceMidPoint(faceVerticesPositions, tri, centerFindingMethod="innerCircleMidPoint", ax=None):
    faceMidPoints = np.full((len(tri.simplices), 2), 0, dtype=float)
    for i, vertexIndicesOfTriangle in enumerate(tri.simplices):
        vertexPositionsOfCurrentFace = faceVerticesPositions[vertexIndicesOfTriangle, :]
        if centerFindingMethod == "innerCircleMidPoint":
            midPointOfFace = innerCircleOfTriangle.calcInnerCircleOfTriangle(vertexPositionsOfCurrentFace)
        elif centerFindingMethod == "centroid":
            triangleAsPolygon = Polygon(vertexPositionsOfCurrentFace)
            midPointOfFace = [triangleAsPolygon.centroid.x, triangleAsPolygon.centroid.y]
        else:
            midPointOfFace = np.mean(vertexPositionsOfCurrentFace, axis=0)
        faceMidPoints[i, :] = midPointOfFace
    return faceMidPoints

def testFunctionality():
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, "./Code/DataStructures/")
    from MultiFolderContent import MultiFolderContent

    dataSetname = "Eng2021Cotyledons"  # "Smit2023Cotyledons" #
    filename = f"Images/{dataSetname}/{dataSetname}.json"
    mfc = MultiFolderContent(filename)
    geometricCentersPerCellKey = "geometricCentersPerCell"
    tissueContent = list(mfc)[0]
    geometricCentersPerCell = tissueContent.LoadKeyUsingFilenameDict(geometricCentersPerCellKey)
    cellCenters = np.array(list(geometricCentersPerCell.values()))
    tri = Delaunay(cellCenters)
    delaunayVectorGraph = pointsAdjacencyGraphFromDelaunayTriangulation(tri)
    delaunayFaceGraph = faceAdjacencyGraphFromDelaunayTriangulation(tri, cellCenters)
    nx.draw_networkx_edges(delaunayFaceGraph, pos=nx.get_node_attributes(delaunayFaceGraph, "pos"))
    nx.draw_networkx(delaunayVectorGraph, pos=nx.get_node_attributes(delaunayVectorGraph, "pos"))
    plt.show()

    plt.triplot(cellCenters[:, 0], cellCenters[:, 1], tri.simplices.copy())
    plt.plot(cellCenters[:, 0], cellCenters[:, 1], 'o')
    for i, p in enumerate(cellCenters):
        plt.text(*p, i, horizontalalignment='center', size='small')
    faceMidPoints = extractFaceMidPoint(cellCenters, tri)
    for i, p in enumerate(faceMidPoints):
        plt.text(*p, i, horizontalalignment='center', size='small', color="blue")

    # plt.plot(cellCenters[[0, 5],0], cellCenters[[0, 5],1], 'o')
    # plt.plot(cellCenters[[7, -1, 36, 7],0], cellCenters[[ 7, -1, 36, 7],1], '-', c="red")
    # plt.plot(cellCenters[[21, 31, 17, 21],0], cellCenters[[21, 31, 17, 21],1], '-', c="blue")
    u, c = np.unique(tri.neighbors, return_counts=True)
    print(np.where(tri.neighbors == 0))
    print(dict(zip(u, c)), sep="\n")
    for i in np.where(tri.neighbors == 0)[0]:
        plt.plot(cellCenters[tri.simplices[i], 0], cellCenters[tri.simplices[i], 1], 'o')
    # plt.plot(cellCenters[[0,1,2, 0],0], cellCenters[[ 0,1,2, 0],1], '-', c="red")
    # plt.plot(cellCenters[[13,4,0, 13],0], cellCenters[[13,4,0, 13],1], '-', c="blue")
    # print("number of neighbors for vertex 0",np.sum(np.isin(tri.simplices, 0)))
    # plt.plot(cellCenters[[ 7, -1, 36],0], cellCenters[[ 7, -1, 36],1], 'o')
    plt.show()

if __name__ == '__main__':
    testFunctionality()
