#region Imports
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

sys.path.insert(0, "./Code/DataStructures/")

from GraphCreatorFromDelaunayTriangulation import faceAdjacencyGraphFromDelaunayTriangulation
from FolderContent import FolderContent
from MultiFolderContent import MultiFolderContent
from scipy.spatial import Delaunay
#endregion

#region MainCode
def createDelaunayFromCellCentersOf(tissue: FolderContent, visualizeStepsInBetween=True):
    orderedJunctionsPerCell = tissue.LoadKeyUsingFilenameDict("orderedJunctionsPerCellFilename")
    centerOfCells = {}
    for cellId, orderedJunctions in orderedJunctionsPerCell.items():
        centerOfCells[cellId] = np.mean(orderedJunctions, axis=0)
    allCellCenters = np.concatenate(list(centerOfCells.values())).reshape(len(centerOfCells), 2)
    tri = Delaunay(allCellCenters)
    delaunayFaceGraph = faceAdjacencyGraphFromDelaunayTriangulation(tri, allCellCenters)
    if visualizeStepsInBetween:
        plotDelaunayTriangulationWithFaceMidPoints(delaunayFaceGraph, allCellCenters, tri, orderedJunctionsPerCell)
    return delaunayFaceGraph
    
def plotDelaunayTriangulationWithFaceMidPoints(delaunayFaceGraph, allCellCenters, tri, biologicalJunctions=None): #: None|dict[int, np.ndarray]
    nx.draw_networkx_edges(delaunayFaceGraph, pos=nx.get_node_attributes(delaunayFaceGraph, "pos"), label="triangulated edges")
    plt.triplot(allCellCenters[:, 0], allCellCenters[:, 1], tri.simplices.copy())
    if biologicalJunctions is not None:
        isFirstCell = True
        for junctionsOfCell in biologicalJunctions.values():
            junctionsToPlot = np.concatenate([junctionsOfCell, [junctionsOfCell[0]]], axis=0)
            plt.plot(junctionsToPlot[:, 0], junctionsToPlot[:, 1], color="lightblue", label= "original edges" if isFirstCell else None)
            if isFirstCell:
                isFirstCell = False
    plt.plot(allCellCenters[:, 0], allCellCenters[:, 1], 'o', label="cell center")
    ylimDistance = np.max(allCellCenters, axis=0)[0] - np.min(allCellCenters, axis=0)[0]
    for i, p in enumerate(allCellCenters):
        x, y = p
        y += ylimDistance * 0.02
        plt.text(x, y, i, horizontalalignment='center', size='small')
    plt.legend()
    plt.show()
#endregion

#region mainCodeExecution
def main():
    dataSetname = "Eng2021Cotyledons"  # "Smit2023Cotyledons" #
    filename = f"Images/{dataSetname}/{dataSetname}.json"
    mfc = MultiFolderContent(filename)
    tissueContent = list(mfc)[0]
    print(tissueContent.GetTissueName())
    delaunayFaceGraph = createDelaunayFromCellCentersOf(tissueContent)
    nx.get_node_attributes(delaunayFaceGraph, "pos")

if __name__ == '__main__':
    main()
#endregion