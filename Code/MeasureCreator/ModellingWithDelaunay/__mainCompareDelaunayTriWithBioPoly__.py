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

def plotDelaunayTriangulationWithFaceMidPoints(delaunayFaceGraph, allCellCenters, tri, biologicalJunctions=None, ax=None,
                                               addIndexOfCellCenters=False, addIndexOfJunction=False):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,8), constrained_layout=True)
    nx.draw_networkx_edges(delaunayFaceGraph, pos=nx.get_node_attributes(delaunayFaceGraph, "pos"), label="triangulated edges", ax=ax)
    plt.triplot(allCellCenters[:, 0], allCellCenters[:, 1], tri.simplices.copy())
    if biologicalJunctions is not None:
        isFirstCell = True
        for junctionsOfCell in biologicalJunctions.values():
            junctionsToPlot = np.concatenate([junctionsOfCell, [junctionsOfCell[0]]], axis=0)
            plt.plot(junctionsToPlot[:, 0], junctionsToPlot[:, 1], color="lightblue", label= "original edges" if isFirstCell else None, ax=ax)
            if isFirstCell:
                isFirstCell = False
    ax.plot(allCellCenters[:, 0], allCellCenters[:, 1], 'o', label="cell center")
    ylimDistance = np.max(allCellCenters, axis=0)[0] - np.min(allCellCenters, axis=0)[0]
    
    if addIndexOfCellCenters:
        for i, p in enumerate(allCellCenters):
            x, y = p
            y += ylimDistance * 0.02
            plt.text(x, y, i, horizontalalignment='center', size='small')
    if addIndexOfJunction:
        for i, p in nx.get_node_attributes(delaunayFaceGraph, "pos").items():
            x, y = p
            ax.text(x, y, i, horizontalalignment='center', size='small')

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