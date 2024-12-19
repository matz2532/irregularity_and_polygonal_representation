import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

sys.path.insert(0, "./Code/DataStructures/")

from FolderContent import FolderContent

def extractOrderedPeripheralNodes(graph: nx.Graph):
    peripheralNodes = findPeripheralNodes(graph, selectFirstLayerPeripheralNodes=True)
    depthFirstEdgeOrdering = list(nx.dfs_edges(graph.subgraph(peripheralNodes)))
    assert len(depthFirstEdgeOrdering) > 0, f"Expected more than one edge to order peripheral nodes, {len(depthFirstEdgeOrdering)} == 0"
    orderedPeripheralNodes = []
    for edge in depthFirstEdgeOrdering:
        orderedPeripheralNodes.append(edge[0])
    orderedPeripheralNodes.append(edge[1])
    return orderedPeripheralNodes

def findPeripheralNodes(graph: nx.Graph, selectFirstLayerPeripheralNodes = True):
    peripheralNodes, innerNodes = [], []
    nextNodesToCheck, checkedNodes = [], []
    nodeClosenessCentralitiesOfNodes = nx.closeness_centrality(graph)
    nodeClosenessCentralities = list(nodeClosenessCentralitiesOfNodes.values())
    nodes = list(nodeClosenessCentralitiesOfNodes.keys())
    startingNode = nodes[np.argmax(nodeClosenessCentralities)]
    nextNodesToCheck = [startingNode]
    while len(nextNodesToCheck) > 0:
        currentNode = nextNodesToCheck.pop()
        neighbors = list(graph.neighbors(currentNode))
        nodesOfFirstNeighborhood = [currentNode]
        nodesOfFirstNeighborhood.extend(neighbors)
        firstNeighborhoodGraph = graph.subgraph(nodesOfFirstNeighborhood)
        numberOfNeighbors = len(neighbors)
        numberOfTrianglesOfCells = nx.triangles(firstNeighborhoodGraph)
        if numberOfNeighbors <= 3 or numberOfNeighbors != numberOfTrianglesOfCells[currentNode]:
            peripheralNodes.append(currentNode)
        else:
            innerNodes.append(currentNode)
        checkedNodes.append(currentNode)
        uncheckedNeighbors = np.array(neighbors)[np.isin(neighbors, checkedNodes, invert=True)]
        nextNodesToCheck.extend(list(uncheckedNeighbors))
    innerNodes = np.unique(innerNodes)
    if not selectFirstLayerPeripheralNodes:
        return peripheralNodes
    allInnerNodesNeighbors = np.unique(np.concatenate([list(graph.neighbors(n)) for n in innerNodes]))
    firstPeripheryNodes = allInnerNodesNeighbors[np.isin(allInnerNodesNeighbors, innerNodes, invert=True)]
    trianglesOfPeripheryNodes = nx.triangles(graph.subgraph(firstPeripheryNodes))
    if np.any(np.array(list(trianglesOfPeripheryNodes.values())) > 0):
        raise NotImplementedError("Found a triangle in the first layer of peripheral nodes and the removal of them is not yet implemented.")
    return firstPeripheryNodes

def calculateAdjacencyGraph(tissue: FolderContent, adjacencyListFilenameKey: str or None = None, neighborDistancesFilenameKey: str or None = None):
    assert adjacencyListFilenameKey is not None or neighborDistancesFilenameKey is not None, f"You have to provide either the adjacencyListFilenameKey or the neighborDistancesFilenameKey for the calculation of the cellular adjacency graph of {tissue.GetTissueName()}"
    if adjacencyListFilenameKey is not None:
        adjacencyList = tissue.LoadKeyUsingFilenameDict(adjacencyListFilenameKey)
        return nx.Graph(adjacencyList)
    else:
        neighborDistancesDf = tissue.LoadKeyUsingFilenameDict(neighborDistancesFilenameKey)
        raise NotImplementedError("Extracting the adjacency graph using the neighborDistancesFilenameKey is not yet implemented.")

def rotateImageToFitMatpltlibOrientation(image):
    return np.rot90(np.flip(image, axis=0), k=3)

def overlAyadjacencyGraphOnLabelledImage(tissue, displayInterestedCellsPositionFromLabelledImage: bool = False):
    # from neighborDistances or labelledImageAdjacencyList -> extract adjacencyGraph
    adjacencyGraph = calculateAdjacencyGraph(tissue, adjacencyListFilenameKey="labelledImageAdjacencyList")

    labelledImage = tissue.LoadKeyUsingFilenameDict("labelledImageFilename")

    pos = {}
    for i in adjacencyGraph.nodes:
        pixelPositionsOfNodes = np.where(labelledImage == i)
        pos[i] = np.mean(pixelPositionsOfNodes, axis=1)
    nx.draw_networkx(adjacencyGraph, pos=pos)

    cellDict = tissue.LoadKeyUsingFilenameDict("cellContours") # enthält nur Zellen mit allen junctions known
    posInnerCells = {}
    sortedInterestedCells = np.sort(list(cellDict.keys()))
    if displayInterestedCellsPositionFromLabelledImage:
        # use labelled image to extract all positions -> know bordering cells from difference of cells
        for i, cellId, in enumerate(sortedInterestedCells):
            contour = cellDict[cellId]
            posInnerCells[cellId] = np.mean(contour, axis=0)
    else:
        for innerCellId in sortedInterestedCells:
            posInnerCells[innerCellId] = pos[innerCellId].tolist()

    # remove non-interested cells
    allCells = list(pos.keys())
    cellsToRemove = np.setdiff1d(allCells, sortedInterestedCells)
    adjacencyGraph.remove_nodes_from(cellsToRemove)
    nx.draw_networkx_nodes(adjacencyGraph, posInnerCells, node_color="red")

    # add labelled image
    plt.imshow(rotateImageToFitMatpltlibOrientation(labelledImage))

    plt.gca().legend(('ignored', 'adjacency', 'interested'))
    plt.show()

def main():
    from MultiFolderContent import MultiFolderContent
    dataSetname = "Eng2021Cotyledons"  # "Smit2023Cotyledons" #
    filename = f"Images/{dataSetname}/{dataSetname}.json"
    mfc = MultiFolderContent(filename)
    tissueContent = list(mfc)[0]
    # overlAyadjacencyGraphOnLabelledImage(tissueContent)
    graph = calculateAdjacencyGraph(tissueContent, adjacencyListFilenameKey="labelledImageAdjacencyList")
    peripheralNodes = findPeripheralNodes(graph)
    extractOrderedPeripheralNodes(graph)

if __name__ == '__main__':
    main()