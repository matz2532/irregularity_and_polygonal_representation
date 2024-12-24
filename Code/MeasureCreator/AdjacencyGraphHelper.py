import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

sys.path.insert(0, "./Code/DataStructures/")

from FolderContent import FolderContent
from Utils import findSharedPoints

def extractOrderedPeripheralNodes(graph: nx.Graph, returnInnerNodesToo: bool = True):
    peripheralNodes = findPeripheralNodes(graph, selectFirstLayerPeripheralNodes=True, returnInnerNodesToo=returnInnerNodesToo)
    if returnInnerNodesToo:
        peripheralNodes, innerNodes = peripheralNodes
    depthFirstEdgeOrdering = list(nx.dfs_edges(graph.subgraph(peripheralNodes)))
    assert len(depthFirstEdgeOrdering) > 0, f"Expected more than one edge to order peripheral nodes, {len(depthFirstEdgeOrdering)} == 0"
    orderedPeripheralNodes = []
    for edge in depthFirstEdgeOrdering:
        orderedPeripheralNodes.append(edge[0])
    orderedPeripheralNodes.append(edge[1])
    if returnInnerNodesToo:
        return orderedPeripheralNodes, innerNodes
    else:
        return orderedPeripheralNodes

def containsAnyTriangles(graph: nx.Graph):
    trianglesOfNodes = nx.triangles(graph)
    triangleValueOfNodes = np.array(list(trianglesOfNodes.values()))
    hasTriangle = triangleValueOfNodes > 0
    return np.any(hasTriangle)

def determinePeripheryViaJunctionsFor(selectedIds: list, junctionsOfIds: dict, neighborsOfSelectedIds: dict or None = None):
    peripheralIds, innerIds = [], []
    allIdsWithJunctions = np.sort(list(junctionsOfIds.keys()))
    # ensure existence of junctions for all ids (and neighbors when given)
    hasSelectedIdNoJunction = np.isin(selectedIds, allIdsWithJunctions, invert=True)
    assert not np.any(hasSelectedIdNoJunction), f"The ids {np.array(selectedIds)[hasSelectedIdNoJunction]} do not have junctions, only the following ids have junctions {allIdsWithJunctions}"
    if neighborsOfSelectedIds is not None:
        neighborIds = np.unique(np.concatenate(list(neighborsOfSelectedIds.values())))
        haveNeighborsNoJunction = np.isin(neighborIds, allIdsWithJunctions, invert=True)
        assert not np.any(haveNeighborsNoJunction), f"The neighbor ids {neighborIds[haveNeighborsNoJunction]} do not have junctions, only the following ids have junctions {allIdsWithJunctions}"
    for i in selectedIds:
        if neighborsOfSelectedIds is not None and i in neighborsOfSelectedIds:
            checkWithOtherIds = neighborsOfSelectedIds[i]
        else:
            checkWithOtherIds = allIdsWithJunctions[np.isin(allIdsWithJunctions, i, invert=True)]
        pooledJunctionPositions = []
        for j, junctions in junctionsOfIds.items():
            if j in checkWithOtherIds:
                pooledJunctionPositions.append(junctions)
        pooledJunctionPositions = np.concatenate(pooledJunctionPositions)
        junctionsOfCurrentId = junctionsOfIds[i]
        _, indicesOfSharedPoints = findSharedPoints(junctionsOfCurrentId, pooledJunctionPositions, tolerance=np.sqrt(5), returnAllSharedIndices=True)
        if np.all(np.array([len(whereFound) > 3 for whereFound in indicesOfSharedPoints])):
            innerIds.append(i)
        else:
            peripheralIds.append(i)
    return peripheralIds, innerIds

def determinePeripheryViaTriangles(graph: nx.Graph):
    peripheralNodes, innerNodes = [], []
    nextNodesToCheck, checkedNodes, checkViaJunctions = [], [], []
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
        if containsAnyTriangles(graph.subgraph(neighbors)):
            checkViaJunctions.append(currentNode)
        else:
            numberOfNeighbors = len(neighbors)
            numberOfTrianglesOfCells = nx.triangles(firstNeighborhoodGraph)
            if numberOfNeighbors < 3 or numberOfNeighbors != numberOfTrianglesOfCells[currentNode]:
                peripheralNodes.append(currentNode)
            else:
                innerNodes.append(currentNode)
        checkedNodes.append(currentNode)
        uncheckedNeighbors = np.array(neighbors)[np.isin(neighbors, checkedNodes, invert=True)]
        nextNodesToCheck.extend(list(uncheckedNeighbors))
    return peripheralNodes, innerNodes, checkViaJunctions

def findPeripheralNodes(graph: nx.Graph, selectFirstLayerPeripheralNodes = True, returnInnerNodesToo: bool = True,
                        junctionsOfNodes: dict or None = None):
    peripheralNodes, innerNodes, checkViaJunctions = determinePeripheryViaTriangles(graph)
    if len(checkViaJunctions) > 0 and junctionsOfNodes is not None:
        neighborsOfSelectedNodes = {nodeToCheck: list(graph.neighbors(nodeToCheck)) for nodeToCheck in checkViaJunctions}
        additionalPeripheralNodes, additionalInnerNodes = determinePeripheryViaJunctionsFor(selectedIds=checkViaJunctions, junctionsOfIds=junctionsOfNodes, neighborsOfSelectedIds=neighborsOfSelectedNodes)
        peripheralNodes.extend(additionalPeripheralNodes)
        innerNodes.extend(additionalInnerNodes)
    innerNodes = np.array(innerNodes)
    if not selectFirstLayerPeripheralNodes:
        if returnInnerNodesToo:
            return peripheralNodes, innerNodes
        else:
            return peripheralNodes
    allInnerNodesNeighbors = np.unique(np.concatenate([list(graph.neighbors(n)) for n in innerNodes]))
    firstPeripheryNodes = allInnerNodesNeighbors[np.isin(allInnerNodesNeighbors, innerNodes, invert=True)]
    trianglesOfPeripheryNodes = nx.triangles(graph.subgraph(firstPeripheryNodes))
    if np.any(np.array(list(trianglesOfPeripheryNodes.values())) > 0):
        raise NotImplementedError("Found a triangle in the first layer of peripheral nodes and the removal of them is not yet implemented.")
    if returnInnerNodesToo:
        return firstPeripheryNodes, innerNodes
    else:
        return firstPeripheryNodes

def extractAdjacencyGraph(tissue: FolderContent, adjacencyListFilenameKey: str or None = None, neighborDistancesFilenameKey: str or None = None):
    assert adjacencyListFilenameKey is not None or neighborDistancesFilenameKey is not None, f"You have to provide either the adjacencyListFilenameKey or the neighborDistancesFilenameKey for the calculation of the cellular adjacency graph of {tissue.GetTissueName()}"
    if adjacencyListFilenameKey is not None:
        adjacencyList = tissue.LoadKeyUsingFilenameDict(adjacencyListFilenameKey)
        return nx.Graph(adjacencyList)
    else:
        neighborDistancesDf = tissue.LoadKeyUsingFilenameDict(neighborDistancesFilenameKey)
        raise NotImplementedError("Extracting the adjacency graph using the neighborDistancesFilenameKey is not yet implemented.")

def rotateImageToFitMatpltlibOrientation(image):
    return np.rot90(np.flip(image, axis=0), k=3)

def overlAyadjacencyGraphOnLabelledImage(tissue, displayInterestedCellsPositionFromLabelledImage: bool = False, adjacencyGraph: nx.Graph or None = None):
    if adjacencyGraph is None:
        # from neighborDistances or labelledImageAdjacencyList -> extract adjacencyGraph
        adjacencyGraph = extractAdjacencyGraph(tissue, adjacencyListFilenameKey="labelledImageAdjacencyList")

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
    sys.path.insert(0, "./Code/ImageToRawDataConversion/")
    from LabelledImageToGraphConverter import LabelledImageToGraphConverter
    from MultiFolderContent import MultiFolderContent
    dataSetname = "Eng2021Cotyledons"  # "Smit2023Cotyledons" #
    filename = f"Images/{dataSetname}/{dataSetname}.json"
    mfc = MultiFolderContent(filename)
    tissueContent = list(mfc)[0]
    # overlAyadjacencyGraphOnLabelledImage(tissueContent)

    # original adjacency graph, where adjacency of cells without contours was ignored with each other
    # graph = extractAdjacencyGraph(tissueContent, adjacencyListFilenameKey="labelledImageAdjacencyList")
    myLabelledImageToGraphConverter = LabelledImageToGraphConverter(folderContent=tissueContent, selectedCellIds=[])
    adjacencyList = myLabelledImageToGraphConverter.GetAdjacencyList()
    graph = nx.Graph(adjacencyList)
    junctionsOfNodes = tissueContent.LoadKeyUsingFilenameDict("orderedJunctionsPerCellFilename")
    peripheralNodes = findPeripheralNodes(graph, returnInnerNodesToo=False, junctionsOfNodes=junctionsOfNodes)
    extractOrderedPeripheralNodes(graph)

if __name__ == '__main__':
    main()