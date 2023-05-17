import json
import networkx as nx
import numpy as np
import re

from pathlib import Path

class MGXGraphFromPlyFileReader (object):

    # default not run variables
    adjacentLabels=None
    plyFilename=""
    # default filename extension (for saving)
    resultsNameExtension="_adjacencyList.json"
    # default patterns in ply file
    endHeaderPattern="end_header"
    beforeContourPattern="element vertex (\d+)"
    lineSeperator=" "

    def __init__(self, plyGraphFilename=None):
        if not plyGraphFilename is None:
            self.readGraphFromPlyFile(plyGraphFilename)

    def GetGraph(self):
        return self.graph

    def PrintAdjacencyList(self):
        print(self.adjacencyListDict)

    def SaveGraphsAdjacencyList(self, filenameToSave=None,
                                tissueBaseFilename=None,
                                resultsNameExtension=None,
                                nameIdx=-1, onlyKeepCellLabels=None):
        if filenameToSave is None:
            if resultsNameExtension is None:
                resultsNameExtension = self.resultsNameExtension
            if not tissueBaseFilename is None:
                filenameToSave = tissueBaseFilename + resultsNameExtension
            else:
                parentOfFilename = Path(self.plyFilename).parent
                if not nameIdx is None:
                    extraName = parentOfFilename.parts[nameIdx]
                    if resultsNameExtension[0] != "_":
                        extraName = extraName + "_"
                else:
                    extraName = ""
                filenameToSave = parentOfFilename.joinpath(extraName + resultsNameExtension)
        if not onlyKeepCellLabels is None:
            adjacencyListDict = {}
            for cellLabel in onlyKeepCellLabels:
                adjacencyListDict[cellLabel] = self.adjacencyListDict[cellLabel]
        else:
            adjacencyListDict = self.adjacencyListDict
        with open(filenameToSave, "w") as fh:
            json.dump(adjacencyListDict, fh)
        return filenameToSave

    def readGraphFromPlyFile(self, plyGraphFilename):
        self.plyFilename = plyGraphFilename
        self.adjacentLabels = self.extractAdjacencyOfLabels(plyGraphFilename)
        self.adjacencyListDict = self.createAdjacencyListFromTableColumns(self.adjacentLabels)
        self.graph = nx.from_dict_of_lists(self.adjacencyListDict)

    def extractAdjacencyOfLabels(self, plyGraphFilename, sep=None):
        if sep is None:
            sep = self.lineSeperator
        allAdjacentLabels = []
        lineCountAfterHeader = 0
        linesBeforeContourIndicesOfCells = None
        isAfterHeader = False
        with open(plyGraphFilename, "r") as fh:
            line = fh.readline()
            while line:
                if isAfterHeader:
                    if lineCountAfterHeader >= linesBeforeContourIndicesOfCells:
                        adjacentLabels = line.split(sep)
                        adjacentLabels = [int(i) for i in adjacentLabels]
                        allAdjacentLabels.append(adjacentLabels)
                    lineCountAfterHeader += 1
                else:
                    line = line.strip()
                    isLineHeader = re.search(self.endHeaderPattern, line)
                    searchForBeforeContourIndicesOfCells = re.search(self.beforeContourPattern, line)
                    if isLineHeader:
                        isAfterHeader = True
                    if searchForBeforeContourIndicesOfCells:
                        linesBeforeContourIndicesOfCells = int(searchForBeforeContourIndicesOfCells.group(1))
                line = fh.readline()
        return allAdjacentLabels

    def createAdjacencyListFromTableColumns(self, adjacentLabelsArray):
        adjacencyListDict = {}
        for fromNode, toNode in adjacentLabelsArray:
            if not fromNode in adjacencyListDict:
                adjacencyListDict[fromNode] = []
            adjacencyListDict[fromNode].append(toNode)
        return adjacencyListDict

def main():
    import matplotlib.pyplot as plt
    tissueReplicateId = "20200220 WT S1"
    baseName = f"Images/full cotyledons/WT/{tissueReplicateId}/{tissueReplicateId}"
    plyGraphFilename = baseName + "_only junctions_cellGraph.ply"
    myMGXGraphFromPlyFileReader = MGXGraphFromPlyFileReader(plyGraphFilename)
    graph = myMGXGraphFromPlyFileReader.GetGraph()
    myMGXGraphFromPlyFileReader.SaveGraphsAdjacencyList()
    nx.draw_spectral(graph)
    plt.show()

if __name__ == '__main__':
    main()
