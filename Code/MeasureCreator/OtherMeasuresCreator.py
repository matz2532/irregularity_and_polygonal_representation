import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import shapely
import sys
import warnings

sys.path.insert(0, "../../OfficialGitHubs/GraVisGUI-1.2/SourceCode/")

from skimage.segmentation import watershed
from ShapeGUI import VisGraph
from shapely import MultiPoint, Polygon, convex_hull
from shapely.geometry import LineString, LinearRing, base

class OtherMeasuresCreator (object):

    pointList=None
    additionalCellInfo: str = ""

    def __init__(self, pointList=None, additionalCellInfo: str = ""):
        self.pointList = pointList
        self.additionalCellInfo = additionalCellInfo

    def SetAdditionalCellInfo(self, additionalCellInfo):
        self.additionalCellInfo = additionalCellInfo

    def calcLobyness(self, pointList: list):
        perimeterOfConvexHull = convex_hull(Polygon(pointList)).length
        perimeter = Polygon(pointList).length
        lobyness = perimeter / perimeterOfConvexHull
        return lobyness

    def calcRelativeCompletenessForOutlines(self, outlinesOfCells: dict, resolutionInDistancePerPixel: float = 1, additionalBoarder: np.ndarray = np.array([10,10])):
        labelledImage, imageToOriginalCellIdConverter = self.createLabelledImage(outlinesOfCells, additionalBoarder)
        relativeCompletenessOfCells = {}
        visibilityGraphCalculator = VisGraph(runOnInit=False)
        for labelledImageCellId, originalCellId in imageToOriginalCellIdConverter.items():
            visibilityGraph, cellContour, = visibilityGraphCalculator.create_visibility_graph(labelledImage, labelledImageCellId, resolutionInDistancePerPixel)
            assert visibilityGraph.number_of_nodes() != 0, f"The cell {labelledImageCellId} has no nodes when creating a visibility graph{' with the info: '+self.additionalCellInfo if self.additionalCellInfo else ''} with {len(cellContour)=}. "
            relativeCompleteness = visibilityGraphCalculator.compute_graph_complexity(visibilityGraph)
            relativeCompletenessOfCells[originalCellId] = relativeCompleteness
        return relativeCompletenessOfCells

    def createLabelledImage(self, outlinesOfCells, additionalBoarder: int = 10):
        max = np.concatenate([np.max(contour, axis=0) for contour in outlinesOfCells.values()])
        shape = np.max(max.reshape(len(max) // 2, 2), axis=0) + 2*additionalBoarder
        labelledImage = np.zeros(shape, dtype=int)
        markerImage = np.zeros(shape, dtype=int)
        imageToOriginalCellIdConverter = {}
        for imageCellId, (originalCellId, outline) in enumerate(outlinesOfCells.items()):
            imageCellId += 3  # to avoid having the background
            for x, y in outline:
                labelledImage[x + additionalBoarder//2, y + additionalBoarder//2] = imageCellId
            imageToOriginalCellIdConverter[imageCellId] = originalCellId
            midPoint = np.mean(outline, axis=0).astype(int)
            markerImage[midPoint[0], midPoint[1]] = imageCellId
        markerImage[-1, -1] = 1  # need to add background marker with label value of 1
        labelledImage = watershed(labelledImage, markers=markerImage)
        labelledImage = labelledImage.astype(int)
        return labelledImage, imageToOriginalCellIdConverter

    def calcRelativeCompletenessFromPointList(self, positionList, numberOfEquallySpacedPoints: int = None, numberOfPointsPerNumberOfPoints: int = 8):
        if numberOfEquallySpacedPoints is None:
            numberOfEquallySpacedPoints = numberOfPointsPerNumberOfPoints * len(positionList)
        extendedPositionGeometry = self.equallySpacePointsBetween(positionList, numberOfEquallySpacedPoints)
        visibilityGraph = self.calcVisbilityGraphFromGeometery(extendedPositionGeometry)
        return VisGraph.compute_graph_complexity(None, visibilityGraph)

    def equallySpacePointsBetween(self, positionList: list, numberOfEquallySpacedPoints: int, connectLastAndFirstPoint: bool = True,
                                  roundToDecimalPlace: int = None, asNumpyNDArray: bool = False):
        """
        adapted from https://stackoverflow.com/questions/62990029/how-to-get-equally-spaced-points-on-a-line-in-shapely
        !!! small rounding error can result in corner positions of points be different to points on a straight line between the end points
        !!!  this can result in the corner points being connected, while visually they should not have been connected !!!
        """
        if len(positionList) >= numberOfEquallySpacedPoints:
            warnings.warn(f"The number of equally spaced points={numberOfEquallySpacedPoints} is lower or equal to the number of given positions={len(positionList)} of {positionList=}")
        if connectLastAndFirstPoint:
            line = LinearRing(positionList)
        else:
            line = LineString(positionList)
        distances = np.linspace(0, line.length, numberOfEquallySpacedPoints, endpoint=False)
        points = [line.interpolate(distance) for distance in distances]
        if not roundToDecimalPlace is None:
            newPoints = []
            for p in points:
                newPoints.append(shapely.geometry.Point(np.round(p.xy, roundToDecimalPlace).flatten().tolist()))
                points = newPoints
        if connectLastAndFirstPoint:
            multipoint = LinearRing(points)
        else:
            multipoint = LineString(points)
        if asNumpyNDArray:
            return shapely.get_coordinates(multipoint)
        return multipoint

    def calcVisbilityGraphFromGeometery(self, linearRing: LinearRing, resolutionInDistancePerPixel: float = 1):
        self.numberOfPoints = len(linearRing.coords) - 1 # the minus one comes from the fact we have a ring where .coords[0] == .coords[-1]
        nodeIndices = list(range(self.numberOfPoints))
        pointsOfGeometry = shapely.get_coordinates(linearRing)
        # initiate graph with nodes and their positions
        nodePositions = dict(zip(nodeIndices, pointsOfGeometry.tolist()))
        visibilityGraph = nx.Graph()
        visibilityGraph.add_nodes_from(nodeIndices)
        nx.set_node_attributes(visibilityGraph, nodePositions, name="pos")
        # add edges to graph
        edges, edgeLengthsAttribute = [], {}
        for node1, node2 in itertools.combinations(nodeIndices, r=2):
            canNodesSeeEachOther = self.determineVisibility(linearRing, node1, node2)
            if canNodesSeeEachOther:
                length = resolutionInDistancePerPixel * np.linalg.norm(pointsOfGeometry[node2] - pointsOfGeometry[node1])
                edges.append((node1, node2))
                edgeLengthsAttribute[(node1, node2)] = length
        visibilityGraph.add_edges_from(edges)
        nx.set_edge_attributes(visibilityGraph, edgeLengthsAttribute, "length")
        return visibilityGraph

    def determineVisibility(self, linearRing: LinearRing, betweenI: int, betweenJ: int, visualizeIntersections: bool = False, visualizeWarning: bool = True):
        isForwardNeighbor = betweenI + 1 == betweenJ
        isBackwardNeighbor = betweenI - 1 == betweenJ or (betweenI == 0 and betweenJ + 1 == self.numberOfPoints)
        if isForwardNeighbor or isBackwardNeighbor:
            canNodesSeeEachOther = True
        else:
            canNodesSeeEachOther = self.calcWhetherEdgeOnlyLocatedInside(linearRing, betweenI, betweenJ, visualizeIntersections, visualizeWarning)
        return canNodesSeeEachOther

    def calcWhetherEdgeOnlyLocatedInside(self, linearRing: LinearRing, betweenI: int, betweenJ: int, visualizeIntersections: bool = False, visualizeWarning: bool = True):
        outsidePolygon = self.calcOutSidePolygon(Polygon(linearRing))
        isNextNext = betweenI + 2 == betweenJ or betweenI - 2 == betweenJ or (betweenI == 0 and betweenJ + 2 == self.numberOfPoints)
        visibilityLine = LineString([linearRing.coords[betweenI], linearRing.coords[betweenJ]])
        isLineIntersectingWithOutside = shapely.intersection(outsidePolygon, visibilityLine)
        if type(isLineIntersectingWithOutside) == MultiPoint:
            canNodesSeeEachOther = True
        elif type(isLineIntersectingWithOutside) == shapely.geometry.GeometryCollection:
            canNodesSeeEachOther = False
        elif type(isLineIntersectingWithOutside) == LineString:
            if isNextNext:
                lengthOverNext = LineString([linearRing.coords[betweenI], linearRing.coords[betweenI + 1], linearRing.coords[betweenJ]]).length
                canNodesSeeEachOther = np.abs(visibilityLine.length - lengthOverNext) < 1E-13
            else:
                canNodesSeeEachOther = False
        elif type(isLineIntersectingWithOutside) == shapely.geometry.MultiLineString:
            if isNextNext:
                lengthOverNext = LineString([linearRing.coords[betweenI], linearRing.coords[betweenI + 1], linearRing.coords[betweenJ]]).length
                canNodesSeeEachOther = np.abs(visibilityLine.length - lengthOverNext) < 1E-13
            else:
                canNodesSeeEachOther = False
        else:
            warnings.warn(f"The response to the type {type(isLineIntersectingWithOutside)} is not yet implemented")
            canNodesSeeEachOther = False
            if visualizeWarning:
                print(betweenI, betweenJ, type(isLineIntersectingWithOutside), canNodesSeeEachOther)
                self.plotGeometryAndIntersections(linearRing, isLineIntersectingWithOutside)
        if visualizeIntersections:
            try:
                print(f"{betweenI=}, {betweenJ=}, {canNodesSeeEachOther=}, posI={linearRing.coords[betweenI]}, posJ={linearRing.coords[betweenJ]}, {visibilityLine.length=}, {lengthOverNext=}, typeOfIntersection={type(isLineIntersectingWithOutside)}")
            except:
                print(f"{betweenI=}, {betweenJ=}, {canNodesSeeEachOther=}, posI={linearRing.coords[betweenI]}, posJ={linearRing.coords[betweenJ]}, typeOfIntersection={type(isLineIntersectingWithOutside)}")
            self.plotGeometryAndIntersections(linearRing, isLineIntersectingWithOutside)
        return canNodesSeeEachOther

    def calcOutSidePolygon(self, interiorPolygon: Polygon, boarderPercentage: float = 0.025):
        positionList = shapely.get_coordinates(interiorPolygon)
        offSet = boarderPercentage / np.sqrt(2) * np.linalg.norm(np.max(positionList, axis=0) - np.min(positionList, axis=0))
        min = np.min(positionList, axis=0) - offSet
        max = np.max(positionList, axis=0) + offSet
        outsidePolygon = Polygon([[min[0], min[1]], [min[0], max[1]], [max[0], max[1]], [max[0], min[1]]])
        outsidePolygon = outsidePolygon.difference(interiorPolygon)
        return outsidePolygon

    def plotGeometryAndIntersections(self, linearRing: base.BaseGeometry, intersection: MultiPoint, title: str = ""):
        plt.plot(shapely.get_coordinates(linearRing)[:, 0], shapely.get_coordinates(linearRing)[:, 1], marker="o", label="geometry")
        if type(intersection) == LineString:
            plt.plot(shapely.get_coordinates(intersection)[:, 0], shapely.get_coordinates(intersection)[:, 1], lw=2, marker="X", label="intersections")
        else:
            for g in intersection.geoms:
                plt.plot(shapely.get_coordinates(g)[:, 0], shapely.get_coordinates(g)[:, 1], lw=2, marker="X", label="intersections")
        plt.legend()
        if title != "":
            plt.title(title)
        plt.show()

def main():
    sys.path.insert(0, "./Code/DataStructures/")
    from MultiFolderContent import MultiFolderContent
    outlineFilenameKey = "cellContours"
    allFolderContentsFilename = "Images/Eng2021Cotyledons.json"
    folderContent = list(MultiFolderContent(allFolderContentsFilename))[0]
    cellOutlines = folderContent.LoadKeyUsingFilenameDict(outlineFilenameKey)
    OtherMeasuresCreator().calcRelativeCompletenessForOutlines(cellOutlines)

def mainTestLobynessForArtificalPointList():
    import json
    filename = "Images/ArtificialPolygons/complexPolygons.json"
    with open(filename, "r") as fh:
        file = json.load(fh)
    numberOfPointsPer = np.arange(1, 90)
    lobynessOfShapes = {}
    for name, positionList in file.items():
        # if name in ["roundish polygon", "star polygon"]:
        #     continue
        positionList = np.asarray(positionList)
        midPoint = np.mean(positionList, axis=0)
        positionList -= midPoint
        lobynessList = []
        for i in numberOfPointsPer:
            max = np.max(positionList, axis=0)
            positionList /= max
            positionList *= i
            lobyness = OtherMeasuresCreator().calcLobyness(positionList)
            lobynessList.append(lobyness)
        lobynessOfShapes[name] = lobynessList
    fig, ax = plt.subplots()
    for name, completenesses in lobynessOfShapes.items():
        ax.plot(numberOfPointsPer, completenesses, marker="o", label=name)
    plt.legend()
    plt.xlabel("size")
    plt.ylabel("lobyness")
    plt.show()

def mainTestRelativeCompletenessForArtificalPointList():
    import json
    filename = "Images/ArtificialPolygons/complexPolygons.json"
    numberOfPointsPer = [1, 2, 3, 5, 8, 11, 15]#, 19, 25]
    with open(filename, "r") as fh:
        file = json.load(fh)
    completenessesOfShapes = {}
    for name, positionList in file.items():
        completenesses = []
        for numberOfPointsPerNumberOfPoints in numberOfPointsPer:
            relativeCompleteness = OtherMeasuresCreator().calcRelativeCompletenessFromPointList(positionList, numberOfPointsPerNumberOfPoints=numberOfPointsPerNumberOfPoints)
            completenesses.append(relativeCompleteness)
        completenessesOfShapes[name] = completenesses
    fig, ax = plt.subplots()
    for name, completenesses in completenessesOfShapes.items():
        ax.plot(numberOfPointsPer, completenesses, marker="o", label=name)
    plt.legend()
    plt.xlabel("# of nodes per corner")
    plt.ylabel("relative completeness")
    plt.show()

if __name__ == '__main__':
    # mainTestRelativeCompletenessForArtificalPointList()
    mainTestLobynessForArtificalPointList()
