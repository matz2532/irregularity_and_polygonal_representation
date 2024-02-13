import networkx as nx
import numpy as np
import pandas as pd
import sys
import time
import warnings

sys.path.insert(0, "./Code/DataStructures/")
sys.path.insert(0, "./Code/ImageToRawDataConversion/")
sys.path.insert(0, "./Code/MeasureCreator/")
sys.path.insert(0, "../../OfficialGitHubs/GraVisGUI-1.2/SourceCode/")

from __mainCreateMeasures__ import createRegularityMeasurements, createResultMeasureTable
from FolderContent import FolderContent
from MGXContourFromPlyFileReader import MGXContourFromPlyFileReader
from MultiFolderContent import MultiFolderContent
from OtherMeasuresCreator import OtherMeasuresCreator
from pathlib import Path
from ShapeGUI import VisGraph
from skspatial.objects import Plane, Points

geometricTableBaseName="_geometricData.csv"
plyContourNameExtension="_outlines.ply"
plyJunctionNameExtension="_only junctions.ply"
polygonGeometricTableBaseName="_geometricData poly.csv"

contourNameExtension = "_cellContour.json"
orderedJunctionsNameExtension = "_orderedJunctionsPerCell.json"

contoursFilenameKey = "cellContours"
orderedJunctionsPerCellFilenameKey = "orderedJunctionsPerCellFilename"
rotatedAndProjectedContoursKey = "rotatedAndProjectedContours"
rotatedAndProjectedOrderedJunctionsKey = "rotatedAndProjectedOrderedJunctions"

levelOfFeedback: int = 1

def extractPathInformations(path: Path, filenameExtensionToIgnoreInPath: str = "", startIndexOffset: int = 0):
    tissueName = path.parts[-1+startIndexOffset].replace(filenameExtensionToIgnoreInPath, "")
    geneName = path.parts[-2+startIndexOffset]
    projectName = path.parts[-3+startIndexOffset]
    return {"projectName": projectName, "genotype": geneName, "replicateId": tissueName, "timePoint": 0}

def extractJsonFormattedContourFrom(plyFilenamesAsPaths: list, baseResultsFolder: str = "",
                                    filenameExtensionToIgnore: str = "", resultNameExtension: str = "_contour.json"):
    for filename in plyFilenamesAsPaths:
        pathInformation = extractPathInformations(filename, filenameExtensionToIgnore)
        tissueName = pathInformation["replicateId"]
        pathsToJoin = [pathInformation["projectName"], pathInformation["genotype"], tissueName, tissueName + resultNameExtension]
        filenameToSave = Path(baseResultsFolder).joinpath(*pathsToJoin)
        filenameToSave.parent.mkdir(parents=True, exist_ok=True)
        contourReader = MGXContourFromPlyFileReader(filename, extract3DContours=True)
        contourReader.SaveCellsContoursPositions(filenameToSave=filenameToSave)

def bundleProjectInformation(filenames, baseResultsFolder: str = "", keyToUseForFilename: str = "key", filenameExtensionToIgnore: str = "", multiFolderContentExtension: str = ".pkl"):
    allMultiFolderContentFilenames = []
    for filename in filenames:
        pathInformation = extractPathInformations(filename, filenameExtensionToIgnore, startIndexOffset=-1)
        projectName = pathInformation["projectName"]
        currentMultiFolderContentFilename = Path(baseResultsFolder).joinpath(projectName, projectName+multiFolderContentExtension)
        currentMultiFolderContent = MultiFolderContent(currentMultiFolderContentFilename)
        allMultiFolderContentFilenames.append(currentMultiFolderContentFilename)
        currentTissueContent = currentMultiFolderContent.GetTissueWithData(pathInformation)
        if currentTissueContent is None:
            currentTissueContent = FolderContent(pathInformation)
            currentMultiFolderContent.AppendFolderContent(currentTissueContent)
        currentTissueContent.AddDataToFilenameDict(filename, keyToUseForFilename)
        currentMultiFolderContent.UpdateFolderContents()
    return np.unique(allMultiFolderContentFilenames)

def mainInitializeMGXData(dataBaseFolder: str = "Images/YangData/", baseResultsFolder: str = "Results/Yang Data/", extractInfoFromRawData: bool = True):
    if extractInfoFromRawData:
        extractContourAndJunctionsFromMGX(dataBaseFolder)
    extractedOutlineFilenames = list(Path(baseResultsFolder).glob("**/*"+contourNameExtension))
    extractedJunctionFilenames = list(Path(baseResultsFolder).glob("**/*"+orderedJunctionsNameExtension))
    outlineMultiFolderContentsFilenames = bundleProjectInformation(extractedOutlineFilenames, baseResultsFolder=baseResultsFolder, keyToUseForFilename=contoursFilenameKey)
    junctionMultiFolderContentsFilenames = bundleProjectInformation(extractedJunctionFilenames, baseResultsFolder=baseResultsFolder, keyToUseForFilename=orderedJunctionsPerCellFilenameKey)
    uniqueProjectFolderContentsFilenames = np.unique(np.concatenate([outlineMultiFolderContentsFilenames, junctionMultiFolderContentsFilenames]))
    for projectsFolderContentsFilename in uniqueProjectFolderContentsFilenames:
        dataFolder = dataBaseFolder+Path(projectsFolderContentsFilename).stem+"/"
        createRegularityMeasurements(projectsFolderContentsFilename, dataFolder,
                                     checkCellsPresentInLabelledImage=False)
        createResultMeasureTable(projectsFolderContentsFilename, baseResultsFolder, loadMeasuresFromFilenameUsingKeys=["regularityMeasuresFilename"], includeCellId=False)
    return uniqueProjectFolderContentsFilenames

def extractContourAndJunctionsFromMGX(dataBaseFolder: str):
    outlineFilenames = list(Path(dataBaseFolder).glob("**/*" + plyContourNameExtension))
    junctionFilenames = list(Path(dataBaseFolder).glob("**/*" + plyJunctionNameExtension))
    extractJsonFormattedContourFrom(outlineFilenames, baseResultsFolder=baseResultsFolder, filenameExtensionToIgnore=plyContourNameExtension, resultNameExtension=contourNameExtension)
    extractJsonFormattedContourFrom(junctionFilenames, baseResultsFolder=baseResultsFolder, filenameExtensionToIgnore=plyJunctionNameExtension, resultNameExtension=orderedJunctionsNameExtension)
    if levelOfFeedback > 0:
        print(f"Saved {len(outlineFilenames)} outlines and {len(junctionFilenames)} junction filenames as .json-files to {baseResultsFolder}")

def create2DContourProjections(projectFolderContentsFilenames: list, baseResultsFolder: str = "Results/Yang Data/",
                               contoursFilenameKey: str = "cellContours", rotatedAndProjectedContoursKey: str = "rotatedAndProjectedContours",
                               projectJunctions: bool = True, orderedJunctionsPerCellFilenameKey: str = "orderedJunctionsPerCellFilename",
                               rotatedAndProjectedOrderedJunctionsKey: str = "rotatedAndProjectedOrderedJunctions",
                               summarisePointDistances: bool = True, summaryNameSuffix: str = "pointDistanceSummary.csv"):
    allProjectNames, allReplicateNames, allMeanPointDistances, allStdPointDistances, allCellIds = [], [], [], [], []
    for projectFolderContentsFilename in projectFolderContentsFilenames:
        projectName = Path(projectFolderContentsFilename).stem
        projectFolderContents = MultiFolderContent(projectFolderContentsFilename)
        for tissueContents in projectFolderContents:
            replicateName = tissueContents.GetReplicateId()
            genotype = tissueContents.GetGenotype()
            contours = tissueContents.LoadKeyUsingFilenameDict(contoursFilenameKey)
            if projectJunctions:
                orderedJunctions = tissueContents.LoadKeyUsingFilenameDict(orderedJunctionsPerCellFilenameKey)
            rotatedAndProjectedContoursOfCells, rotatedAndProjectedOrderedJunctionsOfCells = {}, {}
            for cellId, cellContour in contours.items():
                rotatedProjectedPoints, rotationMatrix = projectCellInto2DPlane(cellContour, returnRotationMatrix=True)
                rotatedProjectedPoints = rotatedProjectedPoints[:, :2]
                rotatedAndProjectedContoursOfCells[cellId] = rotatedProjectedPoints
                if projectJunctions:
                    if cellId in orderedJunctions:
                        junctions = np.asarray(orderedJunctions[cellId])
                        junctions -= np.mean(cellContour, axis=0)
                        assert junctions.shape[1] == 3, f"The junctions to be converted into 2D need to have three columns, but the shape is {junctions.shape} for {cellId=} of {tissueContents.GetTissueName()}"
                        rotatedProjectedPoints = rotationMatrix.dot(junctions.T).T
                        rotatedAndProjectedOrderedJunctionsOfCells[cellId] = rotatedProjectedPoints[:, :2]
                if summarisePointDistances:
                    shiftedPoints = np.concatenate([rotatedProjectedPoints[1:, :], [rotatedProjectedPoints[0, :]]], axis=0)
                    distancesBetweenPoints = np.linalg.norm(rotatedProjectedPoints - shiftedPoints, axis=1)
                    allMeanPointDistances.append(np.mean(distancesBetweenPoints))
                    allStdPointDistances.append(np.std(distancesBetweenPoints))
                    allCellIds.append(cellId)
                    allReplicateNames.append(replicateName)
                    allProjectNames.append(projectName)
            rotatedAndProjectedContoursFilename = Path(baseResultsFolder).joinpath(projectName, genotype, replicateName, replicateName + "_" + rotatedAndProjectedContoursKey + ".json")
            rotatedAndProjectedContoursFilename.parent.mkdir(parents=True, exist_ok=True)
            tissueContents.SaveDataFilesTo(rotatedAndProjectedContoursOfCells, rotatedAndProjectedContoursFilename)
            tissueContents.AddDataToFilenameDict(rotatedAndProjectedContoursFilename, rotatedAndProjectedContoursKey)
            if projectJunctions:
                rotatedAndProjectedOrderedJunctionsFilename = rotatedAndProjectedContoursFilename.with_name(replicateName + "_" + rotatedAndProjectedOrderedJunctionsKey + ".json")
                tissueContents.SaveDataFilesTo(rotatedAndProjectedOrderedJunctionsOfCells, rotatedAndProjectedOrderedJunctionsFilename)
                tissueContents.AddDataToFilenameDict(rotatedAndProjectedOrderedJunctionsFilename, rotatedAndProjectedOrderedJunctionsKey)
        projectFolderContents.UpdateFolderContents()
    if summarisePointDistances:
        pointDistanceSummary = {"projectName": allProjectNames, "replicateName": allReplicateNames, "cellId": allCellIds,
                                "meanPointDistance": allMeanPointDistances, "stdPointDistance": allStdPointDistances}
        pointDistanceSummary = pd.DataFrame(pointDistanceSummary)
        pointDistanceSummaryFilename = baseResultsFolder + summaryNameSuffix
        pointDistanceSummary.to_csv(pointDistanceSummaryFilename, index=False)

def projectCellInto2DPlane(pointArry: np.ndarray, returnRotationMatrix: bool = True):
    pointArry -= np.mean(pointArry, axis=0)
    points = Points(pointArry)
    plane = Plane.best_fit(points)
    projectedPoints = Points([plane.project_point(p) for p in points])
    mat = rotation_matrix_from_vectors(plane.normal, np.array([0, 0, 1]))
    rotatedProjectedPoints = mat.dot(projectedPoints.T).T
    if returnRotationMatrix:
        return rotatedProjectedPoints, mat
    else:
        return rotatedProjectedPoints

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    From Peter https://stackoverflow.com/questions/45142959/calculate-rotation-matrix-to-align-two-vectors-in-3d-space
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def visualiseOriginalVsProjectedContour(contour, rotatedAndProjectedContour):
    import matplotlib.pyplot as plt
    from skspatial.plotting import plot_3d
    points = Points(contour)
    flatPlain = Plane.from_points([0, 0, 0], [1, 0, 0], [0, 1, 0])
    rotatedProjectedPoints = Points(rotatedAndProjectedContour)
    fig, ax = plot_3d(
        points.plotter(alpha=0.2, c='k', s=50, depthshade=False),
        flatPlain.plotter(alpha=0.2, lims_x=(-5, 5), lims_y=(-5, 5), color="blue"),
        rotatedProjectedPoints.plotter(alpha=0.2, c='green', s=50, depthshade=False)
    )
    ax.set_zlim((5, -5))
    plt.show()

def analysePointDistances(baseResultsFolder: str = "Results/Yang Data/"):
    pointDistanceSummaryFilename = baseResultsFolder + "pointDistanceSummary.csv"
    pointDistanceSummary = pd.read_csv(pointDistanceSummaryFilename)
    for i, subTable in pointDistanceSummary.groupby(["projectName", "replicateName"], sort=False):
        print(i, subTable["meanPointDistance"].mean(), subTable["meanPointDistance"].std(), subTable["stdPointDistance"].mean())

def regularityAnalysisOnFlattened(uniqueProjectFolderContentsFilenames, dataBaseFolder: str = "", orderedJunctionsPerCellFilename:str = "rotatedAndProjectedOrderedJunctions",
                                  regularityMeasuresFilenameKey: str = "flattenedRegularityMeasuresFilename", tableBaseName: str = "combinedMeasures_regularity2D_{}.csv"):
    for projectsFolderContentsFilename in uniqueProjectFolderContentsFilenames:
        dataFolder = dataBaseFolder+Path(projectsFolderContentsFilename).stem+"/"
        createRegularityMeasurements(projectsFolderContentsFilename, dataFolder,
                                     regularityMeasuresFilenameKey=regularityMeasuresFilenameKey,
                                     orderedJunctionsPerCellFilename=orderedJunctionsPerCellFilename,
                                     regularityMeasuresBaseName="regularityMeasures2D.json",
                                     checkCellsPresentInLabelledImage=False)
        createResultMeasureTable(projectsFolderContentsFilename, baseResultsFolder, loadMeasuresFromFilenameUsingKeys=[regularityMeasuresFilenameKey],
                                 tableBaseName=tableBaseName, includeCellId=False)

def runVisibilityAnalysisWithContours(projectFolderContentsFilename: str or Path, baseResultsFolder: str = "Results/Yang Data/", reductionFactor: int = 8, printVisGraphCalculationTime: bool = False,
                                          tryToLoad: bool = True, visibilityGraphMatrices: str = "visibilityGraphMatrices", relCompletenessOfCellsKey: str = "relCompletenessOfCells"):
    from shapely import Polygon
    if reductionFactor != 8:
        visibilityGraphMatrices = visibilityGraphMatrices + "_" + str(reductionFactor)
        relCompletenessOfCellsKey = relCompletenessOfCellsKey + "_" + str(reductionFactor)
    measureCreator = OtherMeasuresCreator()
    projectName = Path(projectFolderContentsFilename).stem
    projectFolderContents = MultiFolderContent(projectFolderContentsFilename)
    for tissueContents in projectFolderContents:
        replicateName = tissueContents.GetReplicateId()
        genotype = tissueContents.GetGenotype()
        if tryToLoad and tissueContents.IsKeyInFilenameDict(visibilityGraphMatrices):
            visibilityGraphMatrixOfCells = tissueContents.LoadKeyUsingFilenameDict(visibilityGraphMatrices)
            visibilityGraphsOfCells = {cellId: nx.from_numpy_array(np.asarray(matrix)) for cellId, matrix in visibilityGraphMatrixOfCells.items()}
        else:
            rotatedProjectedPointsOfCells = tissueContents.LoadKeyUsingFilenameDict(rotatedAndProjectedContoursKey)
            visibilityGraphsOfCells, visibilityGraphMatrixOfCells = {}, {}
            for cellId, rotatedProjectedPoints in rotatedProjectedPointsOfCells.items():
                numberOfEquallySpacedPoints = int(len(rotatedProjectedPoints) / reductionFactor)
                positionsAsShapelyGeometry = measureCreator.equallySpacePointsBetween(rotatedProjectedPoints, numberOfEquallySpacedPoints=numberOfEquallySpacedPoints)
                invalidPolygon = not Polygon(positionsAsShapelyGeometry).is_valid
                invalidLinearRing = not positionsAsShapelyGeometry.is_valid
                if invalidPolygon or invalidLinearRing:
                    warnings.warn(f"{cellId} has crossing outline with {invalidPolygon=} {invalidLinearRing=} of {replicateName=} {genotype=} {projectFolderContentsFilename=}")
                else:
                    if printVisGraphCalculationTime:
                        startTime = time.time()
                        print(f"{cellId=} number of points {numberOfEquallySpacedPoints} stating time: {time.strftime('%H:%M', time.localtime(startTime))}")
                    visibilityGraphs = measureCreator.calcVisbilityGraphFromGeometery(positionsAsShapelyGeometry)
                    if printVisGraphCalculationTime:
                        print(f"Finished in {np.round((time.time()-startTime)/60, 2)} min.")
                    visibilityGraphsOfCells[cellId] = visibilityGraphs
                    visibilityGraphMatrixOfCells[cellId] = nx.to_numpy_array(visibilityGraphs)
            visibilityGraphMatricesFilename = Path(baseResultsFolder).joinpath(projectName, genotype, replicateName, replicateName + "_" + visibilityGraphMatrices + ".json")
            tissueContents.SaveDataFilesTo(visibilityGraphMatrixOfCells, visibilityGraphMatricesFilename)
            tissueContents.AddDataToFilenameDict(visibilityGraphMatricesFilename, visibilityGraphMatrices)
        relCompletenessOfCells = calcRelativeCompletenessFor(visibilityGraphsOfCells)
        relCompletenessOfCellsFilename = Path(baseResultsFolder).joinpath(projectName, genotype, replicateName, replicateName + "_" + relCompletenessOfCellsKey + ".json")
        tissueContents.SaveDataFilesTo(relCompletenessOfCells, relCompletenessOfCellsFilename)
        tissueContents.AddDataToFilenameDict(relCompletenessOfCellsFilename, relCompletenessOfCellsKey)
        projectFolderContents.UpdateFolderContents()
        print("finished", tissueContents.GetTissueName())

def createResultsTableForShallowFiles(projectFolderContentsFilename: str or Path, filenameKeyToLoad: str,
                                      baseResultsFolder: str = "", tableBaseName: str = "combinedMeasures_{}.csv"):
    multiFolderContent = MultiFolderContent(projectFolderContentsFilename)
    multiFolderContent.AddDataFromSimpleDictsUsingFilenameKey(filenameKeyToLoad)
    measureResultsTidyDf = multiFolderContent.GetTidyDataFrameOf([filenameKeyToLoad], includeCellId=False)
    scenarioName = Path(projectFolderContentsFilename).stem
    measureTableName = baseResultsFolder + tableBaseName.format(scenarioName)
    Path(measureTableName).parent.mkdir(parents=True, exist_ok=True)
    measureResultsTidyDf.to_csv(measureTableName, index=False)

def calcRelativeCompletenessFor(visibilityGraphsOfCells: dict):
    relCompletenessOfCells = {}
    for cellId, visibilityGraph in visibilityGraphsOfCells.items():
        relativeCompleteness = VisGraph.compute_graph_complexity(None, visibilityGraph)
        relCompletenessOfCells[cellId] = relativeCompleteness
    return relCompletenessOfCells

if __name__ == '__main__':
    baseResultsFolder: str = "Results/Yang Data/"
    projectFolderContentsFilenames = [baseResultsFolder+"Col-0 and act2-3 act7-6 PI staining/Col-0 and act2-3 act7-6 PI staining.pkl",
                                      baseResultsFolder+"Ws and act2-1 act7-1 PI staining/Ws and act2-1 act7-1 PI staining.pkl",
                                      baseResultsFolder+"Ws and act2-1 act7-1 PM reporter/Ws and act2-1 act7-1 PM reporter.pkl",
                                      ]
    # convert MGX .ply-files into human-readable contour and ordered junction .json-files and
    # calculated regularity analysis of tri-way junction spanning polygons
    # projectFolderContentsFilenames = mainInitializeMGXData()
    # # create and calculated regularity analysis of flattened (2D not 2.5D) tri-way junction spanning polygons
    # create2DContourProjections(projectFolderContentsFilenames, baseResultsFolder, contoursFilenameKey=contoursFilenameKey,
    #                            rotatedAndProjectedContoursKey=rotatedAndProjectedContoursKey,
    #                            orderedJunctionsPerCellFilenameKey=orderedJunctionsPerCellFilenameKey,
    #                            rotatedAndProjectedOrderedJunctionsKey=rotatedAndProjectedOrderedJunctionsKey)
    # regularityAnalysisOnFlattened(projectFolderContentsFilenames, baseResultsFolder)
    # create relative completeness on flattened (2D not 2.5D) contours
    reductionFactor = 8
    for projectFolderContentsFilename in projectFolderContentsFilenames:
        runVisibilityAnalysisWithContours(projectFolderContentsFilename, baseResultsFolder, reductionFactor=reductionFactor)
        if reductionFactor != 8:
            filenameKeyToLoad = "relCompletenessOfCells" + "_" + str(reductionFactor)
            tableBaseName = "combinedMeasures_relCompleteness"+str(reductionFactor)+"_{}.csv"
        else:
            filenameKeyToLoad = "relCompletenessOfCells"
            tableBaseName = "combinedMeasures_relCompleteness_{}.csv"
        createResultsTableForShallowFiles(projectFolderContentsFilename, filenameKeyToLoad=filenameKeyToLoad,
                                          baseResultsFolder=baseResultsFolder, tableBaseName=tableBaseName)
