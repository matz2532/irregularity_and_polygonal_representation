import numpy as np
import sys

sys.path.insert(0, "./Code/DataStructures/")
sys.path.insert(0, "./Code/ImageToRawDataConversion/")

from __mainCreateMeasures__ import createRegularityMeasurements, createResultMeasureTable
from FolderContent import FolderContent
from MGXContourFromPlyFileReader import MGXContourFromPlyFileReader
from MultiFolderContent import MultiFolderContent
from pathlib import Path
from skspatial.objects import Plane, Points

geometricTableBaseName="_geometricData.csv"
plyContourNameExtension="_outlines.ply"
plyJunctionNameExtension="_only junctions.ply"
polygonGeometricTableBaseName="_geometricData poly.csv"

contourNameExtension = "_cellContour.json"
orderedJunctionsNameExtension = "_orderedJunctionsPerCell.json"

contoursFilenameKey = "cellContours"
orderedJunctionsPerCellFilenameKey = "orderedJunctionsPerCellFilename"

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

def mainInitializeMGXData(dataBaseFolder: str = "Images/YangData/", baseResultsFolder: str = "Results/Yang Data/", overwrite: bool = True,
                          projectFolderDepth: int = 3):
    outlineFilenames = list(Path(dataBaseFolder).glob("**/*"+plyContourNameExtension))
    junctionFilenames = list(Path(dataBaseFolder).glob("**/*"+plyJunctionNameExtension))
    extractJsonFormattedContourFrom(outlineFilenames, baseResultsFolder=baseResultsFolder, filenameExtensionToIgnore=plyContourNameExtension, resultNameExtension=contourNameExtension)
    extractJsonFormattedContourFrom(junctionFilenames, baseResultsFolder=baseResultsFolder, filenameExtensionToIgnore=plyJunctionNameExtension, resultNameExtension=orderedJunctionsNameExtension)
    if levelOfFeedback > 0:
        print(f"Saved {len(outlineFilenames)} outlines and {len(junctionFilenames)} junction filenames as .json-files to {baseResultsFolder}")
    extractedOutlineFilenames = list(Path(baseResultsFolder).glob("**/*"+contourNameExtension))
    extractedJunctionFilenames = list(Path(baseResultsFolder).glob("**/*"+orderedJunctionsNameExtension))
    outlineMultiFolderContentsFilenames = bundleProjectInformation(extractedOutlineFilenames, baseResultsFolder=baseResultsFolder, keyToUseForFilename=contoursFilenameKey)
    junctionMultiFolderContentsFilenames = bundleProjectInformation(extractedJunctionFilenames, baseResultsFolder=baseResultsFolder, keyToUseForFilename=orderedJunctionsPerCellFilenameKey)
    uniqueProjectFolderContentsFilenames = np.unique(np.concatenate([outlineMultiFolderContentsFilenames, junctionMultiFolderContentsFilenames]))
    for projectsFolderContentsFilename in uniqueProjectFolderContentsFilenames:
        createRegularityMeasurements(projectsFolderContentsFilename, dataBaseFolder,
                                     checkCellsPresentInLabelledImage=False)
        createResultMeasureTable(projectsFolderContentsFilename, baseResultsFolder, loadMeasuresFromFilenameUsingKeys=["regularityMeasuresFilename"], includeCellId=False)
    return uniqueProjectFolderContentsFilenames

def mainRunVisibilityAnalysisWithContours(projectFolderContentsFilenames: list, baseResultsFolder: str = "Results/Yang Data/"):
    for projectFolderContentsFilename in projectFolderContentsFilenames:
        projectFolderContents = MultiFolderContent(projectFolderContentsFilename)
        for tissueContents in projectFolderContents:
            contours = tissueContents.LoadKeyUsingFilenameDict(contoursFilenameKey)
            for c in contours.values():
                rotatedProjectedPoints = projectCellInto2DPlane(c)
                rotatedProjectedPoints = rotatedProjectedPoints[:, :2]
                visualiseOriginalVsProjectedContour(c, rotatedProjectedPoints)

def projectCellInto2DPlane(pointArry: np.ndarray):
    pointArry -= np.mean(pointArry, axis=0)
    points = Points(pointArry)
    plane = Plane.best_fit(points)
    projectedPoints = Points([plane.project_point(p) for p in points])
    mat = rotation_matrix_from_vectors(plane.normal, np.array([0, 0, 1]))
    rotatedProjectedPoints = mat.dot(projectedPoints.T).T
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

if __name__ == '__main__':
    # projectFolderContentsFilenames = mainInitializeMGXData()
    baseResultsFolder: str = "Results/Yang Data/"
    projectFolderContentsFilenames = [baseResultsFolder+"Col-0 and act2-3 act7-6 PI staining/Col-0 and act2-3 act7-6 PI staining.pkl",
                                      baseResultsFolder+"Ws and act2-1 act7-1 PI staining/Ws and act2-1 act7-1 PI staining.pkl",
                                      baseResultsFolder+"Ws and act2-1 act7-1 PM reporter/Ws and act2-1 act7-1 PM reporter.pkl",
                                      ]
    mainRunVisibilityAnalysisWithContours(projectFolderContentsFilenames, baseResultsFolder)
