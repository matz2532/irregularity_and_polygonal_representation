import sys

sys.path.insert(0, "./Code/DataStructures/")
sys.path.insert(0, "./Code/ImageToRawDataConversion/")

from FolderContent import FolderContent
from MGXContourFromPlyFileReader import MGXContourFromPlyFileReader
from MultiFolderContent import MultiFolderContent
from pathlib import Path

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
    for filename in filenames:
        pathInformation = extractPathInformations(filename, filenameExtensionToIgnore)
        projectName = pathInformation["projectName"]
        currentMultiFolderContentFilename = Path(baseResultsFolder).joinpath(projectName, projectName+multiFolderContentExtension)
        currentMultiFolderContent = MultiFolderContent(currentMultiFolderContentFilename)
        currentTissueContent = currentMultiFolderContent.GetTissueWithData(pathInformation)
        if currentTissueContent is None:
            currentTissueContent = FolderContent(pathInformation)
            currentMultiFolderContent.AppendFolderContent(currentTissueContent)
        currentTissueContent.AddDataToFilenameDict(filename, keyToUseForFilename)
        currentMultiFolderContent.UpdateFolderContents()

def mainInitializeMGXData(dataBaseFolder: str = "Images/YangData/", baseResultsFolder: str = "Results/Yang Data/", overwrite: bool = True,
                          projectFolderDepth: int = 3):
    outlineFilenames = list(Path(dataBaseFolder).glob("**/*"+plyContourNameExtension))
    junctionFilenames = list(Path(dataBaseFolder).glob("**/*"+plyJunctionNameExtension))
    extractJsonFormattedContourFrom(outlineFilenames, baseResultsFolder=baseResultsFolder, filenameExtensionToIgnore=plyContourNameExtension, resultNameExtension=contourNameExtension)
    extractJsonFormattedContourFrom(junctionFilenames, baseResultsFolder=baseResultsFolder, filenameExtensionToIgnore=plyJunctionNameExtension, resultNameExtension=orderedJunctionsNameExtension)
    if levelOfFeedback > 0:
        print(f"Saved {len(outlineFilenames)} outlines and {len(junctionFilenames)} junction filenames as .json-files to {baseResultsFolder}")
    bundleProjectInformation(outlineFilenames, baseResultsFolder=baseResultsFolder, keyToUseForFilename=contoursFilenameKey, filenameExtensionToIgnore=plyContourNameExtension)
    bundleProjectInformation(junctionFilenames, baseResultsFolder=baseResultsFolder, keyToUseForFilename=orderedJunctionsPerCellFilenameKey, filenameExtensionToIgnore=plyJunctionNameExtension)

if __name__ == '__main__':
    mainInitializeMGXData()