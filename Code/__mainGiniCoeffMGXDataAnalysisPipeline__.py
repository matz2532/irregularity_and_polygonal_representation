import sys

sys.path.insert(0, "./Code/ImageToRawDataConversion/")

from MGXContourFromPlyFileReader import MGXContourFromPlyFileReader
from pathlib import Path

geometricTableBaseName="_geometricData.csv"
plyContourNameExtension="_outlines.ply"
plyJunctionNameExtension="_only junctions.ply"
polygonGeometricTableBaseName="_geometricData poly.csv"

contourNameExtension = "_cellContour.json"
orderedJunctionsNameExtension = "_orderedJunctionsPerCell.json"

levelOfFeedback: int = 1

def extractJsonFormattedContourFrom(plyFilenamesAsPaths: list, baseResultsFolder: str = "",
                                    filenameExtensionToIgnore: str = "", resultNameExtension: str = "_contour.json"):
    for filename in plyFilenamesAsPaths:
        tissueName = filename.parts[-1].replace(filenameExtensionToIgnore, "")
        geneName = filename.parts[-2]
        projectName = filename.parts[-3]
        pathsToJoin = [projectName, geneName, tissueName, tissueName + resultNameExtension]
        filenameToSave = Path(baseResultsFolder).joinpath(*pathsToJoin)
        filenameToSave.parent.mkdir(parents=True, exist_ok=True)
        contourReader = MGXContourFromPlyFileReader(filename, extract3DContours=True)
        contourReader.SaveCellsContoursPositions(filenameToSave=filenameToSave)

def mainInitializeMGXData(dataBaseFolder: str = "Images/YangData/", baseResultsFolder: str = "Results/Yang Data/", overwrite: bool = True):
    outlineFilenames = list(Path(dataBaseFolder).glob("**/*"+plyContourNameExtension))
    junctionFilenames = list(Path(dataBaseFolder).glob("**/*"+plyJunctionNameExtension))
    extractJsonFormattedContourFrom(outlineFilenames, baseResultsFolder=baseResultsFolder, filenameExtensionToIgnore=plyContourNameExtension, resultNameExtension=contourNameExtension)
    extractJsonFormattedContourFrom(junctionFilenames, baseResultsFolder=baseResultsFolder, filenameExtensionToIgnore=plyJunctionNameExtension, resultNameExtension=orderedJunctionsNameExtension)
    if levelOfFeedback > 0:
        print(f"Saved {len(outlineFilenames)} outlines and {len(junctionFilenames)} junction filenames as .json-files to {baseResultsFolder}")

if __name__ == '__main__':
    mainInitializeMGXData()