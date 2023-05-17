import numpy as np
import pandas as pd
import pickle
import sys

sys.path.insert(0, "./Code/DataStructures/")
sys.path.insert(0, "./Code/ImageToRawDataConversion/")
sys.path.insert(0, "./Code/MeasureCreator/")
sys.path.insert(0, "./Code/SpatialAnalysis/")
sys.path.insert(0, "./Images/")

from AreaMeasureExtractor import AreaMeasureExtractor, combineNeighborDistanceTables
from checkFolderContentsJunctionPositioning import checkFolderContentsJunctionPositioning, extractAndSaveContourAndBaseJunctions
from FolderContent import FolderContent
from LabelledImageToGraphConverter import LabelledImageToGraphConverter
from MGXToFolderContentConverter import mainCreateAllFullCotyledons, mainInitalizeSAMDataAddingContoursAndJunctions, createGeometricdataTableWithRemovedStomata, addGeometricDataFilenameAndKey
from MultiFolderContent import MultiFolderContent
from OtherMeasuresCreator import OtherMeasuresCreator
from pathlib import Path
from PerTissuePolygonRegularityCalculator import PerTissuePolygonRegularityCalculator
from PointOrdererAlongOutline import saveOrderedJunctionsOf
from PreProcessorDummy import PreProcessorDummy

globalVerbosity = 3

def createFolderContentsOfAllTissues(allFolderContentsFilename, inputData, genotypesResolutionDict, dataBaseFolder=None,
                                     addCorrectedJunctionPositionsFilename=True, addGuardCellJunctionPositionsFilename=True, 
                                     overwrite=True):
    if overwrite:
        allFolderContents = MultiFolderContent()
        allFolderContents.SetAllFolderContentsFilename(allFolderContentsFilename)
    else:
        allFolderContents = MultiFolderContent(allFolderContentsFilename)
    contourReader = PreProcessorDummy()
    for genotype, genotypesInputData in inputData.items():
        for replicateId, replicatesInputData in genotypesInputData.items():
            if dataBaseFolder is None:
                replicatesBaseFolder = replicatesInputData["folder"]
            else:
                replicatesBaseFolder = dataBaseFolder + genotype + "/" + replicateId + "/"
            resolution = genotypesResolutionDict[genotype]
            allTimePoints = replicatesInputData["timePoints"]
            labelledImageBaseNames = replicatesInputData["labelledImageFilename"]
            originalImageBaseNames = replicatesInputData["originalImageFilename"]
            contourBaseNames = replicatesInputData["contourFilename"]
            for i, timePointName in enumerate(allTimePoints):
                tissueBaseFolder = replicatesBaseFolder + timePointName + "/"
                contourFilename = tissueBaseFolder + contourBaseNames[i]
                filenameDict = {"originalImageFilename": tissueBaseFolder + originalImageBaseNames[i],
                                "contourFilename": contourFilename,
                                "labelledImageFilename": tissueBaseFolder + labelledImageBaseNames[i]}
                folderContent = {"genotype": genotype,
                                 "replicateId": replicateId,
                                 "timePoint": timePointName,
                                 "resolution": resolution,
                                 "filenameDict": filenameDict,
                                 "extractedFilesDict": {}}
                folderContent = FolderContent(folderContent)
                if addCorrectedJunctionPositionsFilename:
                    correctedJunctionFilename = tissueBaseFolder + "correctedTriWayJunctions.npy"
                    folderContent.AddDataToFilenameDict(correctedJunctionFilename, "finalJunctionFilename")
                if addGuardCellJunctionPositionsFilename:
                    correctedJunctionFilename = tissueBaseFolder + "guardCellJunctionPositions.npy"
                    folderContent.AddDataToFilenameDict(correctedJunctionFilename, "guardCellJunctionPositions")
                isTissuePresent = allFolderContents.IsTissuePresent(replicateId, timePointName)
                if not isTissuePresent:
                    allFolderContents.AppendFolderContent(folderContent)
                    allFolderContents.UpdateFolderContents()
                    if globalVerbosity >= 3:
                        print(f"The {replicateId=}, {timePointName=} was added to {allFolderContentsFilename=}")
                else:
                    if globalVerbosity >= 4:
                        print(f"The {replicateId=}, {timePointName=} was not added as it is already present in {allFolderContentsFilename=}")

def createAllEasyReadableContoursAndSimpleTriWayJunctions(dataBaseFolder, allFolderContentsFilename):
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    for folderContent in multiFolderContent:
        extractAndSaveContourAndBaseJunctions(folderContent, dataBaseFolder, globalVerbosity=globalVerbosity)
        multiFolderContent.UpdateFolderContents()
        if globalVerbosity >= 2:
            print("Updating multiFolderContent saved under {}".format(allFolderContentsFilename))

def createAllAdjacencyListsFromLabelledImage(dataBaseFolder, allFolderContentsFilename):
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    for folderContent in multiFolderContent:
        contourDict = folderContent.LoadKeyUsingFilenameDict("cellContours", convertDictKeysToInt=True, convertDictValuesToNpArray=True)
        selectedCellLabels = np.sort(list(contourDict.keys()))
        myLabelledImageToGraphConverter = LabelledImageToGraphConverter(folderContent=folderContent, selectedCellIds=selectedCellLabels)
        myLabelledImageToGraphConverter.SaveAdjacencyList(dataBaseFolder=dataBaseFolder)
        multiFolderContent.UpdateFolderContents()
        if globalVerbosity >= 2:
            print("Updating multiFolderContent saved under {}".format(allFolderContentsFilename))

def checkTriWayJunctionPositioning(dataBaseFolder, allFolderContentsFilename, redoTriWayJunctionPositioning=True):
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    for folderContent in multiFolderContent:
        if not folderContent.IsKeyInFilenameDict("finalJunctionFilename") or redoTriWayJunctionPositioning:
            checkFolderContentsJunctionPositioning(folderContent, dataBaseFolder, globalVerbosity=globalVerbosity)
            multiFolderContent.UpdateFolderContents()
            if globalVerbosity >= 2:
                print("Updating multiFolderContent saved under {}".format(allFolderContentsFilename))

def orderTriWayJunctionsOnContour(dataBaseFolder, allFolderContentsFilename):
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    for folderContent in multiFolderContent:
        saveOrderedJunctionsOf(folderContent, dataBaseFolder)
        multiFolderContent.UpdateFolderContents()
        if globalVerbosity >= 2:
            print("Updating multiFolderContent saved under {}".format(allFolderContentsFilename))

def createRegularityMeasurements(allFolderContentsFilename, dataBaseFolder, ignoreGuardCells=False,
                      regularityMeasuresBaseName="regularityMeasures.pkl", regularityMeasuresFilenameKey="regularityMeasuresFilename",
                      ignoreGuardCellExtension="_ignoringGuardCells", genotypeResolutionDict=None, checkCellsPresentInLabelledImage=True):
    if globalVerbosity >= 2:
        print(f"Run regularity analysis.")
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    regularityCalculator = PerTissuePolygonRegularityCalculator(None, None, None)
    if ignoreGuardCells:
        regularityMeasuresFilenameKey += ignoreGuardCellExtension
        regularityMeasuresBaseName = regularityMeasuresBaseName.replace(".pkl", ignoreGuardCellExtension + ".pkl")
    for folderContent in multiFolderContent:
        if globalVerbosity >= 3:
            print(f"Run {folderContent.GetTissueName()} regularity analysis.")
        if checkCellsPresentInLabelledImage:
            labelledImage = folderContent.LoadKeyUsingFilenameDict("labelledImageFilename")
        allContours = folderContent.LoadKeyUsingFilenameDict("cellContours", convertDictKeysToInt=True, convertDictValuesToNpArray=True)
        orderedJunctionsPerCell = folderContent.LoadKeyUsingFilenameDict("orderedJunctionsPerCellFilename")
        genotypeName = folderContent.GetGenotype()
        resolution = 1
        if not genotypeResolutionDict is None:
            if genotypeName in genotypeResolutionDict:
                resolution = genotypeResolutionDict[genotypeName]
        dataFolder = dataBaseFolder + folderContent.GetFolder()
        regularityMeasuresFilename = dataFolder + regularityMeasuresBaseName
        Path(regularityMeasuresFilename).parent.mkdir(parents=True, exist_ok=True)
        if ignoreGuardCells:
            if not folderContent.IsKeyInFilenameDict("isSegmentNeighboringGuardCell") or not folderContent.IsKeyInFilenameDict("isAngleAtGuardCellJunction"):
                previousRegularityFilename = dataFolder + regularityMeasuresBaseName.replace(ignoreGuardCellExtension + ".pkl", ".pkl")
                print(f"guard cell segment neighbor and angle is missing, so just copied regularity measure results from {previousRegularityFilename} to name of {regularityMeasuresFilename}")
                previousRegularityFile = folderContent.LoadKeyUsingFilenameDict(regularityMeasuresFilenameKey.replace(ignoreGuardCellExtension, ""))
                copiedRegularityFileOfIgnoringGuardCells = {}
                for key, values in previousRegularityFile.items():
                    copiedRegularityFileOfIgnoringGuardCells[key + ignoreGuardCellExtension] = values
                pickle.dump(copiedRegularityFileOfIgnoringGuardCells, open(regularityMeasuresFilename, "wb"))
                folderContent.AddDataToFilenameDict(regularityMeasuresFilename, regularityMeasuresFilenameKey)
                multiFolderContent.UpdateFolderContents()
                continue
            isSegmentNeighboringGuardCell = folderContent.LoadKeyUsingFilenameDict("isSegmentNeighboringGuardCell")
            isAngleAtGuardCellJunction = folderContent.LoadKeyUsingFilenameDict("isAngleAtGuardCellJunction")
        else:
            isSegmentNeighboringGuardCell = None
            isAngleAtGuardCellJunction = None
        if checkCellsPresentInLabelledImage:
            regularityCalculator.SetCalculator(labelledImage, allContours, orderedJunctionsPerCell,
                                               resolution=resolution, ignoreSegmentOrAngleExtension=ignoreGuardCellExtension,
                                               ignoreSegmentsDict=isSegmentNeighboringGuardCell, ignoreAnglesDict=isAngleAtGuardCellJunction)
        else:
            regularityCalculator.SetCalculatorWithoutChecking(allContours, orderedJunctionsPerCell,
                                                              resolution=resolution, ignoreSegmentOrAngleExtension=ignoreGuardCellExtension,
                                                              ignoreSegmentsDict=isSegmentNeighboringGuardCell, ignoreAnglesDict=isAngleAtGuardCellJunction)
        regularityCalculator.CalcPolygonalComplexityForTissue()
        regularityMeasures = regularityCalculator.GetRegularityMeasuresDict()
        with open(regularityMeasuresFilename, "wb") as fh:
            pickle.dump(regularityMeasures, fh)
        folderContent.AddDataToFilenameDict(regularityMeasuresFilename, regularityMeasuresFilenameKey)
        multiFolderContent.UpdateFolderContents()

def createAreaAndDistanceMeasures(allFolderContentsFilename, dataBaseFolder,
                                areaMeasuresKey="areaMeasuresPerCell", timePointInPath=True,
                                useGeometricData=False):
    if globalVerbosity >= 2:
        print(f"Run area calculation.")
    if useGeometricData:
        allowedLabelFilename = "orderedJunctionsPerCellFilename"
    else:
        allowedLabelFilename = "cellContours"
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    areaExtractor = AreaMeasureExtractor(None)
    for folderContent in multiFolderContent:
        if globalVerbosity >= 3:
            print(f"Run {folderContent.GetTissueName()} area calculation.")
        scenarioName, replicateName, timePoint = folderContent.GetTissueInfos()
        if timePointInPath:
            replicateName = replicateName + "_" + timePoint
            folderExtension = folderContent.GetFolder()
        else:
            replicateName = replicateName
            folderExtension = scenarioName + "/" + replicateName + "/"
        baseResultsFilename = dataBaseFolder + folderExtension + replicateName + "_"
        saveAreaMeasuresAsFilename = baseResultsFilename + areaMeasuresKey + ".pkl"
        areaExtractor.SetFolderContent(folderContent)
        areaExtractor.RunAndSaveAllAreaMeasures(useGeometricData=useGeometricData, saveAreaMeasuresAsFilename=saveAreaMeasuresAsFilename)
        areaExtractor.RunAndSaveAllGeometricCentersAndDistances(useGeometricData=useGeometricData, baseResultsFilename=baseResultsFilename)
        multiFolderContent.UpdateFolderContents()

def createResultMeasureTable(allFolderContentsFilename, resultsFolder,
                             loadMeasuresFromFilenameUsingKeys=["regularityMeasuresFilename", "areaMeasuresPerCell"],
                             tableBaseName="combinedMeasures.csv", includeCellId=True,
                             ensureMeasuresPresenceOverAllTissues=False):
    if globalVerbosity >= 2:
        print(f"Combine measures into single table.")
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    allMeasureNames = []
    for filenameMeasuresKey in loadMeasuresFromFilenameUsingKeys:
        individualDataKeys = multiFolderContent.AddDataFromFilenameContainingMultipleDicts(filenameMeasuresKey, returnIndividualKeysAdded=True)
        if type(individualDataKeys[0]) == str:
            allDataKeys = individualDataKeys
        else:
            allDataKeys = np.concatenate(individualDataKeys)
        if ensureMeasuresPresenceOverAllTissues:
            key, count = np.unique(allDataKeys, return_count=True)
            areKeyCountsEqual = np.all(count == [0])
            if not areKeyCountsEqual:
                print(f"The measure names {key[count != [0]]} of {filenameMeasuresKey=} do not appear in all tissues.")
        allMeasureNames.append(pd.unique(allDataKeys))
    allMeasureNames = pd.unique(np.concatenate(allMeasureNames))
    measureResultsTidyDf = multiFolderContent.GetTidyDataFrameOf(allMeasureNames, includeCellId=includeCellId)
    measureResultsTidyDf[allMeasureNames] = measureResultsTidyDf[allMeasureNames].astype(float)
    measureTableName = resultsFolder + tableBaseName
    Path(measureTableName).parent.mkdir(parents=True, exist_ok=True)
    measureResultsTidyDf.to_csv(measureTableName, index=False)

def addRatioMeasuresToTable(resultsFolder,
                            ratioBetween=[["originalPolygonArea", "labelledImageArea"], ["regularPolygonArea", "labelledImageArea"], ["regularPolygonArea", "originalPolygonArea"]],
                            tableBaseName="combinedMeasures.csv",
                            ratioNamePrefix="ratio_", ratioSeperatorName="_"):
    measureTableName = resultsFolder + tableBaseName
    measureResultsTidyDf = pd.read_csv(measureTableName)
    for dividendName, divisiorName in ratioBetween:
        assert dividendName in measureResultsTidyDf.columns, f"The column {dividendName} is not present in the table from {measureResultsTidyDf}, only the following columns are present {measureResultsTidyDf.columns}"
        assert divisiorName in measureResultsTidyDf.columns, f"The column {divisiorName} is not present in the table from {measureResultsTidyDf}, only the following columns are present {measureResultsTidyDf.columns}"
        dividendValues = measureResultsTidyDf[dividendName]
        divisorValues = measureResultsTidyDf[divisiorName]
        ratioValues = dividendValues / divisorValues
        ratioColumnName = ratioNamePrefix + dividendName + ratioSeperatorName + divisiorName
        measureResultsTidyDf[ratioColumnName] = ratioValues
    measureResultsTidyDf.to_csv(measureTableName, index=False)

def combineTables(firstDf, secondDf, combineOn=["genotype", "replicate id", "time point", "cell label"]):
    firstDfColumns = firstDf.columns.to_numpy()
    secondDfColumns = secondDf.columns.to_numpy()
    secondDfMeasuresColumns = secondDfColumns[np.isin(secondDfColumns, combineOn, invert=True)]
    columnsToDrop = firstDfColumns[np.isin(firstDfColumns, secondDfMeasuresColumns)]
    reducedFirstDf = firstDf.drop(columns=columnsToDrop)
    mergedDf = pd.merge(reducedFirstDf, secondDf,  how='left', left_on=combineOn, right_on=combineOn)
    return mergedDf

def addMeasuresToTableFrom(allFolderContentsFilename, filenameKeys="microtubuleAnisotropyAndOrientation",
                           resultsFolder="", tableBaseName="combinedMeasures.csv"):
    if type(filenameKeys) == str:
        filenameKeys = [filenameKeys]
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    extractedFilesKeys = []
    isFirstTissue = True
    for folderContent in multiFolderContent:
        for key in filenameKeys:
            keysAdded = multiFolderContent.AddDataFromFilenameContainingMultipleDicts(key, returnIndividualKeysAdded=True)
            if isFirstTissue:
                extractedFilesKeys.extend(keysAdded)
        isFirstTissue = False
    additionalResultsTable = multiFolderContent.GetTidyDataFrameOf(extractedFilesKeys)
    measureTableName = resultsFolder + tableBaseName
    measureResultsTidyDf = pd.read_csv(measureTableName)
    measureResultsTidyDf = combineTables(measureResultsTidyDf, additionalResultsTable)
    measureResultsTidyDf.to_csv(measureTableName, index=False)

def calculateLobynessOfFolderContent(folderContent: FolderContent, resolutionInMicroMPerPixel: float = 1,
                                     outlineFilenameKey: str = "cellContours", lobynessFilenameKey: str = "lobyness"):
    cellOutlines = folderContent.LoadKeyUsingFilenameDict(outlineFilenameKey)
    lobynessPerCell = {}
    for cellLabel, outline in cellOutlines.items():
        lobyness = OtherMeasuresCreator().calcLobyness(list(outline)) / resolutionInMicroMPerPixel
        lobynessPerCell[cellLabel] = lobyness
    cellOutlinesFilename = folderContent.GetFilenameDictKeyValue(outlineFilenameKey)
    splitName = Path(cellOutlinesFilename).name.split(outlineFilenameKey)
    if len(splitName) > 1:
        baseName = splitName[0]
        baseName += "_" + lobynessFilenameKey + ".json"
    else:
        baseName = lobynessFilenameKey + ".json"
    lobynessResultsFilename = Path(cellOutlinesFilename).with_name(baseName)
    folderContent.SaveDataFilesTo(lobynessPerCell, lobynessResultsFilename)
    folderContent.AddDataToFilenameDict(lobynessResultsFilename, lobynessFilenameKey)

def calculateAndAddLobynessOf(dataBaseFolder="Images/", folderContentsName="Eng2021Cotyledons.pkl",
                              allFolderContentsFilename=None, reCalculate=True,
                              genotypeResolutionDict=None, lobynessFilenameKey: str = "lobyness"):
    if allFolderContentsFilename is None:
        allFolderContentsFilename = dataBaseFolder + folderContentsName
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    if reCalculate:
        for folderContent in multiFolderContent:
            if not genotypeResolutionDict is None:
                genotype = folderContent.GetGenotype()
                assert genotype in genotypeResolutionDict, f"The {genotype=} is not present in the {genotypeResolutionDict=}"
                resolutionInMicroMPerPixel = genotypeResolutionDict[genotype]
            else:
                resolutionInMicroMPerPixel = 1
            calculateLobynessOfFolderContent(folderContent, resolutionInMicroMPerPixel=resolutionInMicroMPerPixel,
                                             lobynessFilenameKey=lobynessFilenameKey)
        multiFolderContent.UpdateFolderContents()

def calculateRelativeCompleteness(folderContent: FolderContent,
                                 resolutionInMicroMPerPixel: float = 1,
                                 outlineFilenameKey: str = "cellContours", relativeCompletenessFilenameKey: str = "relativeCompleteness",
                                 additionalBoarder: np.ndarray = np.array([10,10])):
    cellOutlines = folderContent.LoadKeyUsingFilenameDict(outlineFilenameKey)
    relativeCompletenessPerCell = {}
    for cellLabel, outline in cellOutlines.items():
        relativeCompleteness = OtherMeasuresCreator().calcRelativeCompletenessForOutlines(list(outline),
                                                                                          resolutionInDistancePerPixel = resolutionInMicroMPerPixel,
                                                                                          additionalBoarder = additionalBoarder)
        relativeCompletenessPerCell[cellLabel] = relativeCompleteness
    cellOutlinesFilename = folderContent.GetFilenameDictKeyValue(outlineFilenameKey)
    splitName = Path(cellOutlinesFilename).name.split(outlineFilenameKey)
    if len(splitName) > 1:
        baseName = splitName[0]
        baseName += "_" + relativeCompletenessFilenameKey + ".json"
    else:
        baseName = relativeCompletenessFilenameKey + ".json"
    relativeCompletenessResultsFilename = Path(cellOutlinesFilename).with_name(baseName)
    folderContent.SaveDataFilesTo(relativeCompletenessPerCell, relativeCompletenessResultsFilename)
    folderContent.AddDataToFilenameDict(relativeCompletenessResultsFilename, relativeCompletenessFilenameKey)


def calculateAndAddRelativeCompletenessOf(dataBaseFolder="Images/", tableResultsFolder="Results/",
                              folderContentsName = "Eng2021Cotyledons.pkl", tableBaseName="combinedMeasures.csv",
                              allFolderContentsFilename=None, resultsTableFilename=None, reCalculate=True,
                              genotypeResolutionDict=None, relativeCompletenessFilenameKey: str = "relativeCompleteness"):
    if allFolderContentsFilename is None:
        allFolderContentsFilename = dataBaseFolder + folderContentsName
    if resultsTableFilename is None:
        resultsTableFilename = tableResultsFolder + tableBaseName
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    if reCalculate:
        for folderContent in multiFolderContent:
            if not genotypeResolutionDict is None:
                genotype = folderContent.GetGenotype()
                assert genotype in genotypeResolutionDict, f"The {genotype=} is not present in the {genotypeResolutionDict=}"
                resolutionInMicroMPerPixel = genotypeResolutionDict[genotype]
            else:
                resolutionInMicroMPerPixel = 1
            calculateRelativeCompleteness(folderContent, resolutionInMicroMPerPixel=resolutionInMicroMPerPixel,
                                         relativeCompletenessFilenameKey=relativeCompletenessFilenameKey)
        multiFolderContent.UpdateFolderContents()

def mainCalculateOnEng2021Cotyledon(reCalculateMeasures=True, redoTriWayJunctionPositioning=False):
    from InputData import GetInputData, GetResolutions
    dataBaseFolder = "Images/Eng2021Cotyledons/"
    resultsFolder = "Results/Eng2021Cotyledons/"
    folderContentsName = "Eng2021Cotyledons.pkl"
    allFolderContentsFilename = dataBaseFolder + folderContentsName
    inputData = GetInputData()
    genotypeResolutionDict = GetResolutions()
    if reCalculateMeasures:
        createFolderContentsOfAllTissues(allFolderContentsFilename, inputData, genotypeResolutionDict, dataBaseFolder)
        createAllEasyReadableContoursAndSimpleTriWayJunctions(dataBaseFolder, allFolderContentsFilename)
        createAllAdjacencyListsFromLabelledImage(dataBaseFolder, allFolderContentsFilename)
        checkTriWayJunctionPositioning(dataBaseFolder, allFolderContentsFilename, redoTriWayJunctionPositioning=redoTriWayJunctionPositioning)
        orderTriWayJunctionsOnContour(dataBaseFolder, allFolderContentsFilename)
        createRegularityMeasurements(allFolderContentsFilename, dataBaseFolder, genotypeResolutionDict=genotypeResolutionDict)
        createAreaAndDistanceMeasures(allFolderContentsFilename, dataBaseFolder, useGeometricData=False)
        # calculateAndAddLobynessOf(dataBaseFolder=dataBaseFolder, tableResultsFolder=resultsFolder, folderContentsName=folderContentsName, tableBaseName="combinedMeasures.csv", genotypeResolutionDict=genotypeResolutionDict)
        # calculateAndAddRelativeCompletenessOf(dataBaseFolder=dataBaseFolder, tableResultsFolder=resultsFolder, folderContentsName=folderContentsName, tableBaseName="combinedMeasures.csv", genotypeResolutionDict=genotypeResolutionDict)
    createResultMeasureTable(allFolderContentsFilename, resultsFolder)
    addRatioMeasuresToTable(resultsFolder)

def mainCalculateOnNewCotyledons(tissueName="full cotyledons", reCalculateMeasures=True,
                              createRegularityResults=True):
    dataBaseFolder = f"Images/{tissueName}/"
    resultsFolder = f"Results/{tissueName}/"
    allFolderContentsFilename = f"{dataBaseFolder}{tissueName}_multiFolderContent.pkl"
    if createRegularityResults:
        if reCalculateMeasures:
            createGeometricdataTableWithRemovedStomata()
            mainCreateAllFullCotyledons(allFolderContentsFilename, dataBaseFolder)
            createRegularityMeasurements(allFolderContentsFilename, dataBaseFolder,
                                         checkCellsPresentInLabelledImage=False)
            createAreaAndDistanceMeasures(allFolderContentsFilename, dataBaseFolder, useGeometricData=True)
        createResultMeasureTable(allFolderContentsFilename, resultsFolder,
                                 includeCellId=False)
        addRatioMeasuresToTable(resultsFolder)

def mainOfSAM(reCalculateMeasures=True):
    dataBaseFolder = "Images/Matz2022SAM/"
    resultsFolder = "Results/Matz2022SAM/"
    allFolderContentsFilename = "Images/Matz2022SAM_multiFolderContent.pkl"
    createAreaAndDistanceMeasures(allFolderContentsFilename, dataBaseFolder, useGeometricData=True)
    if reCalculateMeasures:
        mainInitalizeSAMDataAddingContoursAndJunctions(baseFilename=dataBaseFolder)
        addGeometricDataFilenameAndKey(allFolderContentsFilename, seperator="_", geometricTableBaseName="{}_geometricData.csv", polygonGeometricTableBaseName="{}_geometricData poly.csv")
        createRegularityMeasurements(allFolderContentsFilename, dataBaseFolder,
                                     checkCellsPresentInLabelledImage=False)
        createAreaAndDistanceMeasures(allFolderContentsFilename, dataBaseFolder, useGeometricData=True)
    createResultMeasureTable(allFolderContentsFilename, resultsFolder, loadMeasuresFromFilenameUsingKeys=["regularityMeasuresFilename", "areaMeasuresPerCell"], includeCellId=False)
    addRatioMeasuresToTable(resultsFolder)

if __name__== "__main__":
    mainCalculateOnEng2021Cotyledon(reCalculateMeasures=True)
    # mainCalculateOnNewCotyledons(reCalculateMeasures=True)
    # mainOfSAM(reCalculateMeasures=True)
