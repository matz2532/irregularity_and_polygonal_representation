import json
import numpy as np
import sys
import warnings

import pandas as pd
from pathlib import Path
from shapely.geometry import Polygon
from skimage.measure import regionprops

sys.path.insert(0, "./Code/DataStructures/")

from PolygonalRegularityCalculator import PolygonalRegularityCalculator

class AreaMeasureExtractor (PolygonalRegularityCalculator):

    areaMeasuresKey="areaMeasuresPerCell"
    defaultResolutionInSizePerPixels=1
    geometricCentersKey = "geometricCentersPerCell"
    neighborDistanceKey = "neighborDistance"
    adjacencyListKey="labelledImageAdjacencyList"
    # geometry tables related parameters
    cellAreaColIdxOfGeometricData=2
    cellGeometricCenterColIdxOfGeometricData=[3, 4, 5]
    cellLabelColIdxOfGeometricData=0
    skipFooter=4
    geometricDataFilenameKey="geometricData"
    polygonGeometricDataFilenameKey="polygonalGeometricData"
    areaMeasuresDict=None

    def __init__(self, folderContent=None, saveAreaMeasuresAsFilename=None,
                 overWriteResolutionInSizePerPixels=None, **kwargs):
        if not folderContent is None:
            self.SetFolderContent(folderContent, overWriteResolutionInSizePerPixels)
            if not saveAreaMeasuresAsFilename is None:
                self.RunAndSaveAllAreaMeasures(saveAreaMeasuresAsFilename, **kwargs)
                self.RunAndSaveAllGeometricCentersAndDistances(Path(saveAreaMeasuresAsFilename).with_suffix(""), **kwargs)

    def SetDefaultResolutionInSizePerPixels(self, defaultResolutionInSizePerPixels):
        self.defaultResolutionInSizePerPixels = defaultResolutionInSizePerPixels

    def SetFolderContent(self, folderContent, overWriteResolutionInSizePerPixels=None,
                         printOutWhichResolutionIsUsed=False):
        self.folderContent = folderContent
        tissueName = self.folderContent.GetTissueName()
        if not overWriteResolutionInSizePerPixels is None:
            self.resolutionInSizePerPixels = overWriteResolutionInSizePerPixels
            if printOutWhichResolutionIsUsed:
                print(f"The overwritten {self.resolutionInSizePerPixels=} of {tissueName} is used")
        else:
            resolution = self.folderContent.GetResolution()
            if not resolution is None:
                self.resolutionInSizePerPixels = resolution
                if printOutWhichResolutionIsUsed:
                    print(f"The {self.resolutionInSizePerPixels=} of {tissueName} is used.")
            else:
                self.resolutionInSizePerPixels = self.defaultResolutionInSizePerPixels
                if printOutWhichResolutionIsUsed:
                    print(f"The default parameter {self.resolutionInSizePerPixels=} of {tissueName} is used.")

    def GetAreaMeasuresDict(self):
        return self.areaMeasuresDict

    def RunAndSaveAllAreaMeasures(self, saveAreaMeasuresAsFilename=None, useGeometricData=False,
                                  labelledCellsImageFilename="labelledImageFilename",
                                  allowedLabelFilename="cellContours", orderedJunctionsPerCellFilename="orderedJunctionsPerCellFilename", getLabelsFromDictKeys=True):
        if useGeometricData:
            labelledImageArea = self.ExtractValuesFromGeometricData(geometricDataKey=self.geometricDataFilenameKey,
                                                                    allowedLabelFilename=allowedLabelFilename,
                                                                    colIdxOfValue=self.cellAreaColIdxOfGeometricData,
                                                                    getLabelsFromDictKeys=getLabelsFromDictKeys)
            originalPolygonArea = self.ExtractValuesFromGeometricData(geometricDataKey=self.polygonGeometricDataFilenameKey,
                                                                      colIdxOfValue=self.cellAreaColIdxOfGeometricData,
                                                                      allowedLabelFilename=allowedLabelFilename,
                                                                      getLabelsFromDictKeys=getLabelsFromDictKeys)
        else:
            labelledImageArea = self.ExtractLabelledImageAreaOfCellLabels(labelledCellsImageFilename=labelledCellsImageFilename,
                                                                          allowedLabelFilename=allowedLabelFilename,
                                                                          getLabelsFromDictKeys=getLabelsFromDictKeys)
            originalPolygonArea = self.ExtractOriginalPolygonAreas(allowedLabelFilename=orderedJunctionsPerCellFilename)
        self.areaMeasuresDict = {}
        self.areaMeasuresDict["labelledImageArea"] = labelledImageArea
        self.areaMeasuresDict["originalPolygonArea"] = originalPolygonArea
        regularPolygonArea = self.ExtractRegularPresumedPolygonAreas(orderedJunctionsPerCellFilename=orderedJunctionsPerCellFilename, allowedLabelFilename=allowedLabelFilename)
        self.areaMeasuresDict["regularPolygonArea"] = regularPolygonArea
        if not saveAreaMeasuresAsFilename is None:
            Path(saveAreaMeasuresAsFilename).parent.mkdir(parents=True, exist_ok=True)
            self.folderContent.SaveDataFilesTo(self.areaMeasuresDict, saveAreaMeasuresAsFilename)
            if not self.folderContent is None:
                self.folderContent.AddDataToFilenameDict(saveAreaMeasuresAsFilename, self.areaMeasuresKey)
        return self.areaMeasuresDict

    def ExtractValuesFromGeometricData(self, geometricDataKey, allowedLabelFilename, colIdxOfValue, getLabelsFromDictKeys=True):
        # extract value/s of geometric data
        geometricData = self.folderContent.LoadKeyUsingFilenameDict(geometricDataKey, skipfooter=self.skipFooter)
        allCellLabels = geometricData.iloc[:, self.cellLabelColIdxOfGeometricData]
        allCellValues = geometricData.iloc[:, colIdxOfValue]
        valuesOfCells = {}
        allowedLabels = self.folderContent.LoadKeyUsingFilenameDict(allowedLabelFilename, convertDictKeysToInt=True)
        if getLabelsFromDictKeys:
            allowedLabels = list(allowedLabels.keys())
        for labelId in allowedLabels:
            isLabelledCell = allCellLabels == labelId
            assert np.sum(isLabelledCell) == 1, f"""The {labelId=} is not exactly present once, {np.sum(isLabelledCell)} != 1,
            from the tissue {self.folderContent.GetTissueName()} using the allowed labels {allowedLabels} from the file {self.folderContent.GetFilenameDictKeyValue(allowedLabelFilename)}
            to match the labels {allCellLabels.to_list()} with area values from {self.folderContent.GetFilenameDictKeyValue(geometricDataKey)}"""
            valuesOfCells[labelId] = allCellValues[isLabelledCell].to_numpy()[0]
        return valuesOfCells

    def ExtractLabelledImageAreaOfCellLabels(self, labelledCellsImageFilename="labelledImageFilename",
                                         allowedLabelFilename="cellContours", getLabelsFromDictKeys=True):
        # load labelled image
        labelledCellsImage = self.folderContent.LoadKeyUsingFilenameDict(labelledCellsImageFilename)
        # extract area of labelled image
        labelledImageArea = {}#
        allowedLabels = self.folderContent.LoadKeyUsingFilenameDict(allowedLabelFilename, convertDictKeysToInt=True)
        if getLabelsFromDictKeys:
            allowedLabels = list(allowedLabels.keys())
        for labelId in allowedLabels:
            isLabelledCell = labelledCellsImage == labelId
            nrOfCellLabelPixels = np.sum(isLabelledCell)
            areaOfLabelledCell = nrOfCellLabelPixels * self.resolutionInSizePerPixels * self.resolutionInSizePerPixels
            labelledImageArea[labelId] = areaOfLabelledCell
        return labelledImageArea

    def ExtractOriginalPolygonAreas(self, allowedLabelFilename="cellContours"):
        orderedJunctionsPerCell = self.folderContent.LoadKeyUsingFilenameDict(allowedLabelFilename, convertDictKeysToInt=True, convertDictValuesToNpArray=True)
        originalPolygonArea = {}
        for labelId, orderedJunctions in orderedJunctionsPerCell.items():
            polygon = Polygon(orderedJunctions)
            originalPolygonArea[labelId] = polygon.area * self.resolutionInSizePerPixels * self.resolutionInSizePerPixels
        return originalPolygonArea

    def ExtractRegularPresumedPolygonAreas(self, orderedJunctionsPerCellFilename="orderedJunctionsPerCellFilename", allowedLabelFilename="cellContours"):
        allowedLabel = self.folderContent.LoadKeyUsingFilenameDict(allowedLabelFilename, convertDictValuesToNpArray=True, convertDictKeysToInt=True)
        orderedJunctionsPerCell = self.folderContent.LoadKeyUsingFilenameDict(orderedJunctionsPerCellFilename)
        regularPolygonArea = {}
        if type(allowedLabel) == dict:
            allowedLabel = list(allowedLabel.keys())
        for labelId in allowedLabel:
            assert labelId in orderedJunctionsPerCell, f"The allowed cell label {labelId} does not have ordered junctions based on the filename {self.folderContent.GetFilenameDictKeyValue(orderedJunctionsPerCellFilename)}\nYou can only select from the following {np.sort(list(orderedJunctionsPerCell.keys()))}"
            orderedJunctions = orderedJunctionsPerCell[labelId]
            polygonArea = self.calcRegularPresumedPolygonArea(orderedJunctions)
            regularPolygonArea[labelId] = polygonArea * self.resolutionInSizePerPixels * self.resolutionInSizePerPixels
        return regularPolygonArea

    def RunAndSaveAllGeometricCentersAndDistances(self, baseResultsFilename=None, useGeometricData=False,
                            labelledCellsImageFilename="labelledImageFilename", normalizeWithAreaKey="labelledImageArea",
                            allowedLabelFilename="cellContours", adjacencyListKey=None, getLabelsFromDictKeys=True):
        if useGeometricData:
            geometricCentersOfCells = self.ExtractValuesFromGeometricData(geometricDataKey=self.geometricDataFilenameKey,
                                                                          allowedLabelFilename=allowedLabelFilename,
                                                                          colIdxOfValue=self.cellGeometricCenterColIdxOfGeometricData,
                                                                          getLabelsFromDictKeys=getLabelsFromDictKeys)
        else:
            geometricCentersOfCells = self.ExtractGeometricCentersOf(labelledCellsImageFilename=labelledCellsImageFilename,
                                                                    allowedLabelFilename=allowedLabelFilename,
                                                                    getLabelsFromDictKeys=getLabelsFromDictKeys)
        if adjacencyListKey is None:
            adjacencyListKey = self.adjacencyListKey
        neighborDistances = self.extractNeighborDistances(geometricCentersOfCells, adjacencyListKey, isNonPixelResolution=useGeometricData)
        if not normalizeWithAreaKey is None:
            if isinstance(normalizeWithAreaKey, list):
                for normalizationKey in normalizeWithAreaKey:
                    self.addNormalizedDistance(neighborDistances, normalizationKey)
            else:
                self.addNormalizedDistance(neighborDistances, normalizeWithAreaKey)
        if not baseResultsFilename is None:
            geometricCentersFilename = baseResultsFilename + self.geometricCentersKey + ".json"
            neighborDistanceFilename = baseResultsFilename + self.neighborDistanceKey + ".csv"
            with open(geometricCentersFilename, "w") as fh:
                json.dump({str(k):[float(i) for i in v] for k, v in geometricCentersOfCells.items()}, fh)
            neighborDistances.to_csv(neighborDistanceFilename, index=False)
            if not self.folderContent is None:
                self.folderContent.AddDataToFilenameDict(geometricCentersFilename, self.geometricCentersKey)
                self.folderContent.AddDataToFilenameDict(neighborDistanceFilename, self.neighborDistanceKey)
        return neighborDistances

    def addNormalizedDistance(self, neighborDistanceDf, normalizeWithAreaKey, fromColIdx=0, toColIdx=1, distanceColIdx=2):
        if self.areaMeasuresDict is None:
            warnings.warn(f"For the folder content {self.folderContent.GetTissueName()} no areaMeasuresDict is calculated while the distance should be normalised with the {normalizeWithAreaKey=}\nEither set normalizeWithAreaKey to None or set areaMeasuresDict with RunAndSaveAllAreaMeasures()")
        elif not isinstance(self.areaMeasuresDict, dict):
            warnings.warn(f"For the folder content {self.folderContent.GetTissueName()} the areaMeasuresDict is not a dictionary, but {type(self.areaMeasuresDict)}")
        elif not normalizeWithAreaKey in self.areaMeasuresDict:
            warnings.warn(f"For the folder content {self.folderContent.GetTissueName()} the key {normalizeWithAreaKey=} is not present in self.areaMeasuresDict only the following keys are present{list(self.areaMeasuresDict.keys())}")
        else:
            areasToNormalizeWith = self.areaMeasuresDict[normalizeWithAreaKey]
            uniqueCellIds = np.unique(neighborDistanceDf.iloc[:, [fromColIdx, toColIdx]])
            cellsWithAreaValues = list(areasToNormalizeWith.keys())
            isCellMissingArea = np.isin(uniqueCellIds, cellsWithAreaValues, invert=True)
            assert np.sum(isCellMissingArea) == 0, f"The cell/s {uniqueCellIds[isCellMissingArea]} are missing area values={normalizeWithAreaKey}, only {cellsWithAreaValues} are present."
            normalizedDistance = []
            for _, row in neighborDistanceDf.iterrows():
                fromId, toId, distance = row.iloc[[fromColIdx, toColIdx, distanceColIdx]]
                normalizationFactor = np.sqrt(2 / (areasToNormalizeWith[fromId] + areasToNormalizeWith[toId]))
                normalizedDistance.append(distance * normalizationFactor)
            normalizedDistanceColName = "distance normalised by mean " + normalizeWithAreaKey
            neighborDistanceDf[normalizedDistanceColName] = normalizedDistance

    def ExtractGeometricCentersOf(self, labelledCellsImageFilename="labelledImageFilename",
                                 allowedLabelFilename="cellContours", getLabelsFromDictKeys=True):
        # load labelled image
        labelledCellsImage = self.folderContent.LoadKeyUsingFilenameDict(labelledCellsImageFilename)
        regionPropOfImage = regionprops(labelledCellsImage)
        # extract area of labelled image
        geometricCenterOfCells = {}
        allowedLabels = self.folderContent.LoadKeyUsingFilenameDict(allowedLabelFilename, convertDictKeysToInt=True, convertDictValuesToNpArray=True)
        if getLabelsFromDictKeys:
            allowedLabels = list(allowedLabels.keys())
        uniqueLabelledCells = np.unique(labelledCellsImage)
        uniqueLabelledCells = uniqueLabelledCells[np.isin(uniqueLabelledCells, 0, invert=True)]
        for labelId in allowedLabels:
            isLabelledCell = uniqueLabelledCells == labelId
            idxOfLabelId = np.where(isLabelledCell)[0]
            if len(idxOfLabelId) > 0:
                geometricCenter = regionPropOfImage[idxOfLabelId[0]].centroid
                geometricCenterOfCells[labelId] = geometricCenter
            else:
                print(f"""The {labelId=} is not present in the labelled image, only {uniqueLabelledCells} are allowed.
                The labelled image was loaded from {self.folderContent.GetFilenameDictKeyValue(labelledCellsImageFilename)} and
                the allowed labels from {self.folderContent.GetFilenameDictKeyValue(allowedLabelFilename)}""")
        return geometricCenterOfCells

    def extractNeighborDistances(self, geometricCentersOfCells, adjacencyListKey, convertDictKeysToInt=True, isNonPixelResolution=False, printMissing=False):
        adjacencyList = self.folderContent.LoadKeyUsingFilenameDict(adjacencyListKey, convertDictKeysToInt=convertDictKeysToInt)
        neighborDistances, cellIdPairs = [], []
        for i, neighbors in adjacencyList.items():
            if not i in geometricCentersOfCells:
                if printMissing:
                    print(f"The distance of {i=} to it's neighbors could not be calculated and the cell was ignored as it is not present in the geometric centers of cells")  # , only {np.sort(list(geometricCentersOfCells.keys()))} are present.")
            else:
                isIPresent = np.array([i in pair for pair in cellIdPairs])
                for j in neighbors:
                    isJPresent = np.array([j in pair for pair in cellIdPairs])
                    if len(cellIdPairs) > 0:
                        isPairPresent = isJPresent & isIPresent
                        if np.any(isPairPresent):
                            continue
                    if i in geometricCentersOfCells and j in geometricCentersOfCells:
                        positionOfI = np.array(geometricCentersOfCells[i])
                        positionOfJ = np.array(geometricCentersOfCells[j])
                        distance = np.linalg.norm(positionOfI - positionOfJ)
                        if not isNonPixelResolution:
                            distance *= self.resolutionInSizePerPixels
                        neighborDistances.append([i, j, distance])
                        cellIdPairs.append({i, j})
                        isIPresent = np.array([i in pair for pair in cellIdPairs])
                    else:
                        if printMissing:
                            print(f"The distance between the adjacent cells {i=} and {j=} was ignored as i is not present in the geometric centers of cells") #, only {np.sort(list(geometricCentersOfCells.keys()))} are present from .")

        neighborDistancesDf = pd.DataFrame(neighborDistances, columns=["from", "to", "distance"])
        return neighborDistancesDf

    def calcRegularPresumedPolygonArea(self, orderedJunctions):
        polygonalSideLengths = self.calcPolygonSideLengths(orderedJunctions)
        perimeter = np.sum(polygonalSideLengths)
        apothem = self.calcApothemOfRegularPolygon(polygonalSideLengths)
        regularPolygonArea = 0.5 * perimeter * apothem
        return regularPolygonArea

    def calcApothemOfRegularPolygon(self, polygonalSideLengths):
        sideLength = np.mean(polygonalSideLengths)
        nrOfSides = len(polygonalSideLengths)
        apothem = sideLength / (2 * np.tan(np.deg2rad(180/nrOfSides)))
        return apothem

def testAreaMeasureExtractor(useGeometricData=False, dataBaseFolder=None,
                             testAreaCalculations=True, testDistanceCalculations=True,
                             timePointInPath=True, areaMeasuresKey="areaMeasuresPerCell"):
    from MultiFolderContent import MultiFolderContent
    if useGeometricData:
        allowedLabelFilename = "orderedJunctionsPerCellFilename"
        allFolderContentsFilename = f"Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"
    else:
        allowedLabelFilename = "cellContours"
        allFolderContentsFilename = f"Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    areaExtractor = AreaMeasureExtractor(None)
    for folderContent in multiFolderContent:
        scenarioName, replicateName, timePoint = folderContent.GetTissueInfos()
        print(scenarioName, replicateName, timePoint)
        if timePointInPath:
            replicateName = replicateName + "_" + timePoint
            folderExtension = folderContent.GetFolder()
        else:
            replicateName = replicateName
            folderExtension = scenarioName + "/" + replicateName + "/"
        if  dataBaseFolder is None:
            baseResultsFilename = None
            saveAreaMeasuresAsFilename = None
        else:
            baseResultsFilename = dataBaseFolder + folderExtension + replicateName + "_"
            saveAreaMeasuresAsFilename = baseResultsFilename + areaMeasuresKey + ".pkl"
        areaExtractor.SetFolderContent(folderContent)
        areaMeasuresDict, neighborDistances = None, None
        if testAreaCalculations:
            areaMeasuresDict = areaExtractor.RunAndSaveAllAreaMeasures(useGeometricData=useGeometricData, saveAreaMeasuresAsFilename=saveAreaMeasuresAsFilename)
            print(f"{areaMeasuresDict=}")
        if testDistanceCalculations:
            neighborDistances = areaExtractor.RunAndSaveAllGeometricCentersAndDistances(useGeometricData=useGeometricData, baseResultsFilename=baseResultsFilename)
            print(f"{neighborDistances}")

        return areaMeasuresDict, neighborDistances

def main():
    from MultiFolderContent import MultiFolderContent
    allFolderContentsFilename = "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    for folderContent in multiFolderContent:
        filenameDict = folderContent.GetFilenameDict()
        myAreaMeasureExtractor = AreaMeasureExtractor(folderContent, saveAreaMeasuresAsFilename="test.json", useGeometricData=True)
        areaResults = myAreaMeasureExtractor.GetAreaMeasuresDict()
        firstCellLabel = list(list(areaResults.values())[0].keys())[0]
        print([v[firstCellLabel] for v in areaResults.values()])
        sys.exit()

def testIndividualTissue():
    redoForTissueInfo = ("ktn inflorescence meristem", "ktnP3", "T0")
    from MultiFolderContent import MultiFolderContent
    allFolderContentsFilename = "Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl"
    dataBaseFolder = "Images/SAM/"
    areaMeasuresKey = "areaMeasuresPerCell"
    useGeometricData =True
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    folderContent = multiFolderContent.GetFolderContentOfIdentifier(redoForTissueInfo)
    scenarioName, replicateName, timePoint = folderContent.GetTissueInfos()
    replicateName = replicateName
    folderExtension = scenarioName + "/" + replicateName + "/"
    baseResultsFilename = dataBaseFolder + folderExtension + replicateName + "_"
    saveAreaMeasuresAsFilename = baseResultsFilename + areaMeasuresKey + ".json"
    areaExtractor = AreaMeasureExtractor(folderContent)
    areaExtractor.RunAndSaveAllAreaMeasures(useGeometricData=useGeometricData, saveAreaMeasuresAsFilename=saveAreaMeasuresAsFilename)
    # multiFolderContent.UpdateFolderContents()
    areaMeasuresPerCell = folderContent.LoadKeyUsingFilenameDict("areaMeasuresPerCell")
    cellContours = folderContent.LoadKeyUsingFilenameDict("cellContours", convertDictKeysToInt=True, convertDictValuesToNpArray=True)
    print(len(list(cellContours.values())))
    print([len(v) for v in areaMeasuresPerCell.values()])

def combineNeighborDistanceTables(multiFolderContentsOfTissues = {"SAM":"Images/Matz2022SAM.pkl", "cotyledon patches": "Images/Eng2021Cotyledons.pkl", "full cotyledons":"Images/full cotyledons/full cotyledons.pkl"},
                                  folderExtensionExemption={"cotyledon patches":""}, baseResultsFolder="Results/", tableBaseNameOf="combinedMeasures_{}.csv",
                                  neighborDistanceFilenameKey="neighborDistance", tissueInfoColumnName=["genotype", "replicate id", "time point"]):
    from MultiFolderContent import MultiFolderContent
    for tissueTypeName, tissueTypeFolderContentsFilename in multiFolderContentsOfTissues.items():
        combinedDistanceDf = []
        multiFolderContent = MultiFolderContent(tissueTypeFolderContentsFilename)
        for folderContent in multiFolderContent:
            neighborDistanceDf = folderContent.LoadKeyUsingFilenameDict(neighborDistanceFilenameKey)
            tissueInfo = folderContent.GetTissueInfos()
            nrOfDistances = len(neighborDistanceDf)
            tissueInfoOfEntry = np.repeat(tissueInfo, nrOfDistances).reshape(len(tissueInfo), nrOfDistances).T
            tissueInfoOfEntryDf = pd.DataFrame(tissueInfoOfEntry, columns=tissueInfoColumnName)
            neighborDistanceDf = pd.concat([tissueInfoOfEntryDf, neighborDistanceDf], axis=1)
            combinedDistanceDf.append(neighborDistanceDf)
        combinedDistanceDf = pd.concat(combinedDistanceDf, axis=0)
        if tissueTypeName in folderExtensionExemption:
            folderExtension = folderExtensionExemption[tissueTypeName]
        else:
            folderExtension = f"{tissueTypeName}/"
        filenameToSaveDistanceTable = baseResultsFolder + folderExtension + tableBaseNameOf.format(neighborDistanceFilenameKey)
        combinedDistanceDf.to_csv(filenameToSaveDistanceTable, index=False)

if __name__ == '__main__':
    # main()
    # testAreaMeasureExtractor(useGeometricData=False)
    # testAreaMeasureExtractor(useGeometricData=True)
    # testIndividualTissue()
    combineNeighborDistanceTables()
