import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, "./Code/DataStructures/")
sys.path.insert(0, "./Code/MeasureCreator/")

from copy import deepcopy
from MultiFolderContent import MultiFolderContent
from PolygonalRegularityCalculator import PolygonalRegularityCalculator

class EdgeRandomizationAnalysis:

    folderContents: MultiFolderContent = None
    currentSeed: int = 42
    junctionPositionsOfContent: dict or None = None #dict[str, dict[int, list[list[float]]]] or None = None
    originalMeasuresToCompareTo: dict or None = None
    lastMeasureKey: str or None = None
    lastKeyForNestedMeasureFilename: str or None = None
    originalEdgeDistances: dict or None = None # dict[str, list[float]] or None
    pooledTags: dict or None = None # dict[str, list[tuple]]
    pooledEdgeDistance: dict or None = None # dict[str, list[float]]
    randomizationDifferencesPerContent: dict or None = None # dict[str, list[list[float]]] or None
    # inner list of floats represents difference of original with randomization
    # outer list represents different entries from original
    implementedMeasures: list = ["lengthGiniCoeff"]

    def __init__(self, folderContentsFilename: str or MultiFolderContent):
        self.folderContents = MultiFolderContent(folderContentsFilename)

    def SetJunctionPositionsOfContent(self, junctionPositionsKey: str, folderContentsFilename: str or MultiFolderContent or None = None):
        if folderContentsFilename is not None:
            self.folderContents = MultiFolderContent(folderContentsFilename)
        assert self.folderContents is not None, f"The folder contents needs to be defined, when setting the junction positions of these contents."
        self.junctionPositionsOfContent = {}
        for folderContent in self.folderContents:
            tissueTag: tuple = folderContent.GetTissueInfos()
            self.junctionPositionsOfContent[tissueTag] = folderContent.LoadKeyUsingFilenameDict(junctionPositionsKey,
                                                                                                **dict(convertDictKeysToInt=True,
                                                                                                       convertDictValuesToNpArray=True))

    def SetOriginalMeasuresToCompareTo(self, measureKey, keyForNestedMeasureFilename: str or None = None):
        assert self.folderContents is not None, "The folder contents are not set yet."
        self.originalMeasuresToCompareTo = {}
        if keyForNestedMeasureFilename is None:
            filenameKey = measureKey
        else:
            filenameKey = keyForNestedMeasureFilename
        for folderContent in self.folderContents:
            tissueTag: tuple = folderContent.GetTissueInfos()
            fileContent = folderContent.LoadKeyUsingFilenameDict(filenameKey,
                                                                 **dict(convertDictKeysToInt=False,
                                                                        convertNestedDictKeysToInt=True,
                                                                        convertDictValuesToNpArray=True))
            if keyForNestedMeasureFilename is not None:
                fileContent = fileContent[measureKey]
            self.originalMeasuresToCompareTo[tissueTag] = fileContent
        self.lastMeasureKey = measureKey
        self.lastKeyForNestedMeasureFilename = keyForNestedMeasureFilename

    def RandomizeEdgesWithoutPlanarityCheck(self,
                                            measureKey: str,
                                            randomizationSeed: int or None = None,
                                            junctionPositionsKey: str or None = None,
                                            keyForNestedMeasureFilename: str or None = None,
                                            poolingStrategy: str = "genotype",
                                            randomizationStrategy: str = "withReplacement"
                                            ):
        assert measureKey in self.implementedMeasures, f"The measure named {measureKey} is not in the list of implemented measures yet, only {implementedMeasures} is implemented."
        if junctionPositionsKey is not None:
            self.SetJunctionPositionsOfContent(junctionPositionsKey=junctionPositionsKey)
            if self.lastMeasureKey != measureKey or self.lastKeyForNestedMeasureFilename != keyForNestedMeasureFilename:
                self.SetOriginalMeasuresToCompareTo(measureKey, keyForNestedMeasureFilename)
            self.randomizationDifferencesPerContent = {}
            self.originalEdgeDistances = None
            self.pooledTags = None
            self.pooledEdgeDistance = None
        assert self.junctionPositionsOfContent is not None, f"You need to either specify the junction positions of the corresponding contents (name being key) or specify the junctionPositionsKey parameter."
        if self.originalEdgeDistances is None:
            self.originalEdgeDistances = self.extractEdgeDistanceFromFolderContents()
        if randomizationSeed is None:
            self.currentSeed += 1
        else:
            self.currentSeed = randomizationSeed
        np.random.seed(self.currentSeed)
        #  pooling and randomization of edges
        folderContentTags = list(self.originalEdgeDistances.keys())
        if self.pooledTags is None:
            self.pooledTags = self.determineTagsToPool(folderContentTags, poolingStrategy)
            self.pooledEdgeDistance = self.poolEdgeDistances(self.pooledTags)
            self.randomizationDifferencesPerContent = {identifier: [] for identifier in self.pooledTags.keys()}
        for identifier, tagsToCombine in self.pooledTags.items():
            if randomizationStrategy == "withReplacement":
                edgeDistancesToChooseFrom = self.pooledEdgeDistance[identifier]
            else:
                raise NotImplementedError(f"The randomization strategy {randomizationStrategy} is not implemented yet.")
            for tag in tagsToCombine:
                currentOriginalEdgeDistances = self.originalEdgeDistances[tag]
                randomizedEdgeDistancesOfContent = self.randomizeEdgeDistances(currentOriginalEdgeDistances, edgeDistancesToChooseFrom)
                measuresBasedOnRandomizedContent = self.calculateMeasureOn(randomizedEdgeDistancesOfContent, measureKey)
                originalMeasures = self.originalMeasuresToCompareTo[tag]
                measureDifferences = []
                for cellId in originalMeasures.keys():
                    measureDifferences.append(measuresBasedOnRandomizedContent[cellId] - originalMeasures[cellId])
                self.randomizationDifferencesPerContent[identifier].extend(measureDifferences)

    def AnalyzeRandomizationResults(self, saveProperties: dict or None = None, showPlot: bool = False):
        # <----- implement visualization here
        if saveProperties is not None:
            plt.savefig(**saveProperties)
        elif showPlot:
            plt.show()

    def extractEdgeDistanceFromFolderContents(self):
        edgeDistances = {}
        polygonHelper = PolygonalRegularityCalculator()
        for folderContent in self.folderContents:
            resolution = folderContent.GetResolution()
            if resolution is None:
                resolution = 1
            polygonHelper.SetResolution(resolution)
            tissueTag: tuple = folderContent.GetTissueInfos()
            junctionsOfCells = self.junctionPositionsOfContent[tissueTag]
            edgeDistances[tissueTag] = {}
            for cellId, junctionPositions in junctionsOfCells.items():
                edgeDistances[tissueTag][cellId] = polygonHelper.calcPolygonSideLengths(junctionPositions)
        return edgeDistances

    def determineTagsToPool(self, tagsToPool: list, poolingStrategy: str):
        # thinkable could also be a combination of genotype and time point
        # check GetTissueInfos for order of tag information
        if poolingStrategy == "genotype":
            return self.poolTagsByIndex(tagsToPool, 0)
        else:
            raise NotImplementedError(f"THe pooling strategy {poolingStrategy} is not yet implemented")

    def poolTagsByIndex(self, tagsToPool: list, indexToPoolBy: int):
        pooledTags = [[tagsToPool[0]]]
        identifierOfPools = [tagsToPool[0][indexToPoolBy]]
        for tag in tagsToPool[1::]:
            currentIdentifier = tag[indexToPoolBy]
            indexOfCorrespondingPools = np.where(np.isin(identifierOfPools, currentIdentifier))[0]
            if len(indexOfCorrespondingPools) == 0:
                identifierOfPools.append(currentIdentifier)
                pooledTags.append([tag])
            elif len(indexOfCorrespondingPools) == 1:
                pooledTags[indexOfCorrespondingPools[0]].append(tag)
            else:
                raise IndexError(f"There should never be more than one identifier to pool tags by index, indices {indexOfCorrespondingPools} is the current id {currentIdentifier} in the already existing identifiers {identifierOfPools}")
        return dict(zip(identifierOfPools, pooledTags))

    def poolEdgeDistances(self, pooledTags: dict):
        pooledEdgeDistances = {}
        for identifier, tagsToCombine in pooledTags.items():
            currentEdgeDistances = []
            for tag in tagsToCombine:
                currentEdgeDistances.extend(list(np.concatenate(list(self.originalEdgeDistances[tag].values()))))
            pooledEdgeDistances[identifier] = currentEdgeDistances
        return pooledEdgeDistances

    def randomizeEdgeDistances(self, currentOriginalEdgeDistances, edgeDistancesToChooseFrom):
        randomizedEdgeDistances = {}
        for currentId, currentEdgeDistances in currentOriginalEdgeDistances.items():
            numberOfOriginalDistances = len(currentEdgeDistances)
            randomizedEdgeDistances[currentId] = np.random.choice(edgeDistancesToChooseFrom, size=numberOfOriginalDistances, replace=True)
        return randomizedEdgeDistances

    def calculateMeasureOn(self, randomizedEdgeDistancesOfContent, measureKey):
        measuresOfCells = {}
        if measureKey == "lengthGiniCoeff":
            polygonHelper = PolygonalRegularityCalculator()
            for cellId, edgeDistances in randomizedEdgeDistancesOfContent.items():
                measuresOfCells[cellId] = polygonHelper.calcGiniCoefficient(edgeDistances)
        else:
            raise NotImplementedError(f"The measure named {measureKey} is not in the list of implemented measures yet, only {self.implementedMeasures} is implemented.")
        return measuresOfCells

def testFunctionality():
    dataSetName = "Eng2021Cotyledons" # "Smit2023Cotyledons" #
    junctionPositionsKey = "orderedJunctionsPerCellFilename"
    keyForNestedMeasureFilename = "regularityMeasuresFilename"
    measureKey = "lengthGiniCoeff"

    filename = f"Images/{dataSetName}/{dataSetName}.json"
    randomizer = EdgeRandomizationAnalysis(filename)
    randomizer.RandomizeEdgesWithoutPlanarityCheck(measureKey=measureKey, junctionPositionsKey=junctionPositionsKey, keyForNestedMeasureFilename=keyForNestedMeasureFilename)

if __name__ == '__main__':
    testFunctionality()