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
    originalEdgeDistances: dict or None = None # dict[str, list[float]] or None
    pooledTags: dict or None = None # dict[str, list[tuple]]
    pooledEdgeDistance: dict or None = None # dict[str, list[float]]
    randomizationDifferencesPerContent: dict or None = None # dict[str, list[list[float]]] or None
    # inner list of floats represents difference of original with randomization
    # outer list represents different entries from original

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

    def RandomizeEdgesWithoutPlanarityCheck(self,
                                            randomizationSeed: int or None = None,
                                            junctionPositionsKey: str or None = None,
                                            compareToValuesKey: str or None = None,
                                            poolingStrategy: str = "genotype",
                                            randomizationStrategy: str = "withReplacement"
                                            ):
        if junctionPositionsKey is not None:
            self.SetJunctionPositionsOfContent(junctionPositionsKey=junctionPositionsKey)
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
        for identifier, tagsToCombine in self.pooledTags.items():
            if randomizationStrategy == "withReplacement":
                edgeDistancesToChooseFrom = self.pooledEdgeDistance[identifier]
            else:
                raise NotImplementedError(f"The randomization strategy {randomizationStrategy} is not implemented yet.")
            currentOriginalEdgeDistances = self.originalEdgeDistances[identifier]
            randomizedEdgeDistancesOfContent = self.randomizeEdgeDistances(currentOriginalEdgeDistances, edgeDistancesToChooseFrom)

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
                currentEdgeDistances.extend(self.originalEdgeDistances[tag])
            pooledEdgeDistances[identifier] = currentEdgeDistances
        return pooledEdgeDistances

    def randomizeEdgeDistances(self, currentOriginalEdgeDistances, edgeDistancesToChooseFrom):
        randomizedEdgeDistances = {}
        for currentId, currentEdgeDistances in currentOriginalEdgeDistances.items():
            numberOfOriginalDistances = len(currentEdgeDistances)
            randomizedEdgeDistances[currentId] = np.random.choice(edgeDistancesToChooseFrom, size=numberOfOriginalDistances, replace=True)
        return randomizedEdgeDistances

def testFunctionality():
    dataSetName = "Eng2021Cotyledons" # "Smit2023Cotyledons" #
    junctionPositionsKey = "orderedJunctionsPerCellFilename"

    filename = f"Images/{dataSetName}/{dataSetName}.json"
    randomizer = EdgeRandomizationAnalysis(filename)
    randomizer.RandomizeEdgesWithoutPlanarityCheck(junctionPositionsKey=junctionPositionsKey)

if __name__ == '__main__':
    testFunctionality()