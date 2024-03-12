import json
import numpy as np
import os
import pandas as pd
import pickle
import warnings

from CellIdTracker import CellIdTracker
from FolderContent import FolderContent
from pathlib import Path

class MultiFolderContent:

    _index = 0
    verbose=1
    allFolderContents=None

    def __init__(self, allFolderContentsFilename: str or Path = None, rawFolderContents: list = None, resetFolderContent: bool = False):
        self.allFolderContentsFilename = allFolderContentsFilename
        if resetFolderContent:
            self.ResetFolderContents()
        elif not rawFolderContents is None:
            self.rawFolderContents = rawFolderContents
            self.allFolderContents = self.ConvertRawToProcessedFolderContents(self.rawFolderContents)
        elif not self.allFolderContentsFilename is None:
            self.LoadFolderContentsFromFilename(self.allFolderContentsFilename)
        else:
            self.ResetFolderContents()
            if self.verbose >= 1:
                print("The MultiFolderContent was initialised empty and without a filename to save it under")

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.allFolderContents):
            self._index += 1
            return self.allFolderContents[self._index-1]
        raise StopIteration

    def __str__(self):
        tissueNames = self.GetTissueNames()
        text = "There are {} tissues with the names {}\n".format(len(self.allFolderContents), tissueNames)
        for folderContent in self.allFolderContents:
             text += str(folderContent)
        return text

    def GetDataOfTimePointWithKey(self, timePoint, complexityKey, pooled=False):
        contentsOfTimePoint = self.GetFolderContentsOfTimePoint(timePoint)
        dataOfTimePoint = []
        for content in contentsOfTimePoint:
            data = content.GetExtractedFilesKeyValues(complexityKey)
            if pooled and type(data) == dict:
                data = list(data.values())
                if isinstance(data[0], (list, np.ndarray)):
                    data = np.concatenate(data)
            dataOfTimePoint.append(data)
        if pooled:
            dataOfTimePoint = np.concatenate(dataOfTimePoint)
        return dataOfTimePoint

    def GetDataWithKey(self, complexityKey, pooled=True):
        allData = []
        for content in self.allFolderContents:
            data = content.GetExtractedFilesKeyValues(complexityKey)
            if pooled and type(data) == dict:
                data = list(data.values())
                if isinstance(data[0], (list, np.ndarray)):
                    data = np.concatenate(data)
            allData.append(data)
        if pooled:
            allData = np.concatenate(allData)
        return allData

    def GetFolderContentByIdx(self, idx):
        return self.allFolderContents[idx]

    def GetFolderContentOfIdentifier(self, identifier):
        genotype, selectedReplicateId, timePoint = identifier
        isSelectedTissue = self.isFolderContentIdentifier(identifier)
        assert np.sum(isSelectedTissue) > 0, f"The number of tissues corresponding to {genotype=}, {selectedReplicateId=} and {timePoint=} should be 1 == {np.sum(isSelectedTissue)}.\nThe following identifiers are present {[f.GetTissueName() for f in self.allFolderContents]}"
        if np.sum(isSelectedTissue) > 1:
            warnings.warn(f"The number of tissues corresponding to {genotype=}, {selectedReplicateId=} and {timePoint=} should be 1 == {np.sum(isSelectedTissue)=}.\nThe first folder content is returned.\nThe following identifiers are present {[f.GetTissueName() for f in self.allFolderContents]}")
        return self.allFolderContents[np.where(isSelectedTissue)[0]][0]

    def GetFolderContentOfReplicateAtTimePoint(self, selectedReplicateId, timePoint):
        isSelectedReplicateAtTimePoint = self.isFolderContentReplicateAndTimePointId(selectedReplicateId, timePoint)
        assert np.sum(isSelectedReplicateAtTimePoint) == 1, f"The number of tissues corresponding to {selectedReplicateId=} and {timePoint=} should be 1 != {np.sum(isSelectedReplicateAtTimePoint)=}.\nThe following replicate-time point pairs are present {self.GetReplicateTimePointPairs()}"
        return self.allFolderContents[np.where(isSelectedReplicateAtTimePoint)[0]][0]

    def GetFolderContentsOfGenotype(self, genotypeName):
        genotypeOfFolderContents = self.GetGenotypes()
        isSelectedGenotype = np.isin(genotypeOfFolderContents, genotypeName)
        rawFolderContents = [i.GetFolderContent() for i in self.allFolderContents[isSelectedGenotype]]
        return MultiFolderContent(rawFolderContents=rawFolderContents)

    def GetFolderContentsOfReplicate(self, selectedReplicateId):
        replicateIds = [i.GetReplicateId() for i in self.allFolderContents]
        isSelectedReplicate = np.isin(replicateIds, selectedReplicateId)
        return self.allFolderContents[np.where(isSelectedReplicate)[0]]

    def GetFolderContentsOfTimePoint(self, timePoint):
        timePointIds = [i.GetTimePoint() for i in self.allFolderContents]
        isSelectedTimePoint = np.isin(timePointIds, timePoint)
        return self.allFolderContents[np.where(isSelectedTimePoint)[0]]

    def GetFolderContentWithTissueName(self, selectedTissueName):
        tissueNames = self.GetTissueNames()
        isSelectedTissueName = np.isin(tissueNames, selectedTissueName)
        if np.sum(isSelectedTissueName) > 1:
            warnings.warn(f"The {selectedTissueName=} have been found multiple times ({np.sum(isSelectedTissueName)}) in the folder contents from {self.allFolderContentsFilename}")
        if np.sum(isSelectedTissueName) > 0:
            return self.allFolderContents[np.where(isSelectedTissueName)[0]][0]
        return None

    def GetGenotypes(self):
        genotypeOfFolderContents = [i.GetGenotype() for i in self.allFolderContents]
        return genotypeOfFolderContents

    def GetPresentTissueIdentifiers(self):
        if self.allFolderContents is None:
            return []
        else:
            return [folderContent.GetTissueInfos() for folderContent in self.allFolderContents]

    def GetReplicateTimePointPairs(self, delimiter=" || "):
        allReplicateTimePointPairs = []
        for folderContent in self.allFolderContents:
            replicateId = folderContent.GetReplicateId()
            timePoint = folderContent.GetTimePoint()
            replicateTimePointPair = delimiter.join([replicateId, timePoint])
            allReplicateTimePointPairs.append(replicateTimePointPair)
        return allReplicateTimePointPairs

    def GetTissueWithData(self, targetFolderContentData: dict, allowMultipleToReturn: bool = False, warnAboutMultiple: bool = True):
        indexSelectedOfFolderContent = []
        for i, folderContent in enumerate(list(self)):
            isSelectedTissue = folderContent.IsPresentFolderContentDataTheSame(targetFolderContentData)
            if isSelectedTissue:
                indexSelectedOfFolderContent.append(i)
        if len(indexSelectedOfFolderContent) == 0:
            return None
        elif len(indexSelectedOfFolderContent) == 1:
            return list(self)[indexSelectedOfFolderContent[0]]
        else:
            if allowMultipleToReturn:
                return self.allFolderContents[indexSelectedOfFolderContent]
            else:
                if warnAboutMultiple:
                    print(f"Multiple entries where found while checking the folder content data for {targetFolderContentData}\nwhile searching the MultiFolderContent {self.allFolderContentsFilename}")
                return list(self)[indexSelectedOfFolderContent[0]]

    def GetTidyDataFrameOf(self, valueKeysToInclude, includeCellId=False):
        baseColumns = ["genotype", "replicate id", "time point"]
        if not isinstance(valueKeysToInclude, (list, np.ndarray, tuple)):
            valueKeysToInclude = list(valueKeysToInclude)
        sampleFolderContent = self.allFolderContents[0]
        isValueDictType = [isinstance(sampleFolderContent.GetExtractedFilesKeyValues(key), dict) for key in valueKeysToInclude]
        unwrapDictTyp = np.any(isValueDictType)
        if unwrapDictTyp:
            assert np.all(isValueDictType), "If any value key is of type dict, all have to be of type dict, {} are dicts, but {} are not".format(np.array(valueKeysToInclude)[isValueDictType], np.array(valueKeysToInclude)[np.invert(isValueDictType)])
            baseColumns.append("cell label")
            if includeCellId:
                self.myCellIdTracker = CellIdTracker()
                baseColumns.append("cell id")
        columns = np.concatenate([baseColumns, valueKeysToInclude])
        tidyArray = []
        unwrapDictTypCheckMessages = []
        for content in self.allFolderContents:
            valuesOfKeys = []
            cellLabelsOfValueKey = {}
            for valueKey in valueKeysToInclude:
                if unwrapDictTyp:
                    valuesOfCurrentKey, cellLabelsOfSamples, cellIdsOfSamples = self.extractCellValueAndLabelOf(valueKey, content, includeCellId=includeCellId)
                    valuesOfKeys.append(valuesOfCurrentKey)
                    cellLabelsOfValueKey[valueKey] = cellLabelsOfSamples
                else:
                    valuesOfKeys.append(content.GetExtractedFilesKeyValues(valueKey))
                    cellIdsOfSamples = None
            if unwrapDictTyp:
                msg = self.assertEqualSampleNumbersOfAllKeys(valuesOfKeys, valueKeysToInclude, cellLabelsOfValueKey, content)
                if not msg is None:
                    unwrapDictTypCheckMessages.append(msg)
            else:
                msg = None
            if msg is None:
                tidyArrayOfContent = self.combineValues(content, valuesOfKeys, unwrapDictTyp, cellLabelsOfSamples, includeCellId, cellIdsOfSamples)
                tidyArray.append(tidyArrayOfContent)
        assert len(unwrapDictTypCheckMessages) == 0, "\n\n".join(unwrapDictTypCheckMessages)
        tidyArray = np.vstack(tidyArray)
        tidyDf = pd.DataFrame(tidyArray, columns=columns)
        tidyDf["cell label"] = tidyDf["cell label"].astype(np.int32) # this should be done as the table could not be merged with others, but may lead to later confusions, when one does not want to have cell labels to be integers
        return tidyDf

    def GetTimePoints(self):
        timePointIds = [i.GetTimePoint() for i in self.allFolderContents]
        return timePointIds

    def GetTissueNames(self):
        return [i.GetTissueName() for i in self.allFolderContents]

    def GetUniqueTimePoints(self):
        timePointIds = self.GetTimePoints()
        _, previousOrder = np.unique(timePointIds, return_index=True)
        return np.array(timePointIds)[np.sort(previousOrder)]

    def GetUniqueGenotypes(self):
        genotypeOfFolderContents = self.GetGenotypes()
        _, previousOrder = np.unique(genotypeOfFolderContents, return_index=True)
        return np.array(genotypeOfFolderContents)[np.sort(previousOrder)]

    def SetAllFolderContentsFilename(self, allFolderContentsFilename):
        self.allFolderContentsFilename = allFolderContentsFilename

    def ResetFolderContents(self):
        self.rawFolderContents = []
        self.allFolderContents = np.asarray([], dtype=FolderContent)

    def ConvertRawToProcessedFolderContents(self, rawFolderContents: list or np.ndarray):
        typeOfRawFolderContent = [type(i) for i in rawFolderContents]
        isAllowedType = np.isin(typeOfRawFolderContent, [dict, FolderContent])
        assert np.all(isAllowedType), f"The raw folder contents need to be of type 'dict' or 'FolderContent', but are of type {typeOfRawFolderContent}"
        allFolderContents = np.array([i if type(i) == FolderContent else FolderContent(i) for i in rawFolderContents])
        return allFolderContents

    def LoadFolderContentsFromFilename(self, allFolderContentsFilename: str):
        if Path(allFolderContentsFilename).is_file():
            self.allFolderContentsFilename = allFolderContentsFilename
            self.rawFolderContents = FolderContent().loadFile(allFolderContentsFilename, convertDictKeysToInt=False, convertDictValuesToNpArray=False, convertNestedDictKeysToInt=False)
            self.allFolderContents = self.ConvertRawToProcessedFolderContents(self.rawFolderContents)
        else:
            print("There was no file under {}, so MultiFolderContent was initialised empty.".format(allFolderContentsFilename))
            self.ResetFolderContents()
            if self.verbose >= 1:
                    pass

    def AddDataFromFilenameContainingMultipleDicts(self, extractDataFromFilenameUsingKey, returnIndividualKeysAdded=False, nonNestedKeys: list = []):
        individualDataKeys = []
        for f in self.allFolderContents:
            filenameDict = f.GetFilenameDict()
            if extractDataFromFilenameUsingKey in filenameDict:
                dataFilename = filenameDict[extractDataFromFilenameUsingKey]
                assert Path(dataFilename).is_file(), f"While trying to load the key {extractDataFromFilenameUsingKey}, the file {dataFilename} was not present."
                suffix = Path(dataFilename).suffix
                if suffix == ".json":
                    with open(dataFilename, "r") as fh:
                        data = json.load(fh)
                else:
                    with open(dataFilename, "rb") as fh:
                        data = pickle.load(fh)
                if extractDataFromFilenameUsingKey in nonNestedKeys:
                    data = self.tryToConvertKeysToInts(data)
                    f.AddDataToExtractedFilesDict(data=data, key=extractDataFromFilenameUsingKey)
                    if returnIndividualKeysAdded:
                        individualDataKeys.append(extractDataFromFilenameUsingKey)
                else:
                    for dataKey, dataValues in data.items():
                        if suffix == ".json" and type(dataValues) == dict:
                            # this needs to be done as dictionary keys get converted to strings even if they are cell labels, which should be integer values
                            dataValues = self.tryToConvertKeysToInts(dataValues)
                        f.AddDataToExtractedFilesDict(data=dataValues, key=dataKey)
                    if returnIndividualKeysAdded:
                        individualDataKeys.extend(list(data.keys()))
            else:
                print(f"The filename key {extractDataFromFilenameUsingKey} was not found for {f.GetTissueName()}, only the following filename keys are present: {list(filenameDict.keys())}")
        if returnIndividualKeysAdded:
            unorderedUniqueKeys, counts = np.unique(individualDataKeys, return_counts=True)
            isCountEqualToFirst = counts[0] == counts
            assert np.all(isCountEqualToFirst), f"The key/s {unorderedUniqueKeys[np.invert(isCountEqualToFirst)]} don't appear as often as the first key {counts[np.invert(isCountEqualToFirst)]} != {counts[0]} of key {unorderedUniqueKeys[0]}, while extracting data from {extractDataFromFilenameUsingKey} of {self.allFolderContentsFilename}"
            return pd.unique(individualDataKeys).tolist()

    def tryToConvertKeysToInts(self, data, additionalInfoWithConversionError: str = ""):
        try:
            tmpDict = {}
            for k, v in data.items():
                convertedKey = int(float(k))
                tmpDict[convertedKey] = data[k]
            return tmpDict
        except ValueError as e:
            print("ValueError caught during key to int conversion:", e)
            if additionalInfoWithConversionError:
                print(f"{additionalInfoWithConversionError=}")
            return data

    def AddDataFromFilename(self, dataKey, extractDataFromFilenameUsingKey="regularityMeasuresFilename"):
        for f in self.allFolderContents:
            filenameDict = f.GetFilenameDict()
            if extractDataFromFilenameUsingKey in filenameDict:
                dataFilename = filenameDict[extractDataFromFilenameUsingKey]
                with open(dataFilename, "rb") as fh:
                    data = pickle.load(fh)
                assert dataKey in data, f"The {dataKey=} is not present as a key of the data from {dataFilename=} use one of the keys present: {list(data.values())}"
                dataValues = data[dataKey]
                f.AddDataToExtractedFilesDict(data=dataValues, key=dataKey)

    def AddDataFromSimpleDictsUsingFilenameKey(self, filenameKeyToLoad, **kwargs):
        for f in self.allFolderContents:
            data = f.LoadKeyUsingFilenameDict(filenameKeyToLoad, **kwargs)
            f.AddDataToExtractedFilesDict(data, filenameKeyToLoad)

    def AddMultipleDataValuesFromFilename(self, dataKeys, extractDataFromFilenameUsingKey="regularityMeasuresFilename"):
        for key in dataKeys:
            self.AddDataFromFilename(key, extractDataFromFilenameUsingKey)

    def AppendFolderContent(self, folderContent):
        if type(folderContent) == dict:
            self.rawFolderContents.append(folderContent)
            self.allFolderContents = np.append(self.allFolderContents, FolderContent(folderContent))
        elif type(folderContent) == FolderContent:
            self.rawFolderContents.append(folderContent.folderContent)
            self.allFolderContents = np.append(self.allFolderContents, folderContent)
        else:
            print("No folder content is appended as the type {} is neither dict, nor FolderContent".format(type(folderContent)))

    def IsTissuePresent(self, selectedReplicateId, timePoint):
        isSelectedReplicateAtTimePoint = self.isFolderContentReplicateAndTimePointId(selectedReplicateId, timePoint)
        numberOfSelectedTissues = np.sum(isSelectedReplicateAtTimePoint)
        assert numberOfSelectedTissues <= 1, f"The number of tissues corresponding to {selectedReplicateId=} and {timePoint=} should be 1 <= {numberOfSelectedTissues=}.\nThe following replicate-time point pairs are present {self.GetReplicateTimePointPairs()}"
        return numberOfSelectedTissues > 0

    def IsTissueWithInfoPresent(self, genotype, replicateId, timePoint):
        presentTissueIdentifiers = np.array(self.GetPresentTissueIdentifiers())
        selectedTissueInfo = np.array([genotype, replicateId, timePoint])
        sumOfMatchingInfos = np.sum(presentTissueIdentifiers == selectedTissueInfo, axis=1)
        return np.any(sumOfMatchingInfos == 3)

    def ExchangeFolderContent(self, folderContent):
        currentContentIdentifier = folderContent.GetTissueInfos()
        isContent = self.isFolderContentIdentifier(currentContentIdentifier)
        if np.any(isContent):
            idxOfContent = np.where(isContent)[0][0]
            self.rawFolderContents[idxOfContent] = folderContent.folderContent
            self.allFolderContents[idxOfContent] = folderContent
        else:
            self.AppendFolderContent(folderContent)

    def UpdateFolderContents(self, printOut=False):
        if not self.allFolderContentsFilename is None:
            if isinstance(self.allFolderContentsFilename, Path):
                temporarySaveFilename = self.allFolderContentsFilename.parent.joinpath(self.allFolderContentsFilename.stem + "_tmp.pkl")
            else:
                temporarySaveFilename = self.allFolderContentsFilename[:-4] + "_tmp.pkl"
            if self.verbose >= 2:
                print("Updating folder content of {}.".format(self.allFolderContentsFilename))
            self.SaveFolderContents(temporarySaveFilename)
            self.SaveFolderContents(self.allFolderContentsFilename)
            os.remove(temporarySaveFilename)
            if printOut:
                print(f"Updated and overwrote {self.allFolderContentsFilename}")
        else:
            if self.verbose >= 1:
                print("Could not update MultiFolderContent of tissues {} as self.allFolderContentsFilename is None.".format([i.GetTissueName() for i in self.allFolderContents]))

    def RemoveFolderContentAt(self, idx):
         self.allFolderContents = np.delete(self.allFolderContents, idx)
         if isinstance(self.rawFolderContents, list):
             self.rawFolderContents.pop(idx)
         else:
             self.rawFolderContents = np.delete(self.rawFolderContents, idx)

    def ResetFolderContents(self):
        self.rawFolderContents = []
        self.allFolderContents = np.asarray([], dtype=FolderContent)

    def SaveFolderContents(self, saveToFilename: str or Path):
        if len(self.rawFolderContents) > 0:
            implementedSuffixes = (".pkl", ".json")
            suffix = Path(saveToFilename).suffix
            assert suffix in implementedSuffixes, f"The {suffix=} is not present in the implemented suffixes {implementedSuffixes} for the filename {saveToFilename}"
            if suffix == ".pkl":
                with open(saveToFilename, "wb") as fh:
                    pickle.dump(self.rawFolderContents, fh)
            elif suffix == ".json":
                self.SaveFolderContentsAsJson(saveToFilename)
        else:
            print(f"The multiFolderContent is empty and could not be saved under {saveToFilename}. It originating from {self.allFolderContentsFilename=}")

    def SaveFolderContentsAsJson(self, saveToFilename: str or Path):
        rawContentsToSave = []
        dummyContent = FolderContent()
        for i, rawContent in enumerate(self.rawFolderContents):
            jsonCompatibleFolderContent = None
            if isinstance(rawContent, dict):
                jsonCompatibleFolderContent = dummyContent.ensureSavabilityAsJson(rawContent)
            elif isinstance(rawContent, FolderContent):
                jsonCompatibleFolderContent = dummyContent.ensureSavabilityAsJson(rawContent.GetFolderContent())
            else:
                print(f"Could not save {i}th folder content as it is not of type dict or FolderContent != {type(rawContent)}\nrawContent:\n{rawContent}")
            if jsonCompatibleFolderContent is not None:
                rawContentsToSave.append(jsonCompatibleFolderContent)
        dummyContent.prettyDumpJson(rawContentsToSave, saveToFilename)

    def extractCellValueAndLabelOf(self, valueKey, content, includeCellId=False):
        if includeCellId:
            self.myCellIdTracker.RunCellIdTracker(folderContent=content)
            labelsToIdOfTissue = self.myCellIdTracker.GetLabelsToIdsDict()
        cellLabelsOfSamples, cellIdsOfSamples = [], []
        valuesOfCurrentKey = []
        for cellLabels, values in content.GetExtractedFilesKeyValues(valueKey).items():
            try:
                # use this when only single cell label is given, but multiple values (should implement better check)
                valuesOfCurrentKey.extend(list(values))
                cellLabelsOfSamples.extend([cellLabels] * len(values))
                if includeCellId:
                    assert cellLabels in labelsToIdOfTissue, f"The cellLabel={cellLabels} is not present in the conversion dictionary to get cell ids, {labelsToIdOfTissue=}"
                    cellIdsOfSamples.extend([labelsToIdOfTissue[cellLabels]] * len(values))
            except TypeError as e:
                valuesOfCurrentKey.append(values)
                cellLabelsOfSamples.append(cellLabels)
                if includeCellId:
                    assert cellLabels in labelsToIdOfTissue, f"The cellLabel={cellLabels} is not present in the conversion dictionary to get cell ids, {labelsToIdOfTissue=}"
                    cellIdsOfSamples.append(labelsToIdOfTissue[cellLabels])
        return valuesOfCurrentKey, cellLabelsOfSamples, cellIdsOfSamples

    def combineValues(self, content, valuesOfKeys, unwrapDictTyp, cellLabelsOfSamples, includeCellId=False, cellIdsOfSamples=None):
        valuesOfKeys = np.asarray(valuesOfKeys).T
        attributesOfSample = np.array([content.GetGenotype(), content.GetReplicateId(), content.GetTimePoint()])
        attributesOfSample = attributesOfSample.reshape(1, len(attributesOfSample))
        nrOfSegments = valuesOfKeys.shape[0]
        attributesOfSample = np.repeat(attributesOfSample, nrOfSegments, axis=0)
        if unwrapDictTyp:
            cellLabelsOfSamples = np.array(cellLabelsOfSamples).reshape(len(cellLabelsOfSamples), 1)
            attributesOfSample = np.hstack([attributesOfSample, cellLabelsOfSamples])
            if includeCellId:
                cellIdsOfSamples = np.array(cellIdsOfSamples).reshape(len(cellIdsOfSamples), 1)
                attributesOfSample = np.hstack([attributesOfSample, cellIdsOfSamples])
        tidyArrayOfContent = np.hstack([attributesOfSample, valuesOfKeys])
        return tidyArrayOfContent

    def isFolderContentReplicateAndTimePointId(self, selectedReplicateId, timePoint):
        replicateIds = [i.GetReplicateId() for i in self.allFolderContents]
        isSelectedReplicate = np.isin(replicateIds, selectedReplicateId)
        timePointIds = [i.GetTimePoint() for i in self.allFolderContents]
        isSelectedTimePoint = np.isin(timePointIds, timePoint)
        isSelectedReplicateAtTimePoint = isSelectedReplicate & isSelectedTimePoint
        return isSelectedReplicateAtTimePoint

    def isFolderContentIdentifier(self, identifier):
        genotype, selectedReplicateId, timePoint = identifier
        genotypeOfFolderContents = self.GetGenotypes()
        isSelectedGenotype = np.isin(genotypeOfFolderContents, genotype)
        isSelectedReplicateAtTimePoint = self.isFolderContentReplicateAndTimePointId(selectedReplicateId, timePoint)
        isTissueSelected = isSelectedGenotype & isSelectedReplicateAtTimePoint
        return isTissueSelected

    def assertEqualSampleNumbersOfAllKeys(self, valuesOfKeys, valueKeysToInclude, cellLabelsOfValueKey, content):
        nrOfSamples = np.asarray([len(i) for i in valuesOfKeys])
        expectedSampleNumber = np.median(nrOfSamples)
        isSampleNumberDifferent = nrOfSamples != expectedSampleNumber
        differentKeyNames = np.asarray(valueKeysToInclude)[isSampleNumberDifferent]
        if np.any(isSampleNumberDifferent):
            expectedSampleLabels = cellLabelsOfValueKey[np.asarray(valueKeysToInclude)[np.invert(isSampleNumberDifferent)][0]]
            missingSampleLabels = [np.setdiff1d(expectedSampleLabels, cellLabelsOfValueKey[keyName]) for keyName in differentKeyNames]
            toManySampleLabels = [np.setdiff1d(cellLabelsOfValueKey[keyName], expectedSampleLabels) for keyName in differentKeyNames]
            return f"The given number of samples {nrOfSamples} is not identical for the keys {differentKeyNames} with {nrOfSamples[isSampleNumberDifferent]} samples in the tissue {content.GetTissueName()}.\nThe samples {missingSampleLabels} are missing and {toManySampleLabels} should not be present."
        return None

def convertFilenameDictPathsToStrings(allFolderContentsFilename):
    myMultiFolderContent = MultiFolderContent(allFolderContentsFilename)
    for folderContent in myMultiFolderContent:
        filenameDict = folderContent.GetFilenameDict()
        for key, value in filenameDict.items():
            filenameDict[key] = str(value)
        folderContent.SetFilenameDict(filenameDict)
    myMultiFolderContent.UpdateFolderContents()

def main():
    allFolderContentsFilename = "Images/Eng2021Cotyledons/Eng2021Cotyledons.json"
    myMultiFolderContent = MultiFolderContent(allFolderContentsFilename)
    print(myMultiFolderContent)
    selectedReplicateId = "20170501 WT S2"
    print(myMultiFolderContent.GetFolderContentsOfReplicate(selectedReplicateId))

def mainCreateAndAddAdjacencyList():
    import json
    import networkx as nx
    import sys
    sys.path.insert(0, "./Code/SAM_division_prediction/")
    from GraphCreatorFromAdjacencyList import GraphCreatorFromAdjacencyList
    adjacencyListNameExtension = "{}{}_adjacencyList.json"
    adjacencyListFilenameKey = "labelledImageAdjacencyList"
    graphBaseName = "cellularConnectivityNetwork{}{}.csv"
    allFolderContentsFilename = "Images/Matz2022SAM.json"
    baseFolder = "Images/Matz2022SAM/"
    multiFolderContent = MultiFolderContent(allFolderContentsFilename)
    for folderContent in multiFolderContent:
        scenarioName, replicateName, timePoint = folderContent.GetTissueInfos()
        tissueBaseFolder = f"{baseFolder}{replicateName}/"
        graphFilename = tissueBaseFolder + graphBaseName.format(replicateName, timePoint)
        filenameToSave = tissueBaseFolder + adjacencyListNameExtension.format(replicateName, timePoint)
        graph = GraphCreatorFromAdjacencyList(graphFilename).GetGraph()
        wrongDtypeAdjacencyListDict = nx.to_dict_of_lists(graph)
        adjacencyListDict = {}
        for cellLabel, neighbors in wrongDtypeAdjacencyListDict.items():
            adjacencyListDict[int(cellLabel)] = [int(n) for n in neighbors]
        with open(filenameToSave, "w") as fh:
            json.dump(adjacencyListDict, fh)
        folderContent.AddDataToFilenameDict(filenameToSave, adjacencyListFilenameKey)
    multiFolderContent.UpdateFolderContents()
    print(multiFolderContent)

def mainRemoveDuplicateMultiFolderContents(folderContensFilename="Images/full cotyledons/full cotyledons.pkl", previousVersionSuffix="_beforeRemovingDuplicates.pkl"):
    multiFolderContent = MultiFolderContent(folderContensFilename)
    previousVersionFilename = Path(folderContensFilename).with_name(Path(folderContensFilename).stem + previousVersionSuffix)
    multiFolderContent.SaveFolderContents(previousVersionFilename)
    allTissueNames, duplicatContentIndices = [], []
    for i, folderContent in enumerate(multiFolderContent):
        tissueName = folderContent.GetTissueName()
        if tissueName in allTissueNames:
            duplicatContentIndices.append(i)
        allTissueNames.append(tissueName)
    print(duplicatContentIndices)
    for idx in duplicatContentIndices[::-1]:
        multiFolderContent.RemoveFolderContentAt(idx)
    multiFolderContent.UpdateFolderContents()

def convertPickleToHumanReadableJsonMultiFolderContent(folderContentFilenamesToChange: list):
    for folderContentsFilename in folderContentFilenamesToChange:
        multiFolderContent = MultiFolderContent(folderContentsFilename)
        filenameToSave = folderContentsFilename.replace(".pkl", ".json")
        multiFolderContent.SaveFolderContents(filenameToSave)

if __name__ == '__main__':
    # main()
    # mainRemoveDuplicateMultiFolderContents()
    convertPickleToHumanReadableJsonMultiFolderContent(folderContentFilenamesToChange=["Images/Eng2021Cotyledons/Eng2021Cotyledons.pkl", "Images/Matz2022SAM/Matz2022SAM.pkl", "Images/Smit2023Cotyledons/Smit2023Cotyledons.pkl"])