import json
import numpy as np
import pandas as pd
import pickle
import platform
import skimage.io

from MyEncoder import MyEncoder, NoIndent
from pathlib import Path

class FolderContent (object):

    verbose = 1
    resolutionKey="resolution"

    def __init__(self, folderContent: dict = {}):
        self.folderContent = folderContent
        if not "filenameDict" in self.folderContent:
            self.folderContent["filenameDict"] = {}
        if not "extractedFilesDict" in self.folderContent:
            self.folderContent["extractedFilesDict"] = {}

    def combineNames(self, seperator="", addSeperatorAtEnd=False):
        namesToConcat = [self.folderContent["genotype"],
                         self.folderContent["replicateId"],
                         self.folderContent["timePoint"]]
        combinedName = seperator.join([str(i) for i in namesToConcat])
        if addSeperatorAtEnd:
            combinedName += seperator
        return combinedName

    def AddDataToExtractedFilesDict(self, data, key):
        self.folderContent["extractedFilesDict"][key] = data

    def AddDataToFilenameDict(self, data, key):
        if key in self.folderContent["filenameDict"] and self.verbose >= 1:
            print("Overwriting {} of key {} in filenameDict to {}".format(self.folderContent["filenameDict"][key], key, data))
        self.folderContent["filenameDict"][key] = data

    def IsKeyInExtractedFilesDict(self, key):
        return key in self.folderContent["extractedFilesDict"]

    def IsKeyInFilenameDict(self, key):
        return key in self.folderContent["filenameDict"]

    def LoadKeyUsingFilenameDict(self, key, **kwargs):
        assert key in self.folderContent["filenameDict"], "The key {} is not in the filenameDict of {} only the keys: {} are present.".format(key, self.GetTissueName(), list(self.folderContent["filenameDict"].keys()))
        return self.loadFile(self.folderContent["filenameDict"][key], **kwargs)

    def LoadKeysCalculatingRatioUsingFilenameDict(self, filenameKey, numeratorSubKey, denominatorSubKey, warnAboutMissingLabels=True, **kwargs):
        multiValuesOfDict = self.LoadKeyUsingFilenameDict(filenameKey, **kwargs)
        isNumeratorKeyPresent = numeratorSubKey in multiValuesOfDict
        isDenominatorKeyPresent = denominatorSubKey in multiValuesOfDict
        assert isNumeratorKeyPresent and isDenominatorKeyPresent, f"Both numerator and denominator sub keys need to be present {isNumeratorKeyPresent=} and {isDenominatorKeyPresent=}, with {filenameKey=}, {numeratorSubKey=}, {denominatorSubKey=}"
        ratioValuesDict = {}
        numeratorValuesOfCells = multiValuesOfDict[numeratorSubKey]
        denominatorValuesOfCells = multiValuesOfDict[denominatorSubKey]
        for numeratorCellLabel, numeratorCellValue in numeratorValuesOfCells.items():
            if numeratorCellLabel in denominatorValuesOfCells:
                denominatorCellValue = denominatorValuesOfCells[numeratorCellLabel]
                ratioValuesDict[numeratorCellLabel] = numeratorCellValue / denominatorCellValue
        if warnAboutMissingLabels:
            numberOfCellsWithRatios = len(ratioValuesDict)
            numberOfCellsWithNumerator = len(numeratorValuesOfCells)
            numberOfCellsWithDenominator = len(denominatorValuesOfCells)
            if numberOfCellsWithNumerator != numberOfCellsWithDenominator != numberOfCellsWithRatios:
                print(f"The {numberOfCellsWithNumerator=} != {numberOfCellsWithDenominator=} != {numberOfCellsWithRatios=}, with the present labels being\nnumeratorCellLabels={list(numeratorValuesOfCells.keys())}\ndenominatorCellLabels={list(denominatorValuesOfCells.keys())}\nratiosCellLabels={list(numberOfCellsWithRatios.keys())}")
        return ratioValuesDict

    def loadFile(self, filename, convertDictKeysToInt=False, convertNestedDictKeysToInt=False, convertDictValuesToNpArray=False, **kwargs):
        suffix = Path(filename).suffix
        if platform.system() == "Linux":
            filename = self.convertFilenameToLinux(filename)
        if suffix == ".csv":
            file = pd.read_csv(filename, engine="python", **kwargs)
        elif suffix == ".npy":
            file = np.load(filename)
        elif suffix == ".pkl" or suffix == ".pickle" or suffix == ".gpickle":
            file = pickle.load(open(filename, "rb"))
        elif suffix == ".png" or suffix == ".TIF" or suffix == ".tif":
            file = skimage.io.imread(filename, **kwargs)
        elif suffix == ".txt":
            with open(filename, "r") as fh:
                file = fh.readlines()
        elif suffix == ".json":
            with open(filename, "r") as fh:
                file = json.load(fh)
        else:
            raise NotImplementedError(f"The extension of the {suffix=} is not yet implemented. Aborting while loading {filename=}")
        if convertDictKeysToInt:
            tmpDict = {}
            for k, v in file.items():
                tmpDict[int(k)] = v
            file = tmpDict
        elif convertNestedDictKeysToInt:
            tmpDict = {}
            for kOuter, d in file.items():
                tmpInnerDict = {}
                for kInner, v in d.items():
                    tmpInnerDict[int(kInner)] = v
                tmpDict[kOuter] = tmpInnerDict
            file = tmpDict
        if convertDictValuesToNpArray:
            for k, v in file.items():
                file[k] = np.array(v)
        return file

    def convertFilenameToLinux(self, filename: str or Path):
        if isinstance(filename, str):
            if "\\" in filename:
                filename = filename.replace("\\", "/")
        else:
            filename = filename.as_posix()
        return filename

    def GetExtractedFilesDict(self):
        return self.folderContent["extractedFilesDict"]

    def GetExtractedFilesKeyValues(self, key):
        assert key in self.folderContent["extractedFilesDict"], "The key {} is not in the extractedFilesDict of {}".format(key, str(self))
        return self.folderContent["extractedFilesDict"][key]

    def GetFilenameDict(self):
        return self.folderContent["filenameDict"]

    def GetFilenameDictKeyValue(self, key):
        assert key in self.folderContent["filenameDict"], "The key {} is not in the filenameDict of {}".format(key, str(self))
        filename = self.folderContent["filenameDict"][key]
        if platform.system() == "Linux":
            filename = self.convertFilenameToLinux(filename)
        return filename

    def GetFolder(self):
        folderName = self.combineNames("/", addSeperatorAtEnd=True)
        return folderName

    def GetFolderContent(self):
        return self.folderContent

    def GetGenotype(self):
        return self.folderContent["genotype"]

    def GetReplicateId(self):
        return self.folderContent["replicateId"]

    def GetResolution(self):
        if self.resolutionKey in self.folderContent:
            return self.folderContent[self.resolutionKey]
        else:
            if self.verbose >= 1:
                print("You wanted to get the resolution of tissue {}, but the resolutionKey {} is not present in {}.".format(self.GetTissueName(), self.resolutionKey, list(self.folderContent.keys())))
            return None

    def GetSegmentList(self, segmentKey="segments"):
        extractedFilesDict = self.folderContent["extractedFilesDict"]
        assert segmentKey in extractedFilesDict, "The segment key {} is not present in the tissue {}".format(segmentListKey, self.GetTissueName())
        segmentList = extractedFilesDict[segmentKey]
        return segmentList

    def GetSegmentOfEndPointMode(self, endPointMode):
        extractedFilesDict = self.GetExtractedFilesDict()
        assert endPointMode in extractedFilesDict, "The endPointMode {} does not exist as a key in extractedFilesDict, only {} are allowed.".format(endPointMode, list(extractedFilesDict.keys()))
        return extractedFilesDict[endPointMode]

    def GetTimePoint(self):
        return self.folderContent["timePoint"]

    def GetTimePointIdxFrom(self, allTimePoints):
        isTimePoint = np.isin(allTimePoints, self.folderContent["timePoint"])
        whereIsTimePoint = np.where(isTimePoint)[0]
        if len(whereIsTimePoint) > 0:
            timePointIdx = whereIsTimePoint[0]
        else:
            timePointIdx = None
            if self.verbose >= 1:
                print("The time point of the tissue {} is not in the given time point list {} and is therefore given back as {}".format(self.GetTissueName(), additionallyReturnTimeIdxFromAllTimePointsList, timePointIdx))
        return timePointIdx

    def GetTissueInfos(self, additionallyReturnTimeIdxFromAllTimePointsList=None):
        genotype = self.GetGenotype()
        replicateId = self.GetReplicateId()
        timePoint = self.GetTimePoint()
        if not additionallyReturnTimeIdxFromAllTimePointsList is None:
            timePointIdx = self.GetTimePointIdxFrom(additionallyReturnTimeIdxFromAllTimePointsList)
            return genotype, replicateId, timePoint, timePointIdx
        return genotype, replicateId, timePoint

    def GetTissueName(self, sep="_"):
        tissueName = self.combineNames(sep)
        return tissueName

    def SavePartOfExtractedFilesTo(self, keys, saveToFilename):
        if type(keys) == "str":
            keys = [keys]
        dataToSave = {}
        isKeyPresent = [k in self.extractedFilesDict for k in keys]
        assert np.all(isKeyPresent), f"The keys {np.asarray(keys)[isKeyPresent]} are not present in the extractedFilesDict, only {list(self.extractedFilesDict.keys())} are present as keys."
        for k in keys:
            dataToSave[k] = self.extractedFilesDict[k]
        with open(saveToFilename, "wb") as fh:
            pickle.dump(dataToSave, fh)

    def SaveDataFilesTo(self, dataToSave, saveToFilename: str, convertDictValuesToList: bool = False, prettyDumpJson: bool = True):
        implementedSuffixes = (".pkl", ".json")
        suffix = Path(saveToFilename).suffix
        assert suffix in implementedSuffixes, f"The {suffix=} is not present in the implemented suffixes {implementedSuffixes} for the filename {saveToFilename}"
        if convertDictValuesToList:
            for key, value in dataToSave.items():
                if type(value) == np.ndarray:
                    dataToSave[key] = value.tolist()
        if suffix == ".pkl":
            with open(saveToFilename, "wb") as fh:
                pickle.dump(dataToSave, fh)
        elif suffix == ".json":
            dataToSave = self.ensureSavabilityAsJson(dataToSave)
            if prettyDumpJson:
                self.prettyDumpJson(dataToSave, saveToFilename)
            else:
                with open(saveToFilename, "w") as fh:
                    json.dump(dataToSave, fh)

    def ensureSavabilityAsJson(self, dataToSave, recursiveLayer: int = 0, maxRecursiveLayer: int = 5):
        if isinstance(dataToSave, dict):
            for k, v in dataToSave.items():
                dataToSave[k] = self.ensureSavabilityAsJson(v, recursiveLayer+1)
        elif isinstance(dataToSave, (list, tuple)):
            for i, v in enumerate(dataToSave):
                dataToSave[i] = self.ensureSavabilityAsJson(v, recursiveLayer+1)
        elif isinstance(dataToSave, np.ndarray):
            dataToSave = dataToSave.tolist()
            dataToSave = self.ensureSavabilityAsJson(dataToSave, recursiveLayer+1)
        return dataToSave

    def prettyDumpJson(self, obj: dict, filename: str, indent: int = 2, omitWarning: bool = False):
        if not isinstance(obj, dict):
            if not omitWarning:
                print(f"Warning: Pretty dumping json file with {filename=} is of type {type(dict)} != dict and therefore not made more pretty.")
            with open(filename, "w") as f:
                json.dump(obj, f)
        else:
            data_structure = self.recursiveDictNotImplementationDecision(obj, depth=0)
            prettyJsonFormattedObject = json.dumps(data_structure, cls=MyEncoder, indent=indent)
            with open(filename, "w") as f:
                f.write(prettyJsonFormattedObject)

    def recursiveDictNotImplementationDecision(self, obj, depth: int, maxDepth: int = 2):
        data_structure = {}
        for key, value in obj.items():
            if isinstance(value, dict):
                if depth < maxDepth:
                    data_structure[key] = self.recursiveDictNotImplementationDecision(value, depth=depth + 1, maxDepth=maxDepth)
                else:
                    data_structure[key] = NoIndent(value)
            else:
                data_structure[key] = NoIndent(value)
        return data_structure

    def SetFilenameDict(self, filenameDict):
        self.folderContent["filenameDict"] = filenameDict

    def __str__(self):
        text = "The content of the tissue {}\n".format(self.GetTissueName())
        text += "contains the keys {} and\n".format(list(self.folderContent.keys()))
        text += "the extractedFilesDict contains the keys {}\n".format(list(self.GetExtractedFilesDict().keys()))
        text += "the filenameDict contains the names {}\n".format(self.GetFilenameDict())
        return text
